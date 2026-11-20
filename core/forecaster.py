"""Island-width forecaster and the controllers built on it, for the simulated reactor (H35).

Training data: synthetic shots from the simulated reactor (their generator is not part of this release). One sample per step t and horizon k (1, 4, 8 steps:
0.5, 2, 4 s). Inputs, all things a real plant measures or sets: island width w_t, its change over the last step, the
bootstrap current density at the surface (MA/m^2), the surface position (rho_norm), and over the next k steps the mean
ECCD power (MW) and mean misalignment |aim - surface| (cm). Target: w_(t+k) - w_t (cm). The width inputs get Gaussian
noise of up to 1 cm in training, as the tracking controllers see. One small network per horizon (6 inputs, two hidden
layers of 32 ReLU units), Adam, inputs and targets standardized. Shots with number >= --test_from are held out.
--arch module: the same inputs and about the same number of weights, split the way the island equation is: change = plain(w, dw,
j_bs, rho) + P / 10 MW * effect(w, j_bs, rho, misalignment), two networks of two hidden layers of 22 (input masks as in H30d: the
plasma module cannot see the drive, the drive module cannot see the recent change). --clean drops windows with a seed event inside
the horizon: seeds cannot be forecast, and fitting their mean adds an effect of the drive that is not there.
NetMPC: after every step, try powers {0, P_max / 2, P_max} and aims {surface - 1 cm, surface, surface + 1 cm}; predict
the width 2 s ahead; take the command with the smallest predicted width plus energy_cost times power (cm per MW).
ChipMPC: the same with the network evaluated in fixed point (16-bit words, 10 fractional bits, every product and sum
rounded and saturated), as the companion chip project's chip would compute it. With chip="hgq" the forecaster is instead
the HGQ2 network trained and exported through that project's hardware flow (chip_forecaster.py; load_model(hgq=...)),
evaluated with its own learned word widths, bit for bit as its Verilog computes. A list of models is an ensemble: the
mean prediction.
Usage (TORAX's venv): python core/forecaster.py --shots runs/torax_shots
"""
import argparse
import collections
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr

import chip_forecaster as cf

HORIZONS = (1, 4, 8)
NAMES = ["w_cm", "dw_cm", "j_bs_MA", "rho_s", "P_MW", "mis_cm"]
PLAIN, EFFECT = np.array([0, 1, 2, 3]), np.array([0, 2, 3, 5])   # the module forecaster's input masks
P_SCALE = 10.0


def load_shot(path):
    """A shot's island traces, from <shot>_islands.npz (written by make_shots; made here once for older shots: reading
    netCDF groups takes seconds per file on the laptop)."""
    cache = path.replace(".nc", "_islands.npz")
    if not os.path.exists(cache):
        nc = xr.open_dataset(path, group="islands")
        np.savez(cache, time=nc["time"].values, **{k: nc[k].values for k in nc.data_vars})
    ds = np.load(cache)
    g = lambda k: np.nan_to_num(np.asarray(ds[k], float)) if k in ds.files else np.zeros(len(ds["time"]))
    rho = g("rho_2_1")
    return dict(t=np.asarray(ds["time"], float), w=100 * g("w_2_1"), rho=rho, j_bs=g("j_bs_2_1") / 1e6, P=g("P_cmd"),
                aim=g("aim_cmd") if "aim_cmd" in ds.files else rho)   # no controller: no aim, no misalignment


def samples(sh, k, seeds=None):
    w, rho, jbs, P, aim = sh["w"], sh["rho"], sh["j_bs"], sh["P"], sh["aim"]
    n = len(w)
    t = np.arange(1, n - k)
    if seeds:                                                     # no seed event inside the horizon
        t = np.array([i for i in t if not any(sh["t"][i] < s <= sh["t"][i + k] for s in seeds)])
    Pm = np.array([P[i:i + k].mean() for i in t])               # the command set after step i acts from step i + 1
    mis = np.array([np.mean(np.abs(aim[i:i + k] - rho[i:i + k])) * 200 for i in t])   # rho_norm to cm (a = 2 m)
    mis = np.where(Pm > 0, mis, 0.0)                              # misalignment only matters with the drive on
    X = np.stack([w[t], w[t] - w[t - 1], jbs[t], rho[t], Pm, mis], axis=1)
    return X, w[t + k] - w[t]


def init(key, sizes):
    ks = jax.random.split(key, len(sizes) - 1)
    return [dict(W=jax.random.normal(k_, (a, b)) * np.sqrt(2 / a), b=jnp.zeros(b)) for k_, a, b in zip(ks, sizes[:-1], sizes[1:])]


def forward(params, x, P=None):
    """A plain network (a list of layers) on all inputs, or the module forecaster (a dict of two) with P the power in MW."""
    if isinstance(params, dict):
        return forward(params["plain"], x[..., PLAIN]) + P / P_SCALE * forward(params["effect"], x[..., EFFECT])
    for i, l in enumerate(params):
        x = x @ l["W"] + l["b"]
        if i < len(params) - 1:
            x = jax.nn.relu(x)
    return x[..., 0]


def train(X, y, seed=0, steps=4000, lr=3e-3, arch="mlp"):
    mx, sx, my, sy = X.mean(0), X.std(0) + 1e-6, y.mean(), y.std() + 1e-6
    Xn, yn, Pw = jnp.asarray((X - mx) / sx), jnp.asarray((y - my) / sy), jnp.asarray(X[:, 4])
    key = jax.random.PRNGKey(seed)
    if arch == "module":
        k1, k2 = jax.random.split(key)
        params = dict(plain=init(k1, [len(PLAIN), 22, 22, 1]), effect=init(k2, [len(EFFECT), 22, 22, 1]))
    else:
        params = init(key, [X.shape[1], 32, 32, 1])
    opt = optax.adam(lr)
    st = opt.init(params)
    rng = np.random.default_rng(seed)

    @jax.jit
    def step(p, s, idx):
        l, g = jax.value_and_grad(lambda q: jnp.mean((forward(q, Xn[idx], Pw[idx]) - yn[idx]) ** 2))(p)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, l
    for _ in range(steps):
        params, st, _ = step(params, st, jnp.asarray(rng.integers(0, len(Xn), 512)))
    return dict(params=jax.tree_util.tree_map(np.asarray, params), mx=mx, sx=sx, my=my, sy=sy)


def predict(m, X, chip=False):
    if isinstance(m, (list, tuple)):
        return np.mean([predict(one, X, chip) for one in m], axis=0)
    X = np.asarray(X, float)
    x = (X - m["mx"]) / m["sx"]
    if chip == "hgq":
        return cf.hgq_forward(m["hgq"], x) * m["sy"] + m["my"]
    if not chip:
        return np.asarray(forward(m["params"], jnp.asarray(x), X[..., 4])) * m["sy"] + m["my"]
    q = lambda v: np.clip(np.round(np.asarray(v) * 1024) / 1024, -32, 32 - 1 / 1024)

    def fixed(layers, v):
        v = q(v)
        for i, l in enumerate(layers):
            v = q(q(v[..., :, None] * q(l["W"])).sum(axis=-2) + q(l["b"]))
            if i < len(layers) - 1:
                v = np.maximum(v, 0.0)
        return v[..., 0]
    p = m["params"]
    if isinstance(p, dict):
        out = q(fixed(p["plain"], x[..., PLAIN]) + q(q(X[..., 4] / P_SCALE) * fixed(p["effect"], x[..., EFFECT])))
    else:
        out = fixed(p, x)
    return out * m["sy"] + m["my"]


class NetMPC:
    def __init__(self, model, P_max=20.0, energy_cost=0.002, chip=False, dw_steps=1):
        self.m, self.P_max, self.cost, self.chip = model, P_max, energy_cost, chip
        self.past = collections.deque(maxlen=dw_steps)    # the network's change input is over 0.5 s: dw_steps of the plant's steps

    def __call__(self, obs):
        w, rho = 100 * obs["w"][0], obs["rho"][0]
        dw = w - self.past[0] if len(self.past) == self.past.maxlen else 0.0
        self.past.append(w)
        if not np.isfinite(rho):
            return 0.0, 0.5
        cands = [(P, rho + d / 200) for P in (0.0, self.P_max / 2, self.P_max) for d in (-1.0, 0.0, 1.0)]
        X = np.array([[w, dw, obs["j_bs"][0], rho, P, abs(a - rho) * 200 if P > 0 else 0.0] for P, a in cands])
        score = np.maximum(w + predict(self.m, X, self.chip), 0.0) + self.cost * X[:, 4]
        return cands[int(np.argmin(score))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default=os.path.join("runs", "torax_shots"))
    ap.add_argument("--test_from", type=int, default=80)
    ap.add_argument("--out", default=os.path.join("runs", "torax_shots", "forecaster.npz"))
    ap.add_argument("--seed", type=int, default=0, help="network initialization and batch order (same samples)")
    ap.add_argument("--arch", default="mlp", choices=["mlp", "module"])
    ap.add_argument("--clean", action="store_true", help="drop training windows with a seed event inside the horizon")
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--dump", default=None, help="save the training and test samples here instead of training "
                                                "(for the chip's quantization-aware training, chip_forecaster.py)")
    args = ap.parse_args()
    files = sorted(f for f in os.listdir(args.shots) if f.startswith("shot_") and f.endswith(".nc"))
    shots = {}
    for f in files:
        shots[int(f[5:9])] = load_shot(os.path.join(args.shots, f))
        if len(shots) % 20 == 0:
            print(f"loaded {len(shots)} shots", flush=True)
    index = {r["shot"]: [s_[0] for s_ in r["seeds"]] for r in map(json.loads, open(os.path.join(args.shots, "index.jsonl")))}
    rng, models, report, dump = np.random.default_rng(1), {}, {}, {}
    for k in args.horizons:
        tr = [samples(s, k, index.get(i) if args.clean else None) for i, s in shots.items() if i < args.test_from]
        te = [samples(s, k) for i, s in shots.items() if i >= args.test_from]
        te_clean = [samples(s, k, index.get(i)) for i, s in shots.items() if i >= args.test_from]
        X, y = np.concatenate([a for a, _ in tr]), np.concatenate([b for _, b in tr])
        noise = rng.uniform(0, 1.0, len(X))[:, None] * rng.normal(size=(len(X), 2)) * np.array([1.0, 1.0])
        X[:, :2] += noise
        Xt, yt = np.concatenate([a for a, _ in te]), np.concatenate([b for _, b in te])
        if args.dump:
            dump.update({f"k{k}_X": X, f"k{k}_y": y, f"k{k}_Xt": Xt, f"k{k}_yt": yt})
            continue
        models[k] = train(X, y, seed=args.seed, arch=args.arch)
        rmse = lambda p: float(np.sqrt(np.mean((p - yt) ** 2)))
        report[k] = dict(train_samples=len(X), test_samples=len(Xt), rmse_net=rmse(predict(models[k], Xt)),
                         rmse_chip=rmse(predict(models[k], Xt, chip=True)), rmse_persist=rmse(np.zeros_like(yt)),
                         chip_vs_net_max=float(np.abs(predict(models[k], Xt, True) - predict(models[k], Xt)).max()))
        Xc, yc = np.concatenate([a for a, _ in te_clean]), np.concatenate([b for _, b in te_clean])
        report[k].update(rmse_net_no_seeds=float(np.sqrt(np.mean((predict(models[k], Xc) - yc) ** 2))),
                         rmse_persist_no_seeds=float(np.sqrt(np.mean(yc ** 2))))
        print(f"horizon {k} steps ({0.5 * k} s): " + json.dumps(report[k]), flush=True)
    if args.dump:
        os.makedirs(os.path.dirname(args.dump), exist_ok=True)
        np.savez(args.dump, **dump)
        print("saved", args.dump)
        return
    flat ={f"k{k}_{n}": np.asarray(v) for k, m in models.items() for n, v in m.items() if n != "params"}
    for k, m in models.items():
        parts = m["params"].items() if isinstance(m["params"], dict) else [("", m["params"])]
        for part, layers in parts:
            flat.update({f"k{k}_{part + '_' if part else ''}L{i}_{n}": v for i, l in enumerate(layers) for n, v in l.items()})
    np.savez(args.out, **flat)
    json.dump(report, open(args.out.replace(".npz", ".json"), "w"), indent=1)
    print("saved", args.out)


def load_model(path, k=4, hgq=None):
    d = np.load(path)
    layers = lambda pre: [dict(W=d[f"{pre}L{i}_W"], b=d[f"{pre}L{i}_b"])
                          for i in range(len({key[len(pre):].split("_")[0] for key in d.files if key.startswith(pre + "L")}))]
    m = dict(params=(dict(plain=layers(f"k{k}_plain_"), effect=layers(f"k{k}_effect_")) if f"k{k}_plain_L0_W" in d.files
                     else layers(f"k{k}_")),
             mx=d[f"k{k}_mx"], sx=d[f"k{k}_sx"], my=float(d[f"k{k}_my"]), sy=float(d[f"k{k}_sy"]))
    if hgq:
        m["hgq"] = cf.load_hgq(hgq)   # trained on the same standardization (chip_forecaster.py)
    return m


if __name__ == "__main__":
    main()
