"""Shots in the twin (H36, docs/digital-twin.md): start the reactor from any moment of a shot, run it on, compare with what the shot did.

A Shot holds one discharge in a neutral format: t (s), rho (rho_norm), T_e and T_i (keV), n_e (m^-3), psi (Wb), each [time, rho]; Ip (A),
P_heat (MW) and w (island width in m) per time. Only t, rho, T_e and n_e are required. Shot.load reads the same names from an npz,
so any real data source only has to write that file. Shot.from_torax makes one from a TORAX run: a "shot" whose truth is known, which is
how this machinery is tested before real data goes in.

continue_from() builds the TORAX config that starts at t_cut from the shot's profiles. After the cut, boundary values, plasma current and
heating come from the shot while it lasts and hold their last value past its end ("run past the data end"). A real test of the end
of a shot is a backtest: pretend the data ends at t_cut, run on, and score against the part held back. score() gives the relative RMS error
of each profile over 0.1 < rho < 0.9 at each time. The trust window runs from the settling time (when every profile first comes inside the
tolerance after the cut) to the horizon (how long after the cut they stay inside).
The default tolerance of 10% is what integrated modelling reaches on real shots: about 10% rms for T_i, T_e and n_e over 33 JET and DIII-D
H-modes (Bateman et al. 2003), 9 to 11% for temperatures over 99 discharges of four machines (Rafiq et al. 2023).

Closure tests (main): the truth is a TORAX run of the ITER hybrid flat top with a heating step and an island, cut at several times.
  same      the twin has the same physics and perfect profiles: must stay near zero, or the machinery is wrong
  no_psi    the same without the measured current profile (real shots often lack one; psi comes from TORAX's current formula)
  noisy     profiles with 5% noise and a 9-point running mean, no T_i (T_e stands in), no psi: raw gradients upset stiff transport
  fitted    the same noise, but a smooth curve is fitted to each profile, as real profile fits are (fit_profiles)
  wrong     the truth's pedestal is 20% lower than the twin's: a physics error the twin cannot see, so its error must grow; how fast is the point
Usage (TORAX's venv): python core/shot.py
"""
import argparse
import copy
import dataclasses
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ntm  # noqa: E402
import run_ntm  # noqa: E402
import torax  # noqa: E402

CHANNELS = ("T_e", "T_i", "n_e")


@dataclasses.dataclass
class Shot:
    t: np.ndarray
    rho: np.ndarray
    T_e: np.ndarray
    n_e: np.ndarray
    T_i: np.ndarray | None = None
    psi: np.ndarray | None = None
    Ip: np.ndarray | None = None
    P_heat: np.ndarray | None = None
    w: np.ndarray | None = None
    name: str = "shot"

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        return cls(name=os.path.splitext(os.path.basename(path))[0], **{f.name: d[f.name] for f in dataclasses.fields(cls) if f.name in d.files})

    def save(self, path):
        np.savez(path, **{f.name: getattr(self, f.name) for f in dataclasses.fields(self) if getattr(self, f.name) is not None and f.name != "name"})

    @classmethod
    def from_torax(cls, tree, records=None, name="torax"):
        p, s = tree["profiles"].to_dataset(), tree["scalars"].to_dataset()
        w = None
        if records:
            w = np.array([r.get("w_2_1", 0.0) for r in records])
        return cls(t=p["time"].values, rho=p["rho_norm"].values, T_e=p["T_e"].values, T_i=p["T_i"].values, n_e=p["n_e"].values, psi=p["psi"].values,
                   Ip=s["Ip"].values, P_heat=s["P_aux_total"].values / 1e6, w=w, name=name)

    def at(self, name, t):
        """A channel at time t, linear in time, held at the ends."""
        v = getattr(self, name)
        i = np.clip(np.searchsorted(self.t, t), 1, len(self.t) - 1)
        f = np.clip((t - self.t[i - 1]) / (self.t[i] - self.t[i - 1]), 0.0, 1.0)
        return (1 - f) * v[i - 1] + f * v[i]


def continue_from(shot, t_cut, t_end, base):
    """The TORAX config that starts at t_cut from the shot's profiles. base gives everything the shot does not: geometry, transport, sources.
    Data after t_cut is used for boundary values and plasma current only, never for the profiles it is scored on."""
    cfg = copy.deepcopy(base)
    cfg["numerics"].update(t_initial=float(t_cut), t_final=float(t_end))
    T_e, n_e = shot.at("T_e", t_cut), shot.at("n_e", t_cut)
    T_i = shot.at("T_i", t_cut) if shot.T_i is not None else T_e
    now = np.array([float(t_cut)])
    pc = cfg["profile_conditions"]
    later = shot.t[shot.t >= t_cut]
    trace = lambda f: {float(t): float(v) for t, v in zip(later, f(later))}
    pc.update(T_e=(now, shot.rho, T_e[None, :]), T_i=(now, shot.rho, T_i[None, :]), n_e=(now, shot.rho, n_e[None, :]),
              n_e_nbar_is_fGW=False, normalize_n_e_to_nbar=False, n_e_right_bc_is_fGW=False,
              T_e_right_bc=trace(lambda ts: [shot.at("T_e", t)[-1] for t in ts]),
              T_i_right_bc=trace(lambda ts: [(shot.at("T_i", t) if shot.T_i is not None else shot.at("T_e", t))[-1] for t in ts]),
              n_e_right_bc=trace(lambda ts: [shot.at("n_e", t)[-1] for t in ts]))
    if shot.Ip is not None:
        pc["Ip"] = trace(lambda ts: [shot.at("Ip", t) for t in ts])
    if shot.psi is not None:
        pc.update(psi=(now, shot.rho, shot.at("psi", t_cut)[None, :]), initial_psi_mode="profile_conditions")
    return cfg


def score(shot, tree, t_cut):
    """Relative RMS error per channel per time, against the shot, over 0.1 < rho < 0.9."""
    p = tree["profiles"].to_dataset()
    t, rho = p["time"].values, p["rho_norm"].values
    keep = t >= t_cut - 1e-9
    band = (rho > 0.1) & (rho < 0.9)
    out = dict(t=t[keep])
    for name in CHANNELS:
        if getattr(shot, name) is None:
            continue
        true = np.array([[np.interp(r, shot.rho, shot.at(name, tt)) for r in rho[band]] for tt in t[keep]])
        sim = p[name].values[keep][:, band]
        out[name] = np.sqrt(np.mean((sim - true) ** 2, axis=1)) / np.mean(true, axis=1)
    return out


def trust_window(sc, t_cut, tol=0.10):
    """(settling, horizon): lead times at which every channel first comes inside the tolerance and at which it first leaves again. Horizon
    equals the whole window if it stays inside to the end; both are None if it never comes inside."""
    bad = np.zeros(len(sc["t"]), bool)
    for name in CHANNELS:
        if name in sc:
            bad |= sc[name] > tol
    t = sc["t"] - t_cut
    if bad.all():
        return None, None
    i = int(np.argmin(bad))
    later = np.where(bad[i:])[0]
    return float(t[i]), float(t[i + later[0]]) if len(later) else float(t[-1])


def fit_profiles(rho, v, knots=5):
    """A smooth curve through each time slice: a least-squares cubic spline with a few interior knots, more of them towards the edge where
    the pedestal is steep."""
    from scipy.interpolate import LSQUnivariateSpline
    inner = 1 - (1 - np.linspace(0.0, 1.0, knots + 2)[1:-1]) ** 1.5
    return np.array([LSQUnivariateSpline(rho, row, inner, k=3)(rho) for row in v])


def truth_config(t_final, pedestal_factor=1.0):
    cfg = run_ntm.base_config(t_final)
    P0 = cfg["sources"]["generic_heat"]["P_total"]
    cfg["sources"]["generic_heat"]["P_total"] = {0.0: P0, 59.999: P0, 60.0: 0.8 * P0}
    for k in ("T_i_ped", "T_e_ped"):
        cfg["pedestal"][k] *= pedestal_factor
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t_final", type=float, default=120.0)
    ap.add_argument("--cuts", type=float, nargs="+", default=[30.0, 50.0, 70.0, 90.0])
    ap.add_argument("--lead", type=float, default=30.0)
    ap.add_argument("--tests", nargs="+", default=["same", "no_psi", "noisy", "fitted", "wrong"])
    ap.add_argument("--out", default=os.path.join("runs", "twin", "closure"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    results = {}
    base = truth_config(args.t_final)
    mode = lambda w0=0.0, seeds=(): ntm.Mode(2, 1, w0=w0, seeds=seeds)
    t0 = time.perf_counter()
    tree_ok, _ = ntm.run(torax.ToraxConfig.from_dict(base), [mode()])
    truth_ok = Shot.from_torax(tree_ok, name="same_physics")
    tree_bad, _ = ntm.run(torax.ToraxConfig.from_dict(truth_config(args.t_final, 0.8)), [mode()])
    truth_bad = Shot.from_torax(tree_bad, name="lower_pedestal")
    print(f"truths ready, {time.perf_counter() - t0:.0f} s", flush=True)
    for test in args.tests:
        shot = truth_bad if test == "wrong" else truth_ok
        data = dataclasses.replace(shot)
        if test == "no_psi":
            data = dataclasses.replace(shot, psi=None)
        if test in ("noisy", "fitted"):
            def noisy(v):
                n = v * (1 + 0.05 * rng.standard_normal(v.shape))
                if test == "fitted":
                    return fit_profiles(shot.rho, n)
                k = np.ones(9) / 9
                return np.array([np.convolve(np.pad(row, 4, mode="edge"), k, mode="valid") for row in n])
            data = dataclasses.replace(shot, T_e=noisy(shot.T_e), n_e=noisy(shot.n_e), T_i=None, psi=None)
        rows = []
        for cut in args.cuts:
            cfg = continue_from(data, cut, min(cut + args.lead, args.t_final), base)
            if data.psi is None:
                cfg["profile_conditions"].update(psi=None, initial_psi_mode="j")
            t1 = time.perf_counter()
            tree, _ = ntm.run(torax.ToraxConfig.from_dict(cfg), [mode()])
            sc = score(shot, tree, cut)
            settle, horizon = trust_window(sc, cut)
            row = dict(cut=cut, settle_s=settle, horizon_s=horizon, wall_s=round(time.perf_counter() - t1, 1),
                       **{f"{c}_at_{lead}s": float(np.interp(lead, sc["t"] - cut, sc[c])) for c in CHANNELS for lead in (0.5, 2, 5, 10, 20, 30)})
            rows.append(row)
            print(test, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        results[test] = rows
    path = os.path.join(args.out, "closure.json")
    json.dump({**(json.load(open(path)) if os.path.exists(path) else {}), **results}, open(path, "w"), indent=1)
    print("\nrelative RMS error (%) at 0.5, 2, 5, 10, 20, 30 s after the cut, mean over cuts, then the trust window at 10%")
    for test, rows in results.items():
        print(f"  {test:7s} " + "  ".join(f"{c}: " + " ".join(f"{100 * np.mean([r[f'{c}_at_{lead}s'] for r in rows]):.1f}" for lead in (0.5, 2, 5, 10, 20, 30)) for c in CHANNELS)
              + f"   settles {[r['settle_s'] for r in rows]} s, horizon {[r['horizon_s'] for r in rows]} s (run {args.lead:.0f} s)")


if __name__ == "__main__":
    main()
