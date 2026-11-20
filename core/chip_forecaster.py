"""The reactor's island forecaster as a real chip design, through the companion chip project's hardware flow (H35).

The companion chip project turns its control surrogate into hardware with HGQ2 (quantization-aware training that learns
every bit width, Keras 3 on JAX), da4ml (the trained network to pipelined Verilog, constant multiplications as adder
graphs) and Verilator (the Verilog simulated and compared bit for bit with Keras), with Yosys for an area on Nangate
45 nm cells (its docs/hardware-flow.md). This script puts the 2 s forecaster (forecaster.py, horizon 4 steps) through
the same flow, with that project's own model builder and settings (sim/control/surrogate.build_model: 6-32-32-1, ReLU,
weights 12 bits and activations 10 fractional bits to start, beta 1e-7; Adam 2e-3, 200 epochs, batch 256), saturating
activations so that an input beyond the training range clips instead of wrapping. Training data: forecaster.py --dump
(the float network's samples and standardization). Outputs in <out>:
  forecaster_hgq.keras   the trained HGQ2 model
  rtl/                   da4ml's Verilog and the Verilator build
  hgq_forecaster.npz     per layer the quantized weights and biases and the input quantizers' (k, i, f), rounding and
                         overflow, with check inputs and the Keras and Verilator outputs on them
  report.json            test error (cm) against the float network and persistence, EBOPs, pipeline stages, the
                         bit-exact checks, and the Yosys area
hgq_forward (numpy only, importable without Keras) evaluates the exported layers exactly as the chip does; the reactor
uses it (forecaster.NetMPC with chip="hgq"), and this script checks it against Keras and Verilator.
Usage (in WSL, the companion chip project's venv and tools):
  KERAS_BACKEND=jax python core/chip_forecaster.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROUND = {"RND": lambda v: np.floor(v + 0.5), "TRN": np.floor}


def quantize(x, k, i, f, round_mode, overflow):
    """HGQ2's fixed-point quantizer (quantizers.fixed_point): saturate, then round to 2^-f; WRAP rounds, then wraps."""
    if overflow == "WRAP":
        v = ROUND[round_mode](x * 2.0 ** f) / 2.0 ** f
        bias = k * 2.0 ** (i + k - 1)
        return (v + bias) % 2.0 ** (i + k) - bias
    hi = 2.0 ** i - 2.0 ** -f
    lo = -k * (hi if overflow == "SAT_SYM" else 2.0 ** i)
    return ROUND[round_mode](np.clip(x, lo, hi) * 2.0 ** f) / 2.0 ** f


def hgq_forward(layers, x):
    """The exported network on standardized inputs, in float64 (exact for these word widths): each layer quantizes its
    input, then multiplies by its quantized weights and adds its quantized bias; ReLU between layers."""
    x = np.asarray(x, float)
    for j, L in enumerate(layers):
        x = quantize(x, L["k"], L["i"], L["f"], L["round"], L["overflow"]) @ L["W"] + L["b"]
        if j < len(layers) - 1:
            x = np.maximum(x, 0.0)
    return x[..., 0]


def load_hgq(path):
    d = np.load(path)
    n = len({key.split("_")[0] for key in d.files if key.startswith("L")})
    return [dict(W=d[f"L{j}_W"], b=d[f"L{j}_b"], k=d[f"L{j}_k"], i=d[f"L{j}_i"], f=d[f"L{j}_f"],
                 round=str(d[f"L{j}_round"]), overflow=str(d[f"L{j}_overflow"])) for j in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/torax_chip/forecaster_data.npz")
    ap.add_argument("--float_model", default="runs/torax_shots/forecaster.npz")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--chip_project", default=os.environ.get("XTORAX_CHIP_PROJECT"), help="the chip project's folder (not public)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--beta0", type=float, default=1e-7)
    ap.add_argument("--port_margin", type=int, default=4)
    ap.add_argument("--latency_cutoff", type=float, default=5.0)
    ap.add_argument("--clock", type=float, default=5.0)
    ap.add_argument("--no_synth", action="store_true")
    ap.add_argument("--float_twin", action="store_true",
                    help="train the same recipe unquantized and save it in forecaster.py's format instead")
    ap.add_argument("--out", default="runs/torax_chip")
    args = ap.parse_args()
    if not args.chip_project:
        raise SystemExit("training needs the chip project's hardware flow, which is not part of this release (--chip_project or XTORAX_CHIP_PROJECT)")
    os.environ.setdefault("KERAS_BACKEND", "jax")
    sys.path[:0] = [args.chip_project, os.path.join(args.chip_project, "scripts")]
    import hgq  # noqa: F401
    import keras
    from da4ml.codegen.rtl.rtl_model import VerilogModel
    from da4ml.converter import trace_model
    from da4ml.trace import comb_trace
    from rtl_surrogate import yosys_area
    from sim.control.surrogate import build_model, calibrate

    d, fm = np.load(args.data), np.load(args.float_model)
    k = args.k
    mx, sx, my, sy = fm[f"k{k}_mx"], fm[f"k{k}_sx"], float(fm[f"k{k}_my"]), float(fm[f"k{k}_sy"])
    X = ((d[f"k{k}_X"] - mx) / sx).astype("float32")
    Xt = ((d[f"k{k}_Xt"] - mx) / sx).astype("float32")
    y, yt = ((d[f"k{k}_y"] - my) / sy).astype("float32"), d[f"k{k}_yt"]
    os.makedirs(args.out, exist_ok=True)
    rmse = lambda p: float(np.sqrt(np.mean((p - yt) ** 2)))
    if args.float_twin:   # is a change in the controller quantization, or only a different training run?
        model = build_model(X.shape[1], 1, (32, 32), args.beta0, seed=0, quantized=False)
        model.compile(optimizer=keras.optimizers.Adam(2e-3), loss="mse")
        model.fit(X, y[:, None], epochs=args.epochs, batch_size=256, verbose=0)
        Ws = [np.asarray(v, float) for v in model.get_weights()]
        path = os.path.join(args.out, "forecaster_keras_float.npz")
        np.savez(path, **{f"k{k}_{n}": fm[f"k{k}_{n}"] for n in ("mx", "sx", "my", "sy")},
                 **{f"k{k}_L{j}_{n}": Ws[2 * j + (n == "b")] for j in range(len(Ws) // 2) for n in ("W", "b")})
        pred = np.asarray(model(Xt, training=False))[:, 0] * sy + my
        print(json.dumps(dict(float_twin_rmse_cm=rmse(pred), saved=path)))
        return

    t0 = time.perf_counter()
    model = build_model(X.shape[1], 1, (32, 32), args.beta0, seed=0, f0=10, b0=12, overflow="SAT")
    model.compile(optimizer=keras.optimizers.Adam(2e-3), loss="mse")
    model.fit(X, y[:, None], epochs=args.epochs, batch_size=256, verbose=0)
    ebops = calibrate(model, X)
    train_s = time.perf_counter() - t0
    model.save(os.path.join(args.out, "forecaster_hgq.keras"))
    pred = np.asarray(model(Xt, training=False))[:, 0] * sy + my

    # the float network (forecaster.py) on the same test samples
    h = (d[f"k{k}_Xt"] - mx) / sx
    for j in range(len({key.split("_")[1] for key in fm.files if key.startswith(f"k{k}_L")})):
        h = h @ fm[f"k{k}_L{j}_W"] + fm[f"k{k}_L{j}_b"]
        h = np.maximum(h, 0.0) if f"k{k}_L{j + 1}_W" in fm.files else h
    pred_float = h[:, 0] * sy + my

    # check inputs: the test set, and inputs spread past the training range on every feature
    rng = np.random.default_rng(0)
    lo, hi = X.min(0), X.max(0)
    wide = rng.uniform(lo - 0.5 * (hi - lo), hi + 0.5 * (hi - lo), size=(4000, X.shape[1])).astype("float32")
    xc = np.concatenate([Xt, wide])
    y_keras = np.asarray(model(xc, training=False))[:, 0].astype(float)

    layers = []
    for L in model.layers:
        q = L.iq.quantizer
        kk, ii, ff = (np.asarray(v, float) for v in q.kif)
        layers.append(dict(W=np.asarray(L.qkernel, float), b=np.asarray(L.qbias, float), k=kk, i=ii, f=ff,
                           round=q.round_mode, overflow=q.overflow_mode))
    y_np = hgq_forward(layers, xc)

    t1 = time.perf_counter()
    first = layers[0]
    kif = (1, int(first["i"].max()) + args.port_margin, int(first["f"].max()) + 1)
    inp, outp = trace_model(model, inputs_kif=kif)
    comb = comb_trace(inp, outp)
    rtl = VerilogModel(comb, prj_name="forecaster", path=os.path.join(args.out, "rtl"),
                       latency_cutoff=args.latency_cutoff, clock_period=args.clock)
    rtl.compile(metadata=dict(model="forecaster_hgq.keras"))
    y_rtl = np.asarray(rtl.predict(xc)).reshape(-1).astype(float)
    rtl_s = time.perf_counter() - t1

    np.savez(os.path.join(args.out, "hgq_forecaster.npz"), x_check=xc, y_keras=y_keras, y_rtl=y_rtl,
             **{f"L{j}_{n}": v for j, L in enumerate(layers) for n, v in L.items()})
    rep = dict(horizon_steps=k, train_samples=len(X), test_samples=len(Xt), epochs=args.epochs, beta0=args.beta0,
               train_s=round(train_s, 1), rtl_s=round(rtl_s, 1), ebops=ebops,
               rmse_cm=dict(hgq=rmse(pred), float=rmse(pred_float), persistence=rmse(np.zeros_like(yt))),
               nonzero_weights=[int((L["W"] != 0).sum()) for L in layers], weights=[int(L["W"].size) for L in layers],
               input_bits=[dict(i_max=int(L["i"].max()), f_max=int(L["f"].max()), pruned=int((L["i"] + L["f"] <= 0).sum()))
                           for L in layers],
               check_inputs=len(xc), numpy_vs_keras_max=float(np.abs(y_np - y_keras).max()),
               rtl_vs_keras_max=float(np.abs(y_rtl - y_keras).max()), input_port_kif=kif,
               latency_cutoff=args.latency_cutoff, clock_ns=args.clock)
    rep["bit_exact"] = rep["rtl_vs_keras_max"] == 0.0 and rep["numpy_vs_keras_max"] == 0.0
    for attr in ("cost", "latency"):
        v = getattr(comb, attr, None)
        if v is not None:
            rep[attr] = v if isinstance(v, (int, float)) else str(v)
    if not args.no_synth:
        rep.update(yosys_area(os.path.join(args.out, "rtl"), "forecaster"))
    json.dump(rep, open(os.path.join(args.out, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
