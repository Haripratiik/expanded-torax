"""The etched chip, the 2D solver surrogate and the expanded TORAX, in one loop (H44, docs/digital-twin.md). Does the connection work, and can the surrogate stand in for the solver inside it?

Plasma: ITER hybrid in the harness, settled, as elsewhere. Island: one of three in place on the 2/1 surface, on the same seeds:
  mre        the modified Rutherford equation, as in every earlier run
  solver     solver/ntm.py, the 2D full solver of the NTM testbed (the reference)
  surrogate  H31's module architecture, k = 2, stride 4, as trained (never retrained here)
Controller: none, the plain rule (3 cm, through sensors and gyrotrons), or the etched chip (white noise, or real MAST background with --background).
Each trial: an island of w_cm at t = settle + 1 s, then a window of seconds. Recorded: width, drive energy, whether it disrupted or the 2D model blew up, fusion power, seconds of wall clock per plasma second by stage.
Usage (TORAX's venv): python core/loop2d.py --plants mre surrogate solver --controllers none rule chip --w_cm 4 8 --window 20
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twin  # noqa: E402

BASE = {"name": "loop2d", "t_final": 2000.0, "dt": 0.1, "edge": False, "impurities": {"Ne": 0.01, "W": 1e-5, "Ar": 0.0}, "physics": {"a_bs": 1.5, "a_cd": 0.5}, "gyrotrons": {"units": 20, "unit_MW": 1.0},
        "sensors": {"sigma_w_cm": 0.3, "sigma_rho_cm": 0.5, "latency_steps": 1, "seed": 1}, "decision": {"kind": "none"}, "events": []}
CONTROLLERS = dict(prop={"kind": "python", "target": "island2d:proportional", "args": {"gain": 0.8}}, prop_low={"kind": "python", "target": "island2d:proportional", "args": {"gain": 0.4}}, none={"kind": "none"}, rule={"kind": "rule", "w_on_cm": 3, "w_off_cm": 3}, rule_tight={"kind": "rule", "w_on_cm": 1.5, "w_off_cm": 1.0, "hold": 0.0}, fast_rule={"kind": "etched", "final": True, "controller": "plain rule (fast detection, modulated)", "cl": -1.0}, chip={"kind": "etched", "final": True, "controller": "sim chip + decision chip", "cl": -1.0},
                   chip_bg={"kind": "etched", "final": True, "controller": "sim chip + decision chip", "cl": -1.0, "background": "runs/mast_background/bank_dc622_clean.npz"})


def trial(plant, controller, w_cm, window, settle, seed, coupled=False):
    s = dict(BASE)
    if plant != "mre":
        s["plant2d"] = dict(backend=plant, continuous_eff=1.0 if controller in ("rule_tight", "prop", "prop_low") else 0.5)
    if coupled:
        s["sawtooth"] = True
        s["elms"] = dict(seed=seed)
    h = twin.Harness(s, keep_history=False)
    r = h.reactor
    t0 = time.perf_counter()
    h.apply(dict(cmd="step", until=settle))
    if coupled:                                          # the whole coupled plasma under the same test: FreeGSNKE's shape, and L-H, EPED-NN's pedestal, density feedback, ELMs, sawteeth, natural seeds and the beta limit (coupled.py)
        h.apply(dict(cmd="free_boundary", machine="ITER", mode="control", interval=1.0))
        h.apply(dict(cmd="couple", models=["hmode", "elms", "sawtooth", "modes", "collapse"], H0=True, seed=seed))
    spec = dict(CONTROLLERS[controller])
    if spec["kind"] == "etched":
        spec["seed"] = seed
    h.apply(dict(cmd="decision", spec=spec))
    h.apply(dict(cmd="step", until=settle + 1.0))
    h.apply(dict(cmd="island", m=2, n=1, w_cm=w_cm))
    n0 = len(r.records)
    t1 = time.perf_counter()
    while r.t < settle + 1.0 + window - 1e-9 and r.step() is not None:
        pass
    wall = time.perf_counter() - t1
    recs = r.records[n0:]
    t = np.array([x["t"] for x in recs])
    w = np.array([100 * x.get("w_2_1", 0.0) for x in recs])
    P = np.array([x.get("P_cmd", 0.0) for x in recs])
    pf = np.array([x["P_fusion"] for x in recs]) / 1e6
    steps = np.diff(t, prepend=settle + 1.0)
    if coupled and h.natural is not None:
        print(f"  coupled: H {h.natural.H}, seeds {len(h.natural.seeds)}, collapses {len(h.natural.collapses)}, ELMs {len(r.elm_times)}, solves {h.free.solves}, refused {h.free.rejected}", flush=True)
    if h.free is not None:
        h.free.close()
    out = dict(plant=plant, controller=controller, w_cm=w_cm, seed=seed, disrupted=bool(r.disrupted), t_end=float(t[-1] - settle - 1.0), w_max=float(w.max()), w_end=float(w[-1]), E_MJ=float((P * steps).sum()), P_fus_MW=float((pf * steps).sum() / window), t_below_1cm=(float(t[np.argmax(w < 1.0)] - settle - 1.0) if (w < 1.0).any() else None), wall_per_plasma_s=wall / max(t[-1] - settle - 1.0, 1e-3),
               blown=bool(h.island2d.blown) if h.island2d else False, ms_per_2d_step=float(1e3 * np.median(h.island2d.walls)) if h.island2d and h.island2d.walls else None, trace_t=(t - settle - 1.0).round(2).tolist()[::5], trace_w=w.round(3).tolist()[::5], trace_P=P.round(2).tolist()[::5], settle_wall_s=round(t1 - t0, 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plants", nargs="+", default=["mre", "surrogate", "solver"])
    ap.add_argument("--controllers", nargs="+", default=["none", "rule", "chip"])
    ap.add_argument("--w_cm", nargs="+", type=float, default=[4.0, 8.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--window", type=float, default=20.0)
    ap.add_argument("--settle", type=float, default=40.0)
    ap.add_argument("--out", default=os.path.join("runs", "twin", "loop2d", "trials.jsonl"))
    ap.add_argument("--coupled", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            x = json.loads(line)
            done.add((x["plant"], x["controller"], x["w_cm"], x["seed"], round(x.get("window", args.window), 1)))
    log = open(args.out, "a")
    for plant in args.plants:
        for controller in args.controllers:
            for w in args.w_cm:
                for seed in args.seeds:
                    if (plant, controller, w, seed, round(args.window, 1)) in done:
                        continue
                    try:
                        x = trial(plant, controller, w, args.window, args.settle, seed, coupled=args.coupled)
                    except Exception as err:
                        import traceback
                        traceback.print_exc()
                        print(plant, controller, w, seed, "failed", repr(err)[:300], flush=True)
                        continue
                    x["window"] = args.window
                    log.write(json.dumps(x) + "\n")
                    log.flush()
                    print(f"{plant:9s} {controller:7s} w {w:4.1f} seed {seed}: disrupted {x['disrupted']}, ended {x['t_end']:.1f} s, largest {x['w_max']:.1f} cm, end {x['w_end']:.1f} cm, below 1 cm at {x['t_below_1cm']}, drive {x['E_MJ']:.0f} MJ, fusion {x['P_fus_MW']:.0f} MW, 2D blown {x['blown']}, {x['wall_per_plasma_s']:.2f} wall s per plasma s, {x['ms_per_2d_step']} ms per 2D step", flush=True)


if __name__ == "__main__":
    main()
