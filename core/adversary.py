"""An automated attacker on the harness (H39): find what breaks a decision model.

Random fault scripts are drawn from a grammar of what can go wrong: islands, turbulence bursts, gyrotron trips, a stuck mirror, sensor faults, energy kicks, forced crashes, heating drops, torque impulses, a growing error field. Every decision model
faces the same scripts from the same settled plasma, branched by snapshot, so each trial costs seconds and comparisons are paired. A trial is a fixed window; the outcome is whether the plasma disrupts, plus fusion lost, island size and drive energy.
Each script that breaks a model is then shrunk, as a software fuzzer shrinks a failing input: faults are removed one at a time while it still breaks, leaving a small case a person can read.
Outputs in <out>: trials.jsonl (every trial), failures.json (the shrunk scripts), summary.json.
Usage (TORAX's venv): python core/adversary.py --n 40 --models rule phys net hgq
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twin  # noqa: E402

DECISIONS = {
    "none": {"kind": "none"},
    "rule": {"kind": "rule", "w_on_cm": 3, "w_off_cm": 3},
    "phys": {"kind": "phys", "oracle": False, "horizon": 2, "cost": 0.15},
    "net": {"kind": "net", "cost": 0.03},
    "hgq": {"kind": "net", "cost": 0.03, "chip": "hgq", "hgq_model": "runs/torax_chip/hgq_forecaster.npz"},
    # that project's loop and island model, fed by this plasma: the etched chip, and on the same plant its fast rule and an ITER-like loop
    "chip": {"kind": "etched", "final": True, "controller": "sim chip + decision chip", "seed": 0, "cl": -1.0},
    "fast_rule": {"kind": "etched", "final": True, "controller": "plain rule (fast detection, modulated)", "seed": 0, "cl": -1.0},
    "iter_like": {"kind": "etched", "final": True, "controller": "detect-then-suppress, 0.15 s", "seed": 0, "cl": -1.0},
    "chip_plant_none": {"kind": "etched", "final": True, "controller": "no control", "seed": 0, "cl": -1.0},
    # the same on real coil signals: a stretch of MAST background in place of white noise
    "chip_bg": {"kind": "etched", "final": True, "controller": "sim chip + decision chip", "seed": 0, "cl": -1.0, "background": "runs/mast_background/bank_dc622_clean.npz"},
    "fast_rule_bg": {"kind": "etched", "final": True, "controller": "plain rule (fast detection, modulated)", "seed": 0, "cl": -1.0, "background": "runs/mast_background/bank_dc622_clean.npz"},
    "iter_like_bg": {"kind": "etched", "final": True, "controller": "detect-then-suppress, 0.15 s", "seed": 0, "cl": -1.0, "background": "runs/mast_background/bank_dc622_clean.npz"},
}
BASE = {"name": "adversary", "t_final": 2000.0, "dt": 0.1, "edge": False, "physics": {"a_bs": 1.5, "a_cd": 0.5}, "gyrotrons": {"units": 20, "unit_MW": 1.0},
        "sensors": {"sigma_w_cm": 0.3, "sigma_rho_cm": 0.5, "latency_steps": 1, "seed": 1}, "decision": {"kind": "none"}, "events": []}
SURFACE = 0.637
FULL = dict(BASE, impurities={"Ne": 0.01, "W": 1e-5, "Ar": 0.0})      # the expanded harness: the same plasma with live impurities, for the full grammar


def draw(rng, window, full=False):
    """One script: an island near the start of the trouble, and up to three faults around it. full adds the expanded harness's plasma events: fast-ion bursts, a wall flake, edge collapse, a back transition, a seed gas, coils, a fuelling pellet."""
    t0 = float(rng.uniform(1.0, window / 3))
    script = [dict(cmd="island", m=2, n=1, w_cm=round(float(rng.uniform(2.5, 7.0)), 1), t=round(t0, 1))]

    def when():
        return round(float(max(t0 + rng.uniform(-2.0, 4.0), 0.1)), 1)

    def zone():
        c = float(rng.uniform(0.2, 0.8))
        h = float(rng.uniform(0.05, 0.2))
        return [round(max(c - h, 0.0), 2), round(min(c + h, 0.95), 2)]

    makers = (
        lambda: dict(cmd="turbulence", factor=round(float(rng.uniform(1.5, 6.0)), 1), rho=zone(), duration=round(float(rng.uniform(1.0, 8.0)), 1), t=when()),
        lambda: dict(cmd="trip", units=int(rng.integers(8, 21)), duration=round(float(rng.uniform(0.5, 5.0)), 1), t=when()),
        lambda: dict(cmd="mirror_stuck", duration=round(float(rng.uniform(1.0, 5.0)), 1), t=when()),
        lambda: dict(cmd="sensor_bias", surface_cm=round(float(rng.uniform(1.0, 4.0) * rng.choice([-1, 1])), 1), duration=round(float(rng.uniform(1.0, 10.0)), 1), t=when()),
        lambda: dict(cmd="sensor_bias", width_cm=round(float(rng.uniform(-3.0, 0.0)), 1), duration=round(float(rng.uniform(1.0, 6.0)), 1), t=when()),
        lambda: dict(cmd="kick", frac=round(float(rng.uniform(0.1, 0.4)), 2), rho=[0.0, round(float(rng.uniform(0.3, 0.7)), 2)], t=when()),
        lambda: dict(cmd="crash", rho=[0.0, round(float(rng.uniform(0.3, 0.6)), 2)], t=when()),
        lambda: dict(cmd="brake", m=2, n=1, factor=round(float(rng.uniform(0.3, 0.9)), 2), t=round(t0 + 0.1, 1)),
        lambda: dict(cmd="error_field", factor=round(float(rng.uniform(1.2, 2.5)), 2), t=when()),
    )
    if full:
        makers = makers + (
            lambda: dict(cmd="fishbone", period=round(float(rng.uniform(0.2, 1.0)), 2), expelled=round(float(rng.uniform(0.1, 0.5)), 2), seed_prob=round(float(rng.uniform(0.0, 0.3)), 2), duration=round(float(rng.uniform(2.0, 10.0)), 1), t=when()),
            lambda: dict(cmd="alfven", redistributed=round(float(rng.uniform(0.1, 0.5)), 2), duration=round(float(rng.uniform(2.0, 10.0)), 1), t=when()),
            lambda: dict(cmd="wall_flake", species="W", atoms=float(10 ** rng.uniform(18.0, 20.0)), t=when()),
            lambda: dict(cmd="marfe", share=round(float(rng.uniform(0.2, 0.8)), 2), pedestal=round(float(rng.uniform(0.5, 0.9)), 2), duration=round(float(rng.uniform(1.0, 5.0)), 1), t=when()),
            lambda: dict(cmd="back_transition", T=round(float(rng.uniform(0.4, 0.8)), 2), n=round(float(rng.uniform(0.6, 0.9)), 2), ramp=round(float(rng.uniform(0.2, 2.0)), 1), duration=round(float(rng.uniform(2.0, 8.0)), 1), t=when()),
            lambda: dict(cmd="gas", species="Ne", rate=float(10 ** rng.uniform(20.0, 21.0)), duration=round(float(rng.uniform(1.0, 5.0)), 1), t=when()),
            lambda: dict(cmd="rmp", level=round(float(rng.uniform(0.2, 0.9)), 2), t=when()),
            lambda: dict(cmd="pellet", r_mm=round(float(rng.uniform(2.0, 4.0)), 1), side=str(rng.choice(["hfs", "lfs"])), t=when()),
        )
    for i in rng.choice(len(makers), int(rng.integers(0, 4)), replace=False):
        script.append(makers[int(i)]())
    return sorted(script, key=lambda c: c["t"])


class Bench:
    def __init__(self, settle, window, base=BASE):
        self.h = twin.Harness(base, keep_history=False)
        self.window = window
        self.h.apply(dict(cmd="step", until=settle))
        self.h.apply(dict(cmd="snapshot", name="settled"))
        self.t0 = self.h.reactor.t
        self.base = self.trial("none", [])     # a quiet plasma over the same window
        print(f"settled at {self.t0:.1f} s; quiet window: mean fusion {self.base['P_fus_MW']:.1f} MW", flush=True)

    def trial(self, model, script, seed=0):
        h = self.h
        h.apply(dict(cmd="restore", name="settled"))
        # the chip's noise, background stretch and deposition width come from this seed: one per script, the same for every controller, so comparisons stay paired
        h.apply(dict(cmd="decision", spec=dict(DECISIONS[model], seed=seed) if DECISIONS[model]["kind"] == "etched" else DECISIONS[model]))
        h.script = sorted((dict(c, t=self.t0 + c["t"]) for c in script), key=lambda c: c["t"])
        n0 = len(h.reactor.records)
        while h.reactor.t < self.t0 + self.window - 1e-9 and h.reactor.step() is not None:
            pass
        recs = h.reactor.records[n0:]
        t = np.array([r["t"] for r in recs])
        steps = np.diff(t, prepend=self.t0)
        pf = np.array([r["P_fusion"] for r in recs]) / 1e6
        is_width = lambda k: re.fullmatch(r"w_\d+_\d+", k) is not None        # not w_d_ or w_dep_
        w = 100 * np.array([max([r[k] for k in r if is_width(k)] or [0.0]) for r in recs])
        P = np.array([r.get("P_cmd", 0.0) for r in recs])
        dis = bool(h.reactor.disrupted)
        return dict(model=model, disrupted=dis, t_end=float(t[-1] - self.t0), P_fus_MW=float((pf * steps).sum() / self.window), w_int=float((w * steps).sum()), w_max=float(w.max()),
                    E_MJ=float((P * steps).sum()), locked=any(any(k.startswith("locked_") for k in r) for r in recs))

    def shrink(self, model, script, seed=0):
        """Remove faults one at a time while it still disrupts."""
        kept = list(script)
        changed = True
        while changed and len(kept) > 1:
            changed = False
            for c in list(kept):
                rest = [x for x in kept if x is not c]
                if self.trial(model, rest, seed)["disrupted"]:
                    kept, changed = rest, True
                    break
        return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--models", nargs="+", default=["none", "rule", "phys", "net", "hgq"])
    ap.add_argument("--settle", type=float, default=40.0)
    ap.add_argument("--window", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shrink", type=int, default=6, help="failures to shrink per model")
    ap.add_argument("--full", action="store_true", help="the expanded harness's events too")
    ap.add_argument("--out", default=os.path.join("runs", "twin", "adversary"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    scripts = [draw(rng, args.window, args.full) for _ in range(args.n)]
    bench = Bench(args.settle, args.window, FULL if args.full else BASE)
    table = {m: [] for m in args.models}
    log = open(os.path.join(args.out, "trials.jsonl"), "a")
    t0 = time.perf_counter()
    for i, script in enumerate(scripts):
        for m in args.models:
            r = bench.trial(m, script, i + 1)
            table[m].append(r)
            log.write(json.dumps(dict(script_id=i, script=script, **r)) + "\n")
        log.flush()
        if (i + 1) % 5 == 0:
            print(f"{i + 1} scripts, {time.perf_counter() - t0:.0f} s: " + ", ".join(f"{m} {np.mean([x['disrupted'] for x in table[m]]):.2f}" for m in args.models), flush=True)
    failures = {}
    for m in args.models:
        if m == "none":
            continue
        broke = [i for i, r in enumerate(table[m]) if r["disrupted"]]
        failures[m] = []
        for i in broke[:args.shrink]:
            small = bench.shrink(m, scripts[i], i + 1)
            failures[m].append(dict(script_id=i, faults_before=len(scripts[i]), shrunk=small, result=bench.trial(m, small, i + 1)))
            print(f"{m}: script {i} shrinks from {len(scripts[i])} to {len(small)} faults: {small}", flush=True)
    json.dump(failures, open(os.path.join(args.out, "failures.json"), "w"), indent=1)
    summary = dict(n=args.n, window_s=args.window, quiet_P_fus_MW=bench.base["P_fus_MW"])
    print(f"\n{args.n} scripts over {args.window:.0f} s from a settled plasma; quiet fusion {bench.base['P_fus_MW']:.0f} MW")
    print(f"{'model':6s} {'disrupted':>9s} {'mean drive MJ':>13s} {'mean fusion MW':>14s} {'mean largest island cm':>22s}")
    for m in args.models:
        rows = table[m]
        summary[m] = dict(disrupted=float(np.mean([r["disrupted"] for r in rows])), E_MJ=float(np.mean([r["E_MJ"] for r in rows])), P_fus_MW=float(np.mean([r["P_fus_MW"] for r in rows])), w_max=float(np.mean([r["w_max"] for r in rows])))
        print(f"{m:6s} {summary[m]['disrupted']:9.2f} {summary[m]['E_MJ']:13.0f} {summary[m]['P_fus_MW']:14.0f} {summary[m]['w_max']:22.1f}")
    names = [m for m in args.models if m != "none"]
    print("\nscripts that break the row's model and not the column's:")
    print("       " + " ".join(f"{m:>5s}" for m in names))
    for a in names:
        print(f"{a:6s} " + " ".join(f"{sum(x['disrupted'] and not y['disrupted'] for x, y in zip(table[a], table[b])):5d}" for b in names))
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
