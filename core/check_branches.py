"""Branches come back exactly (package 1.1, H91). A snapshot; a run (A); back to the snapshot and a branch with things fired in it (P); back again and the
same run as A (B); back once more (C). B and C must equal A in every record of every step: anything a step reads that the snapshot left out shows up here.
An AI agent driving the harness found the first such leak (the transport table) by comparing its null branches.

Cases: mast, a MAST shot hands-off with the recommended version and FreeGSNKE (the coupled plasma, the beams, the bolometer, MAST's free boundary);
iter, TORAX's ITER plasma with FreeGSNKE and EPED-NN attached (the scenario studio_iter_coupled).

Usage (TORAX's environment, from the repository):
  python core/check_branches.py [--case mast iter] [--steps 40]
  python core/check_branches.py --case mast --core FOLDER      (the same check on another copy of core/, such as release 1.0's)
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402


def records(h, n):
    r = h.reactor
    start = len(r.records)
    h.apply(dict(cmd="step", n=n))
    return [dict(x) for x in r.records[start:]]


def compare(a, b):
    """The largest difference between two runs' records, key by key: (key, step, a, b) or None if every value is equal (NaN equal to NaN)."""
    worst, n_keys = None, set()
    for i, (x, y) in enumerate(zip(a, b)):
        for k in set(x) | set(y):
            n_keys.add(k)
            u, v = x.get(k), y.get(k)
            try:
                fu, fv = np.asarray(u, float), np.asarray(v, float)
                same = fu.shape == fv.shape and bool(np.all((fu == fv) | (np.isnan(fu) & np.isnan(fv))))
                gap = float(np.max(np.abs(fu - fv) / np.maximum(np.abs(fu), 1e-30))) if fu.shape == fv.shape and not same else (np.inf if not same else 0.0)
            except (TypeError, ValueError):
                same, gap = u == v, (0.0 if u == v else np.inf)
            if not same and (worst is None or gap > worst[4]):
                worst = (k, i, u, v, gap)
    if len(a) != len(b):
        worst = ("number of steps", 0, len(a), len(b), np.inf)
    return worst, len(n_keys)


def build(case):
    if case == "mast":
        import natural
        h = natural.hands_off(20689)
        return h, [dict(cmd="pellet", d_mm=1.1, r_mm=0.63), dict(cmd="gas", species="D", rate=5e21, delay=0.0, duration=0.02), dict(cmd="island", m=2, n=1, w_cm=3),
                   dict(cmd="ecrh", P=0.5, rho0=0.3, width=0.1)], 5
    if case == "iter":
        import twin
        s = json.load(open(os.path.join(HERE, "..", "scenarios", "studio_iter_coupled.json")))
        h = twin.Harness(s, keep_history=False)
        h.apply(dict(cmd="step", until=42.6))           # past the scenario's free boundary (41 s) and EPED-NN (42 s)
        return h, [dict(cmd="pellet"), dict(cmd="island", m=2, n=1, w_cm=4), dict(cmd="ecrh", P=10.0, rho0=0.3, width=0.1),
                   dict(cmd="gas", species="D", rate=1e22, delay=0.0, duration=1.0)], 3
    raise SystemExit(f"no case {case}")


def run_case(case, steps):
    t0 = time.perf_counter()
    h, fire, settle = build(case)
    h.apply(dict(cmd="step", n=settle))
    import twin
    models = [type(o).__name__ for o in twin.branch_models(h)] if hasattr(twin, "branch_models") else "(this code has no branch_models)"
    print(f"{case}: built and settled in {time.perf_counter() - t0:.0f} s, t = {h.reactor.t:.4f} s; models {models}", flush=True)
    t1 = time.perf_counter()
    ans = h.apply(dict(cmd="snapshot", name="branch_point"))
    t_snap = time.perf_counter() - t1
    if ans.get("kept_by_reference"):
        print(f"  kept by reference, not copied: {ans['kept_by_reference']}", flush=True)
    A = records(h, steps)
    t1 = time.perf_counter()
    h.apply(dict(cmd="restore", name="branch_point"))
    t_rest = time.perf_counter() - t1
    answers = [h.apply(c) for c in fire]
    P = records(h, steps)
    h.apply(dict(cmd="restore", name="branch_point"))
    B = records(h, steps)
    h.apply(dict(cmd="restore", name="branch_point"))
    C = records(h, steps)
    pel = answers[0]
    moved = {k: (P[-1].get(k, np.nan) / A[-1].get(k, np.nan) - 1.0) for k in ("n_line", "W_thermal", "T_e0") if A[-1].get(k)}
    print(f"  snapshot {t_snap:.2f} s, restore {t_rest:.2f} s; pellet from {pel.get('size_from')}: {pel.get('pellet_atoms', 0):.2e} atoms, "
          f"{pel.get('share_of_plasma', 0):.2f} of the plasma's electrons" + (f", WARNING {pel['warning']}" if pel.get("warning") else ""), flush=True)
    print(f"  the branch moved, after {steps} steps: " + ", ".join(f"{k} {100 * v:+.1f}%" for k, v in moved.items()), flush=True)
    ok = True
    for label, run in (("B, after the branch", B), ("C, restored again", C)):
        worst, n = compare(A, run)
        if worst is None:
            print(f"  {label}: identical to A on all {n} keys over {len(A)} steps", flush=True)
        else:
            ok = False
            k, i, u, v, gap = worst
            print(f"  {label}: DIFFERS from A, worst {k} at step {i}: {u} against {v} (relative {gap:.2e})", flush=True)
    if h.free is not None:
        h.free.close()
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", nargs="+", default=["mast", "iter"])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--core", default=HERE, help="the folder of modules to test (default: this one)")
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.core))
    results = {case: run_case(case, a.steps) for case in a.case}
    print("branches: " + ", ".join(f"{k} {'exact' if v else 'FAILED'}" for k, v in results.items()), flush=True)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
