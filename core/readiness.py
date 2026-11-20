"""Is the twin ready? (H86, docs/digital-twin.md "Ready: what it means"). Reads whole-shot summaries made by whole_shot.py and checks them against the
criteria written before any claim:
  1. runs: every shot reaches the end of its window or a real event; no numerical stop; the twin disrupts only within 20 ms of a real disruption
  2. the chip's continuous effects: inside the 10% bar at half the run-times or more (median error over all run-times at or below 10%; shape in its own
     units against a stated bar: elongation and triangularity 0.05, axis radius and vertical position 2 cm)
  3. events, per shot: ELMs and sawtooth crashes within a factor 2 of the measured count (|ln((twin + 1) / (measured + 1))| <= ln 2), island and H-mode time
     within a factor 2 (or both under 5% of the time), and no H-mode on a shot that stayed in L-mode
  4. the harness: natural.hands_off defaults to the recommended version (natural.RECOMMENDED); the smoke test's last summary has every check at exit 0
Summaries are pooled run by run, so the 11 H82 shots and the 14 development shots count once each.
Usage: python -I core/readiness.py v9s,v9d [--version 9]
"""
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import whole_shot as ws  # noqa: E402

CHIP = ("plasma current", "line density", "core temperature", "stored energy", "q95", "elongation", "triangularity", "minor radius", "axis radius", "vertical position", "radiated power", "neutrons")
BAR = {"elongation": 0.05, "triangularity": 0.05, "axis radius": 2.0, "vertical position": 2.0}      # absolute bars for the shape, in its units (cm for the last two)
LN2 = float(np.log(2.0))


def pooled(names):
    S = {n: json.load(open(ws.out_summary(n))) for n in names}
    errs, per_run, ends = {}, [], []
    for n in names:
        R = ws.run_errors(S[n]["tags"])
        for key, (t0, e) in R.items():
            errs[key] = e
        per_run += S[n]["per_run"]
        ends += S[n]["ends"]
    return errs, per_run, ends


def main(names, version=None):
    errs, per_run, ends = pooled(names)
    verdict = {}
    print(f"readiness of {', '.join(names)}: {len(per_run)} runs on {len({x['shot'] for x in per_run})} shots")
    # 1. runs
    bad = [x for x in ends if x["twin_stop"] != "SimError.NO_ERROR"]
    early = [x for x in ends if x["twin_disrupted"] and (x["real_end"] not in ("disruption", "quench") or x["early_ms"] > 20)]
    short = [x for x in per_run if x["t_last"] < x["t_end"] - 0.02 and not any(e["shot"] == x["shot"] and e["twin_disrupted"] for e in ends)]
    ok1 = not bad and not early
    verdict["runs"] = ok1
    print(f"1. runs: numerical stops {len(bad)} {[x['shot'] for x in bad]}; own disruptions not near a real one {len(early)} {[(x['shot'], round(x['early_ms'])) for x in early]};"
          f" ended short otherwise {len(short)} {[x['shot'] for x in short]}  ->  {'met' if ok1 else 'not met'}")
    # 2. effects
    print("2. effects: median error over all run-times; share of run-times inside the bar  (bar 10%, shape in its own units)")
    met2 = []
    for k in CHIP:
        a = np.concatenate([e[k] for e in errs.values() if k in e])
        a = a[np.isfinite(a)]
        if not len(a):
            print(f"   {k:18s} no points")
            continue
        kind = ws.KIND[k]
        bar = BAR.get(k, 0.10)
        inside = float(np.mean(a <= bar))
        med = float(np.median(a))
        ok = med <= bar
        met2.append((k, ok))
        unit = (lambda v: f"{100 * v:5.1f}%") if kind == "rel" else ((lambda v: f"{v:5.3f}") if kind == "abs" else (lambda v: f"{v:4.2f}cm"))
        print(f"   {k:18s} median {unit(med)}  inside {100 * inside:3.0f}% of {len(a):4d}   {'met' if ok else 'NOT met'}" + ("   (given: the programme)" if k == "plasma current" else ""))
    verdict["effects"] = [k for k, ok in met2 if ok]
    print(f"   met {sum(ok for _, ok in met2)} of {len(met2)}: {', '.join(k for k, ok in met2 if ok)}")
    # 3. events, per shot
    def ln_ratio(p):
        return abs(np.log((p[0] + 1.0) / (p[1] + 1.0)))
    def share_ok(p):
        if p[0] is None or p[1] is None:
            return None
        if p[0] < 0.05 and p[1] < 0.05:
            return True
        return abs(np.log(max(p[0], 0.01) / max(p[1], 0.01))) <= LN2
    rows = []
    for x in per_run:
        rows.append(dict(shot=x["shot"], elms=x["elms"], crashes=x["crashes"], island=x["island_share"], H=x["H_share"],
                         elms_ok=ln_ratio(x["elms"]) <= LN2, crashes_ok=ln_ratio(x["crashes"]) <= LN2, island_ok=share_ok(x["island_share"]), H_ok=share_ok(x["H_share"]),
                         false_H=(x["H_share"][0] or 0) > 0.05 and (x["H_share"][1] or 0) == 0))
    print("3. events per shot (twin; measured)            ELMs        crashes      n = 1 mode time  H time")
    for r_ in rows:
        f = lambda ok: "ok" if ok else ("--" if ok is None else "NO")
        sh = lambda p: f"{p[0]:.2f}; {p[1]:.2f}" if p[0] is not None and p[1] is not None else "-"
        print(f"   {r_['shot']}  {r_['elms'][0]:4d}; {r_['elms'][1]:4d} {f(r_['elms_ok'])}   {r_['crashes'][0]:3d}; {r_['crashes'][1]:3d} {f(r_['crashes_ok'])}   {sh(r_['island']):11s} {f(r_['island_ok'])}   {sh(r_['H']):11s} {f(r_['H_ok'])}" + ("   false H-mode" if r_["false_H"] else ""))
    share = lambda key: np.mean([r_[key] for r_ in rows if r_[key] is not None]) if rows else float("nan")
    print(f"   shots within a factor 2: ELMs {100 * share('elms_ok'):.0f}%, crashes {100 * share('crashes_ok'):.0f}%, n = 1 mode time {100 * share('island_ok'):.0f}%, H time {100 * share('H_ok'):.0f}%; false H-mode on {sum(r_['false_H'] for r_ in rows)} shots")
    verdict["events"] = {k: float(share(k)) for k in ("elms_ok", "crashes_ok", "island_ok", "H_ok")}
    # 4. the harness
    import natural
    import inspect
    default = inspect.signature(natural.hands_off).parameters["version"].default
    default = getattr(natural, "RECOMMENDED", default) if default in (None, "recommended") else default
    smoke = os.path.join(ws.ROOT, "runs", "twin", "smoke", "summary.txt")
    lines = [s for s in open(smoke).read().splitlines() if " exit " in s] if os.path.exists(smoke) else []
    smoke_ok = bool(lines) and all(" exit 0 " in s for s in lines)
    ok4 = smoke_ok and (version is None or str(default) == str(version))
    verdict["harness"] = ok4
    print(f"4. harness: hands_off defaults to version {default}{'' if version is None else f' (recommended {version})'}; smoke test {len(lines)} checks, {'all pass' if smoke_ok else 'not all pass'}  ->  {'met' if ok4 else 'not met'}")
    return verdict


if __name__ == "__main__":
    names = sys.argv[1].split(",") if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else ["v9s", "v9d"]
    version = sys.argv[sys.argv.index("--version") + 1] if "--version" in sys.argv else None
    main(names, version)
