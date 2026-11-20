"""ITER's hybrid scenario from Python, with nothing outside this repository. The plasma runs to 40 s first, past the 25 s in which
the scenario lets current diffuse 200 times faster to settle (as TORAX's own example does), the point the front end starts from.
Then a branch seeds a 4 cm 2/1 island and runs 3 s, and a second branch runs the same 3 s without it. The scenario's decision rule
fires the gyrotrons at an island wider than 3 cm, so the first branch shows what the island and its ECCD cost.
The commands are the harness's own (README.md, "The harness").
Usage (from the repository): python examples/run_iter.py      a few minutes; the first step compiles TORAX's step with JAX"""
import json
import os
import sys

sys.dont_write_bytecode = True
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "core"))
import twin  # noqa: E402

scenario = json.load(open(os.path.join(REPO, "scenarios", "studio_iter.json")))
h = twin.Harness(scenario, keep_history=False)


def show(label):
    a = h.apply({"cmd": "observe"})
    island = a["islands"]["2/1"]
    print(f"{label:26s} t {a['t']:5.1f} s   fusion {a['P_fusion_MW']:5.0f} MW   Q {a['Q']:4.1f}   "
          f"2/1 island {island['w_cm']:4.1f} cm   ECCD {a['P_cmd_MW']:4.1f} MW", flush=True)


h.apply({"cmd": "step", "until": 40})                   # settled, as the front end starts
show("settled flat top")
h.apply({"cmd": "snapshot", "name": "before"})
h.apply({"cmd": "island", "m": 2, "n": 1, "w_cm": 4})
h.apply({"cmd": "step", "n": 30})                       # 30 steps of 0.1 s
show("3 s after a 4 cm island")
h.apply({"cmd": "restore", "name": "before"})           # back to the snapshot, exactly
h.apply({"cmd": "step", "n": 30})
show("the same 3 s, no island")
