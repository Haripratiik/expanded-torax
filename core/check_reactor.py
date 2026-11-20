"""The steppable reactor against the numbers before it, and its edits. Run on the same seeded island under the rule:
  1. run() gives the same widths and fusion power as the saved traces of the same setup from before the restructuring (--save to make them).
  2. A branch: snapshot, go on, restore, go on again: identical, bit for bit.
  3. Each edit does what it says.
Usage (TORAX's venv): python core/check_reactor.py [--save]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import controllers as ctl  # noqa: E402
import ntm  # noqa: E402
import run_ntm  # noqa: E402
import torax  # noqa: E402

SAVED = os.path.join("runs", "twin", "reactor_reference.npz")      # --save writes the workspace's own reference here
SHIPPED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "reactor_reference.npz")
PATH = SAVED if (os.path.exists(SAVED) or "--save" in sys.argv) else SHIPPED     # else the reference that ships with the repository


def setup(t_final=30.0):
    cfg = run_ntm.base_config(t_final, dt=0.25, physical_current_after=10.0)
    return torax.ToraxConfig.from_dict(cfg), [ntm.Mode(2, 1, seeds=((12.0, 0.05),), rotation=ntm.Rotation(quench_rate=0.5))], dict(controller=ctl.Tracking(P_max=20.0), elms=ntm.ELMs(), keep_history=False)


def trace(recs):
    return np.array([r.get("w_2_1", 0.0) for r in recs]), np.array([r["P_fusion"] for r in recs])


cfg, modes, kw = setup()
if "--save" in sys.argv:
    _, recs = ntm.run(cfg, modes, **kw)
    w, pf = trace(recs)
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    np.savez(PATH, w=w, pf=pf)
    print("saved", PATH, "steps", len(w), "w max", w.max(), "P_fus end", pf[-1] / 1e6)
    sys.exit()
ref = np.load(PATH)
_, recs = ntm.run(cfg, modes, **kw)
w, pf = trace(recs)
print("1. against the code before the restructuring: largest width difference", np.abs(w - ref["w"]).max(), "m; largest fusion difference", np.abs(pf - ref["pf"]).max() / 1e6, "MW;", len(w), "steps")

cfg, modes, kw = setup()
r = ntm.Reactor(cfg, modes, **kw)
while r.t < 14.0:
    r.step()
snap = r.snapshot()
for _ in range(20):
    r.step()
a = trace(r.records)
r.restore(snap)
for _ in range(20):
    r.step()
b = trace(r.records)
print("2. a branch and its repeat: largest width difference", np.abs(a[0] - b[0]).max(), "m; largest fusion difference", np.abs(a[1] - b[1]).max() / 1e6, "MW;", len(a[0]), "steps")
print("   the same as the straight run:", np.abs(b[0] - w[:len(b[0])]).max(), "m")

r.restore(snap)
T0 = np.asarray(r.state.core_profiles.T_e.value).copy()
rc = np.asarray(r.state.geometry.rho_norm)
r.edit_profile("T_e", 0.5, rho=(0.0, 0.3))
T1 = np.asarray(r.state.core_profiles.T_e.value)
print("3. edit_profile T_e x 0.5 inside rho 0.3: ratio at rho 0.1:", round(float(T1[np.argmin(abs(rc - 0.1))] / T0[np.argmin(abs(rc - 0.1))]), 3), "at rho 0.6:", round(float(T1[np.argmin(abs(rc - 0.6))] / T0[np.argmin(abs(rc - 0.6))]), 3))
pf0 = r.records[-1]["P_fusion"] / 1e6
for _ in range(8):
    r.step()
print("   the plasma answers: fusion", round(pf0, 1), "to", round(r.records[-1]["P_fusion"] / 1e6, 1), "MW after 2 s")
r.restore(snap)
ntm.HEAT[:] = 0.5
for _ in range(20):
    r.step()
print("   heating x 0.5 for 5 s: exhaust power", round(trace(r.records)[1][-1] / 1e6, 1), "MW fusion,", round(r.records[-1]["P_SOL"] / 1e6, 1), "MW P_SOL (branch without it:", round(a[1][-1] / 1e6, 1), "MW fusion)")
r.restore(snap)
ntm.TURBULENCE[:2, (ntm.TURB_RHO > 0.3) & (ntm.TURB_RHO < 0.7)] = 3.0
for _ in range(20):
    r.step()
print("   turbulence x 3 over rho 0.3 to 0.7 for 5 s: stored energy", round(r.records[-1]["W_thermal"] / 1e6, 1), "MJ, fusion", round(r.records[-1]["P_fusion"] / 1e6, 1), "MW")
r.restore(snap)
r.flatten_zone((0.0, 0.5))
for _ in range(4):
    r.step()
print("   forced crash inside rho 0.5, 1 s later: fusion", round(r.records[-1]["P_fusion"] / 1e6, 1), "MW")
