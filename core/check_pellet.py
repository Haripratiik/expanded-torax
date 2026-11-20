"""A fuelling pellet, step by step: what happens to density and temperature in the 2 s after it, and where the particles go.
Usage (TORAX's venv): python core/check_pellet.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_plant  # noqa: E402
import twin  # noqa: E402

h = twin.Harness(check_plant.scenario(False), keep_history=False)
r = h.reactor
h.apply(dict(cmd="step", until=40.0))
GRID = [0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95]


def show(label):
    cp, geo = r.state.core_profiles, r.state.geometry
    rc = np.asarray(geo.rho_norm)
    dv = np.asarray(geo.vpr) * float(np.asarray(geo.drho_norm).ravel()[0])
    N = float((np.asarray(cp.n_e.value) * dv).sum())
    print(f"{label:10s} t {r.t - 40:5.2f} electrons {N:.3e} n_e", np.round(np.interp(GRID, rc, np.asarray(cp.n_e.value)) / 1e20, 3), "T_e", np.round(np.interp(GRID, rc, np.asarray(cp.T_e.value)), 2), "W", round(float(r.ppo.W_thermal_total) / 1e6, 1), "ped", np.round(r.ped_extra, 3), flush=True)


show("before")
print(h.apply(dict(cmd="pellet", r_mm=4.0, side="hfs")))
show("just after")
for k in range(20):
    r.step()
    if k < 5 or k % 5 == 4:
        show("")
