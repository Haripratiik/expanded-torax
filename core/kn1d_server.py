"""KN1D (KN1DPy, LGPL-3.0, downloaded into external/kn1d-venv) as a separate process for the twin (H85): the neutral atoms and molecules from the wall into the edge
at the outboard midplane, from the twin's own profiles and a wall molecular pressure; returns the ionization inside the separatrix as a source per unit volume on the twin's rho
grid, taken as uniform on each flux surface. The same protocol as freegsnke_server.py: pickled (name, kwargs) on stdin, pickled {ok, value | error, trace} on stdout.
Commands:
  source   rho (cells), n_e (m^-3), T_e, T_i (eV) on rho, R_out (outboard midplane R of each cell, m), gauge (wall molecular density, D2 per m^3), R_wall, R_lim, LC_shadow (m),
           lam_n, lam_T (scrape-off layer e-folding lengths, m), rho_inner (KN1D's grid runs from the wall to this rho), dir (scratch folder)
           n_sep, T_sep (optional: the SOL's own separatrix density and temperature, sol.py; default the core's values at rho 1)
           -> dict(S (m^-3 s^-1 on rho), n0_sep (m^-3), f_inside (share of all ionization that falls inside the separatrix), S_sol, S_in (per unit area, m^-2 s^-1))
Usage: not by hand; neutrals.py starts it with external/kn1d-venv/Scripts/python -I.
"""
import os
import pickle
import sys
import traceback

import numpy as np

from KN1DPy.kn1d import kn1d

ROOT = os.path.abspath(os.environ.get("XTORAX_HOME") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the workspace: runs/twin and external/ (README, "Where things live")
CONFIG = os.path.join(ROOT, "external", "downloads", "kn1dpy", "src", "KN1DPy-main", "examples", "MAST-U", "python_output", "config.toml")
KB = 1.380649e-23


def source(rho, n_e, T_e, T_i, R_out, gauge, R_wall, R_lim, LC_shadow, lam_n, lam_T, rho_inner, dir, n_x=300, max_gen=50, n_sep=None, T_sep=None):
    rho, n_e, T_e, T_i, R_out = (np.asarray(a, float) for a in (rho, n_e, T_e, T_i, R_out))
    R_sep = float(R_out[-1] + (R_out[-1] - R_out[-2]) * (1.0 - rho[-1]) / max(rho[-1] - rho[-2], 1e-9))     # the last cell's centre carried out to rho 1
    R_in = float(np.interp(rho_inner, rho, R_out))
    x = np.linspace(0.0, R_wall - R_in, n_x)
    R = R_wall - x
    inside = R <= R_sep
    rho_x = np.interp(R, R_out, rho)
    n_sep_core, Te_sep, Ti_sep = (float(np.interp(1.0, rho, a)) for a in (n_e, T_e, T_i))
    n_sep = n_sep_core if n_sep is None else float(n_sep)          # the scrape-off layer's own separatrix values when the twin has a SOL model (sol.py, H86)
    if T_sep is not None:
        Te_sep = Ti_sep = float(T_sep)
    n = np.where(inside, np.interp(rho_x, rho, n_e), n_sep * np.exp(-(R - R_sep) / lam_n))
    Te = np.where(inside, np.interp(rho_x, rho, T_e), Te_sep * np.exp(-(R - R_sep) / lam_T))
    Ti = np.where(inside, np.interp(rho_x, rho, T_i), Ti_sep * np.exp(-(R - R_sep) / lam_T))
    n, Te, Ti = np.maximum(n, 1e16), np.maximum(Te, 2.0), np.maximum(Ti, 2.0)
    LC = np.where(R > R_lim, LC_shadow, 0.0)
    os.makedirs(dir, exist_ok=True)
    kn1d(x=x, xlimiter=R_wall - R_lim, xsep=R_wall - R_sep, GaugeH2=gauge * KB * 300.0 / 0.133322, mu=2.0, Ti=Ti, Te=Te, n=n, vxi=np.zeros_like(x), LC=LC,
         PipeDia=np.zeros_like(x), max_gen=max_gen, compute_errors=0, debrief=0, Hdebrief=0, H2debrief=0, Hdebug=0, H2debug=0, File=dir, config_path=CONFIG)
    H, H2 = dict(np.load(os.path.join(dir, "KN1D_H.npz"))), dict(np.load(os.path.join(dir, "KN1D_H2.npz")))
    xH, S = np.asarray(H["xH"], float), np.asarray(H["Sion"], float)
    if "SP" in H2:
        S = S + np.interp(xH, np.asarray(H2["xH2"], float), np.asarray(H2["SP"], float))
    S_rho = np.interp(R_wall - R_out, xH, S, left=0.0, right=0.0)
    S_rho[rho < rho_inner] = 0.0
    xsep = R_wall - R_sep
    total = float(np.trapezoid(S, xH))
    inner = float(np.trapezoid(np.where(xH >= xsep, S, 0.0), xH))
    ly = np.asarray(H["Lyman"], float) if "Lyman" in H else np.zeros_like(xH)
    return dict(S=S_rho, n0_sep=float(np.interp(xsep, xH, np.asarray(H["nH"], float))), f_inside=inner / max(total, 1e-30),
                S_sol=total - inner, S_in=inner,                 # per unit area of the slab (m^-2 s^-1): outside and inside the separatrix
                ly=float(np.trapezoid(ly, xH)), ly_in=float(np.trapezoid(np.where(xH >= xsep, ly, 0.0), xH)))     # Lyman-alpha per unit area (W/m^2), all and inside


def main():
    sys.stdout = sys.stderr                          # KN1D prints; replies go on the real stdout only
    out = os.fdopen(1, "wb")
    inp = os.fdopen(0, "rb")
    while True:
        try:
            name, kw = pickle.load(inp)
        except EOFError:
            return
        if name == "quit":
            return
        try:
            if name == "source":
                value = source(**kw)
            elif name == "ping":
                value = "pong"
            else:
                raise ValueError(f"unknown command {name}")
            reply = dict(ok=True, value=value)
        except Exception as err:
            reply = dict(ok=False, error=repr(err), trace=traceback.format_exc())
        pickle.dump(reply, out, protocol=4)
        out.flush()


if __name__ == "__main__":
    main()
