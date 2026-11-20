"""The L-H threshold with published multi-machine numbers (H49, docs/lit/28), and whether it says when MAST is in H-mode. Martin 2008 for the conventional machines it was fitted on, 2/M from Delabie 2026, the low-density branch at 0.35 of the Greenwald density (0.23 to 0.46 on AUG, C-Mod, JET and MAST), the power that crosses the separatrix as P_ohm + P_abs - dW/dt.
Spherical tokamaks: Takizuka 2004's aspect-ratio term stays 3 to 8 times low on later MAST, and measured thresholds at low aspect ratio are about 8 times Martin on MAST (A 1.3), 5 to 10 on NSTX (A 1.3), 7 to 15 on Pegasus (A 1.2), with the ratio growing as A falls (Howlett 2023, Kaye 2011, Thome 2017). (3 / A)^2.2 below A = 3 carries those three. The form and exponent are mine, from those published numbers; no shot of ours set them.
Usage (TORAX's venv): python -I core/lh_general.py
"""
import json
import os
import sys

import numpy as np

MU0 = 4e-7 * np.pi


def surface_area(R, a, kappa):
    return 4 * np.pi ** 2 * R * a * np.sqrt(0.5 * (1 + kappa ** 2))


def P_LH(n20, B0, S, R, a, Ip_MA, M=2.0):
    """In MW. Below n_min = 0.35 n_GW the threshold rises as (n_min / n)^2 (docs/lit/28, section 1)."""
    n_min = 0.35 * Ip_MA / (np.pi * a ** 2)
    high = 0.0488 * np.maximum(n20, n_min) ** 0.717 * B0 ** 0.803 * S ** 0.941 * (2 / M) * np.maximum(3 * a / R, 1.0) ** 2.2
    return high * np.where(n20 < n_min, (n_min / np.maximum(n20, 1e-3)) ** 2, 1.0)


def test():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.dont_write_bytecode = True
    import density_drops as dd
    cat = dd.catalog()
    rows = []
    for s in dd.shots():
        g = dd.load(s)
        if not dd.sound(g):
            continue
        t = g["t"]
        win = cat.get(str(s), {}).get("phases", {}).get("hmode", {}).get("windows") or []
        H = np.array([any(w["t0"] <= x <= w["t1"] for w in win) for x in t])
        nel = g["nel"] / 1e20
        Ip = np.abs(g["Ip"]) / 1e6
        R, a, B0, kappa = g["R0"], g["a"], np.abs(g["B0"]), g["kappa"]
        S = surface_area(R, a, kappa)
        P_nbi, P_ohm = np.nan_to_num(g["P_heat"]), np.abs(np.nan_to_num(g["V_loop"])) * Ip
        path = 2 * np.sqrt(np.maximum((R + a) ** 2 - 0.7 ** 2, 0.01))          # the tangential beam's path through the plasma
        absorbed = 1 - np.exp(-np.maximum(g["nel"], 0) * 2.9e-20 * path)
        dW = np.gradient(np.nan_to_num(g["W_th"]) / 1e6, t)
        P_net = P_ohm + P_nbi * absorbed - dW
        lh = P_LH(nel, B0, S, R, a, Ip)
        n_G = Ip / (np.pi * a ** 2)
        on = P_nbi > 0.5
        for i in np.flatnonzero(on & np.isfinite(lh) & np.isfinite(P_net)):
            rows.append((s, t[i], float(H[i]), P_net[i] / lh[i], nel[i] / n_G[i], lh[i], P_net[i]))
    R = np.array(rows, float)
    Y = R[:, 2] > 0

    def rank(x, y):
        good = np.isfinite(x)
        x, y = x[good], y[good]
        r = np.empty(len(x))
        r[np.argsort(x)] = np.arange(len(x))
        n1, n0 = y.sum(), (~y).sum()
        return (r[y].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)

    print(f"{len(R)} moments of beam heating on {len(set(R[:, 0]))} shots, {Y.mean():.2f} of them in H-mode")
    for name, c in (("P_net / P_LH", 3), ("n / n_GW", 4)):
        v = rank(R[:, c], Y)
        print(f"  {name:36s} rank against H-mode {v:.2f}; median in H {np.median(R[Y, c]):.2f}, in L {np.median(R[~Y, c]):.2f}")
    ratio = R[:, 3]
    print(f"  share above 1: in H {np.mean(ratio[Y] > 1):.2f}, in L {np.mean(ratio[~Y] > 1):.2f}. Threshold median {np.median(R[:, 5]):.2f} MW, power {np.median(R[:, 6]):.2f} MW")
    for lo, hi in ((0, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 5), (5, 100)):
        m = (ratio >= lo) & (ratio < hi)
        print(f"    P/P_LH {lo} to {hi}: {m.sum():5d} moments, share in H {Y[m].mean() if m.any() else float('nan'):.2f}")


if __name__ == "__main__":
    test()
