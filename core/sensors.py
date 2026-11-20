"""What each sensor would read, from a plasma's state (H66, docs/digital-twin.md, "What each sensor would read"). No random numbers: every reading is a calculation from the fields and profiles.

The store has no sensor positions. EFIT's map of the flux is built from these same sensors, so each one's place and direction are recovered from real shots: the grid point where EFIT's flux, or its field along some direction, tracks that channel best. The same goes for each soft X-ray chord, whose weight over flux surfaces comes from real profiles.

  place     magnetics on the fit shots: runs/twin/sensors/geometry.json
  refine    pickup coils' places to 5 mm, below the map's grid
  chords    soft X-ray: each chord's closest approach and the filter's cut-off; the interferometer's factor: runs/twin/sensors/chords.npz
  edge      D-alpha's level and bursts, and the mode tracks' size against the catalog's field: runs/twin/sensors/edge.json
  check     level 1: EFIT's map and measured profiles through each calculation, on shots not used to place them

Usage (TORAX's venv): python -u core/sensors.py place|chords|check
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.environ.get("XTORAX_HOME") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the workspace: runs/twin and external/ (README, "Where things live")
OUT = os.path.join(ROOT, "runs", "twin", "sensors")
FLUX, PROBES = "flux", ("ccbv", "obr", "obv")
CHANNELS = dict(flux=15, ccbv=40, obr=18, obv=18)
RHO_BINS = np.linspace(0.0, 1.0, 11)                    # sqrt of normalized poloidal flux


def shots():
    return json.load(open(os.path.join(OUT, "shots.json")))


def load(shot):
    path = os.path.join(OUT, "real", f"{shot}.npz")
    return dict(np.load(path, allow_pickle=False)) if os.path.exists(path) else None


def at_times(z, name, t):
    """A channel array on the 1 ms grid, at the given times."""
    x = z[name]
    idx = np.clip(np.round((t - z["t"][0]) / 1e-3).astype(int), 0, len(z["t"]) - 1)
    return x[..., idx]


def field_maps(psi, R, Z):
    """B_R and B_Z from psi[z, R, ...], with psi in Wb per radian: B_R = -(1 / R) dpsi/dZ, B_Z = (1 / R) dpsi/dR."""
    dz = np.gradient(psi, Z, axis=0)
    dr = np.gradient(psi, R, axis=1)
    rr = R[None, :, None] if psi.ndim == 3 else R[None, :]
    return -dz / rr, dr / rr


# ---------- a step in the sensors themselves. Between shots 23667 and 23822 the outboard flux loops read about 9% less against the same field, the outboard vertical coils 10 to 30% less, and the centre column and outboard radial coils do not change. The same drop shows on runs through FreeGSNKE, which know nothing of EFIT, so the sensors moved rather than EFIT's maps. CC04 reads nothing from 23983 on.
STEP = 23750
DEAD = {("flux", 1): 23950}
_ERA = None


def gains(shot):
    """Each channel's gain on this shot: 1 before the step, after it the median over the other list of shots (runs/twin/sensors/era_gain.json). So a check shot's gain never comes from itself."""
    global _ERA
    out = {f: np.ones(CHANNELS[f]) for f in (FLUX,) + PROBES}
    if shot >= STEP:
        if _ERA is None:
            path = os.path.join(OUT, "era_gain.json")
            _ERA = json.load(open(path)) if os.path.exists(path) else {}
        if _ERA:
            which = "from_others" if shot in shots()["check"] else "from_check"
            for f in out:
                out[f] = np.array(_ERA[which][f], float)
    for (f, c), since in DEAD.items():
        if shot >= since:
            out[f][c] = 0.0
    return out


# ---------- magnetics: where each channel sits, and which way it points
def place(fit=None, save=True):
    """For every channel, over all good EFIT slices of the fit shots, the node where the reading is best explained. Flux loops: y = a psi. Pickup coils: y = a B_R + b B_Z. Normal equations are summed shot by shot, so this stays small."""
    fit = shots()["fit"] if fit is None else fit
    acc = {}
    Rg = Zg = None
    n_used = 0
    for s in fit:
        z = load(s)
        if z is None or "efit_psi" not in z or "flux" not in z:
            continue
        ok = np.isfinite(z["efit_psi"]).all(axis=(0, 1)) & (z["efit_t"] >= z["t"][0]) & (z["efit_t"] <= z["t"][-1])
        if ok.sum() < 5:
            continue
        psi = z["efit_psi"][:, :, ok].astype(np.float64)
        Rg, Zg = z["efit_R"], z["efit_Z"]
        t = z["efit_t"][ok]
        BR, BZ = field_maps(psi, Rg, Zg)
        n_used += 1
        y = at_times(z, FLUX, t).astype(np.float64)
        good = np.isfinite(y)
        for c in range(y.shape[0]):
            m = good[c]
            if m.sum() < 3:
                continue
            a = acc.setdefault(("flux", c), dict(xx=0.0, xy=0.0, yy=0.0, n=0))
            a["xx"] = a["xx"] + np.sum(psi[:, :, m] ** 2, axis=2)
            a["xy"] = a["xy"] + np.sum(psi[:, :, m] * y[c, m], axis=2)
            a["yy"] += float(np.sum(y[c, m] ** 2))
            a["n"] += int(m.sum())
        for fam in PROBES:
            if fam not in z:
                continue
            y = at_times(z, fam, t).astype(np.float64)
            good = np.isfinite(y)
            for c in range(y.shape[0]):
                m = good[c]
                if m.sum() < 3:
                    continue
                a = acc.setdefault((fam, c), dict(rr=0.0, zz=0.0, rz=0.0, ry=0.0, zy=0.0, yy=0.0, n=0))
                a["rr"] = a["rr"] + np.sum(BR[:, :, m] ** 2, axis=2)
                a["zz"] = a["zz"] + np.sum(BZ[:, :, m] ** 2, axis=2)
                a["rz"] = a["rz"] + np.sum(BR[:, :, m] * BZ[:, :, m], axis=2)
                a["ry"] = a["ry"] + np.sum(BR[:, :, m] * y[c, m], axis=2)
                a["zy"] = a["zy"] + np.sum(BZ[:, :, m] * y[c, m], axis=2)
                a["yy"] += float(np.sum(y[c, m] ** 2))
                a["n"] += int(m.sum())
    geo = dict(R=Rg.tolist(), Z=Zg.tolist(), shots=n_used, flux={}, ccbv={}, obr={}, obv={})
    for (fam, c), a in sorted(acc.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if fam == "flux":
            with np.errstate(divide="ignore", invalid="ignore"):
                slope = a["xy"] / a["xx"]
                res = a["yy"] - slope * a["xy"]
            res[~np.isfinite(res)] = np.inf
            j, i = np.unravel_index(np.argmin(res), res.shape)
            geo[fam][int(c)] = dict(R=float(Rg[i]), Z=float(Zg[j]), a=float(slope[j, i]), rms=float(np.sqrt(max(res[j, i], 0) / a["n"])), y_rms=float(np.sqrt(a["yy"] / a["n"])), n=a["n"])
        else:
            det = a["rr"] * a["zz"] - a["rz"] ** 2
            with np.errstate(divide="ignore", invalid="ignore"):
                ca = (a["ry"] * a["zz"] - a["zy"] * a["rz"]) / det
                cb = (a["zy"] * a["rr"] - a["ry"] * a["rz"]) / det
                res = a["yy"] - ca * a["ry"] - cb * a["zy"]
            res[~np.isfinite(res)] = np.inf
            res[det < 1e-12 * np.nanmax(det)] = np.inf
            j, i = np.unravel_index(np.argmin(res), res.shape)
            geo[fam][int(c)] = dict(R=float(Rg[i]), Z=float(Zg[j]), a_R=float(ca[j, i]), a_Z=float(cb[j, i]), rms=float(np.sqrt(max(res[j, i], 0) / a["n"])), y_rms=float(np.sqrt(a["yy"] / a["n"])), n=a["n"])
    if not save:
        return geo
    json.dump(geo, open(os.path.join(OUT, "geometry.json"), "w"), indent=0)
    print(f"placed on {n_used} shots")
    for fam in (FLUX,) + PROBES:
        print(fam)
        for c, g in sorted(geo[fam].items(), key=lambda kv: int(kv[0])):
            co = f"a {g['a']:7.3f}" if fam == "flux" else f"a_R {g['a_R']:6.2f} a_Z {g['a_Z']:6.2f}"
            print(f"  {int(c):2d}: R {g['R']:5.2f} Z {g['Z']:6.2f}  {co}  rms {g['rms']:.4g} of {g['y_rms']:.4g}")


def read_magnetics(psi, R, Z, geo):
    """Flux loops and pickup coils from a flux map psi[z, R] on grid R, Z. Returns {family: channels}; psi in Wb per radian."""
    from scipy.interpolate import RectBivariateSpline
    spl = RectBivariateSpline(Z, R, psi, kx=3, ky=3)
    out = {}
    g = geo[FLUX]
    v = np.full(CHANNELS[FLUX], np.nan)                  # by channel number: a channel with too little real data to place stays NaN
    for c in g:
        v[int(c)] = g[c]["a"] * float(spl(g[c]["Z"], g[c]["R"])[0, 0])
    out[FLUX] = v
    for fam in PROBES:
        g = geo[fam]
        v = np.full(CHANNELS[fam], np.nan)
        for c in g:
            r, zz = g[c]["R"], g[c]["Z"]
            br = -float(spl(zz, r, dx=1)[0, 0]) / r
            bz = float(spl(zz, r, dy=1)[0, 0]) / r
            v[int(c)] = g[c]["a_R"] * br + g[c]["a_Z"] * bz
        out[fam] = v
    return out


def score_series(pred, real):
    """Error over the channel's spread within the shot, the relative error and the correlation, over finite points."""
    m = np.isfinite(pred) & np.isfinite(real)
    if m.sum() < 5:
        return None
    p, r = pred[m], real[m]
    sd = float(np.std(r))
    return dict(rms=float(np.sqrt(np.mean((p - r) ** 2))), of_spread=float(np.sqrt(np.mean((p - r) ** 2)) / max(sd, 1e-12)), rel=float(np.sqrt(np.mean((p - r) ** 2)) / max(np.mean(np.abs(r)), 1e-12)), corr=float(np.corrcoef(p, r)[0, 1]) if sd > 0 and np.std(p) > 0 else float("nan"), sd=sd)


def forward_score(pred, real, gain=False):
    """A run starts from a measured state. The sensor's offset (or gain) comes from the first third of the points; the last two thirds are scored, beside holding the first third's level."""
    m = np.isfinite(pred) & np.isfinite(real)
    if m.sum() < 12:
        return None
    p, r = pred[m], real[m]
    n = len(p) // 3
    if gain:
        b = np.mean(p[:n])
        if b <= 0:
            return None
        q = p * np.mean(r[:n]) / b
    else:
        q = p + np.mean(r[:n]) - np.mean(p[:n])
    h = np.full(len(p) - n, np.mean(r[:n]))
    a = score_series(q[n:], r[n:])
    if a is None:
        return None
    a["hold"] = float(np.sqrt(np.mean((h - r[n:]) ** 2)) / max(a["sd"], 1e-12))
    a["hold_rel"] = float(np.sqrt(np.mean((h - r[n:]) ** 2)) / max(np.mean(np.abs(r[n:])), 1e-12))
    return a


def check_magnetics(geo, which):
    rows = {f: [] for f in (FLUX,) + PROBES}
    for s in which:
        z = load(s)
        if z is None or "efit_psi" not in z:
            continue
        ok = np.isfinite(z["efit_psi"]).all(axis=(0, 1)) & (z["efit_t"] >= z["t"][0]) & (z["efit_t"] <= z["t"][-1])
        if ok.sum() < 8:
            continue
        psi, t = z["efit_psi"][:, :, ok].astype(np.float64), z["efit_t"][ok]
        pred = {f: [] for f in rows}
        for k in range(psi.shape[2]):
            o = read_magnetics(psi[:, :, k], z["efit_R"], z["efit_Z"], geo)
            for f in rows:
                pred[f].append(o[f])
        g = gains(s)
        for f in rows:
            if f not in z:
                continue
            y = at_times(z, f, t)
            P = np.array(pred[f]).T * g[f][:, None]
            for c in range(min(y.shape[0], P.shape[0])):
                q = score_series(P[c], y[c].astype(np.float64))
                if q:
                    q = dict(q, shot=s, c=c)
                    fw = forward_score(P[c], y[c].astype(np.float64))
                    if fw:
                        q.update(fw_of_spread=fw["of_spread"], fw_rel=fw["rel"], fw_corr=fw["corr"], fw_hold=fw["hold"], fw_hold_rel=fw["hold_rel"])
                    rows[f].append(q)
    return rows


# ---------- profiles on flux surfaces, the interferometer, soft X-ray
def rho_map(psi, psi_ax, psi_b):
    """Square root of normalized poloidal flux on a map."""
    return np.sqrt(np.clip((psi - psi_ax) / (psi_b - psi_ax), 0.0, 4.0))


def efit_axis_boundary(z, k):
    """Flux at the axis and at the boundary of slice k, read off the map at EFIT's own axis and last surface."""
    from scipy.interpolate import RectBivariateSpline
    spl = RectBivariateSpline(z["efit_Z"], z["efit_R"], z["efit_psi"][:, :, k].astype(np.float64), kx=3, ky=3)
    ax = float(spl(z["efit_magnetic_axis_z"][k], z["efit_magnetic_axis_r"][k])[0, 0])
    r, zz = z["efit_lcfs_r"][:, k], z["efit_lcfs_z"][:, k]
    m = np.isfinite(r) & np.isfinite(zz) & (r > 0)
    if m.sum() < 8:
        return ax, np.nan, spl
    b = float(np.median(spl(zz[m], r[m], grid=False)))
    return ax, b, spl


def thomson_on_rho(z, k_ts):
    """A Thomson slice as T_e and n_e against rho, using the EFIT slice nearest in time. Returns rho, T_e (eV), n_e (m^-3), R at Z = 0; points that are bad are left out."""
    t = z["te_t"][k_ts]
    k = int(np.argmin(np.abs(z["efit_t"] - t)))
    if abs(z["efit_t"][k] - t) > 0.012:
        return None
    ax, b, spl = efit_axis_boundary(z, k)
    if not np.isfinite(b) or b == ax:
        return None
    R = z["R_ts"]
    rho = np.sqrt(np.clip((spl(0.0, R, grid=False) - ax) / (b - ax), 0.0, 4.0))
    te, ne = z["te"][:, k_ts].astype(np.float64), z["nek"][:, k_ts].astype(np.float64)
    m = np.isfinite(te) & np.isfinite(ne) & (te > 5) & (ne > 1e17)
    if m.sum() < 15:
        return None
    return rho[m], te[m], ne[m], R[m]


def binned(rho, v):
    """Mean of v in each rho bin, in and out of the axis together; NaN where a bin is empty."""
    idx = np.digitize(rho, RHO_BINS) - 1
    out = np.full(len(RHO_BINS) - 1, np.nan)
    for j in range(len(out)):
        m = idx == j
        if m.any():
            out[j] = np.mean(v[m])
    return out


def emissivity(te_ev, ne, E_keV):
    """Bremsstrahlung and recombination seen through a filter with cut-off E: n_e^2 T^-1/2 exp(-E / T), in units of 1e38."""
    T = np.maximum(te_ev / 1e3, 1e-3)
    return (ne / 1e19) ** 2 * T ** -0.5 * np.exp(-E_keV / T)


SXR_CUTS = (0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0)
RHO_FINE = np.linspace(0.0, 1.0, 21)
IMPACTS = np.linspace(0.0, 0.95, 20)


def path_weights(impact, edges=RHO_FINE):
    """A straight sightline that comes no closer to the axis than impact, in rho: its length in each shell, as on circles. One number per chord."""
    e2 = edges ** 2
    b2 = impact ** 2
    up = np.sqrt(np.maximum(e2[1:] - b2, 0.0))
    lo = np.sqrt(np.maximum(e2[:-1] - b2, 0.0))
    return 2.0 * (up - lo)


def shell_profile(rho, v, edges=RHO_FINE):
    """v in each shell, filled across gaps; NaN if too few shells are covered."""
    idx = np.digitize(rho, edges) - 1
    n = len(edges) - 1
    o = np.full(n, np.nan)
    for j in range(n):
        m = idx == j
        if m.any():
            o[j] = np.mean(v[m])
    ok = np.isfinite(o)
    if ok.sum() < 8:
        return None
    c = 0.5 * (edges[1:] + edges[:-1])
    return np.interp(c, c[ok], o[ok])


def sxr_rows(z, E_keV):
    """For every Thomson slice of a shot: emission in each shell, and the 54 chords' reading at that time."""
    if "sxr_u" not in z or "te" not in z or "R_ts" not in z:
        return None
    X, Y, T = [], [], []
    for k in range(z["te_t"].shape[0]):
        t = z["te_t"][k]
        if t < z["t"][0] or t > z["t"][-1]:
            continue
        q = thomson_on_rho(z, k)
        if q is None:
            continue
        rho, te, ne, _ = q
        e = shell_profile(rho, emissivity(te, ne, E_keV))
        if e is None:
            continue
        y = np.concatenate([at_times(z, n, np.array([t]))[:, 0] if n in z else np.full(18, np.nan) for n in ("sxr_l", "sxr_u", "sxr_t")])
        X.append(e)
        Y.append(y)
        T.append(t)
    return (np.array(X), np.array(Y), np.array(T)) if len(X) >= 8 else None


def line_factor(z):
    """The interferometer's reading over the midplane integral of Thomson's n_e, slice by slice."""
    out = []
    if "ne" in z and "nek" in z:
        for k in range(z["te_t"].shape[0]):
            t = z["te_t"][k]
            if t < z["t"][0] or t > z["t"][-1]:
                continue
            n, R = z["nek"][:, k].astype(np.float64), z["R_ts"]
            m = np.isfinite(n) & (n > 1e17)
            if m.sum() < 30:
                continue
            o = np.argsort(R[m])
            line = float(np.trapezoid(n[m][o], R[m][o]))
            y = float(at_times(z, "ne", np.array([t]))[0])
            if np.isfinite(y) and y > 1e18 and line > 1e18:
                out.append(y / line)
    return out


def chords():
    """Each chord's closest approach, and the filter's cut-off, from how readings move within a shot. A chord's gain and the plasma's impurities are unknown and differ by shot, so a shot's own level is taken out: only its changes count."""
    fit = shots()["fit"]
    data = {E: [] for E in SXR_CUTS}
    ratio = []
    for s_ in fit:
        z = load(s_)
        if z is None:
            continue
        for E in SXR_CUTS:
            q = sxr_rows(z, E)
            if q:
                data[E].append(q)
        ratio += line_factor(z)
    ratio = np.array(ratio)
    print(f"interferometer over the midplane integral of Thomson's n_e: median {np.median(ratio):.3f}, quartiles {np.percentile(ratio, 25):.3f} to {np.percentile(ratio, 75):.3f}, n {len(ratio)}")
    paths = np.array([path_weights(b) for b in IMPACTS])
    best = None
    for E in SXR_CUTS:
        cost = np.zeros((54, len(IMPACTS)))
        count = np.zeros(54)
        for X, Y, _ in data[E]:
            M = X @ paths.T                                # slices, impacts
            for c in range(54):
                m = np.isfinite(Y[:, c])
                if m.sum() < 8 or np.std(Y[m, c]) <= 0:
                    continue
                y = Y[m, c] - np.mean(Y[m, c])
                mm = M[m] - np.mean(M[m], axis=0)
                g = np.sum(mm * y[:, None], axis=0) / np.maximum(np.sum(mm ** 2, axis=0), 1e-30)
                g = np.maximum(g, 0.0)                     # a gain cannot be negative
                cost[c] += np.sum((y[:, None] - g * mm) ** 2, axis=0) / np.sum(y ** 2)
                count[c] += 1
        keep = count > 0
        cost[keep] /= count[keep, None]
        impact = IMPACTS[np.argmin(cost, axis=1)]
        score = float(np.median(np.sqrt(np.min(cost[keep], axis=1))))
        print(f"filter cut-off {E} keV: {sum(len(a[0]) for a in data[E])} slices, within-shot error over spread, median of chords {score:.3f}")
        if best is None or score < best[0]:
            best = (score, E, impact, np.min(cost, axis=1))
    np.savez(os.path.join(OUT, "chords.npz"), impact=best[2], E=best[1], cost=best[3], edges=RHO_FINE, ne_factor=float(np.median(ratio)))
    print("kept cut-off", best[1], "keV, score", round(best[0], 3))
    for name, a, b in (("lower camera", 0, 18), ("upper camera", 18, 36), ("tangential camera", 36, 54)):
        print(f"  {name}: closest approach " + " ".join(f"{v:.2f}" for v in best[2][a:b]))
        print(f"  {'':{len(name)}s}  fit error     " + " ".join(f"{np.sqrt(v):.2f}" for v in best[3][a:b]))


def catalog():
    return json.load(open(os.path.join(ROOT, "runs", "twin", "mast_catalog", "catalog.json")))["shots"]


def h_flag(x, t):
    """1 in the catalog's H-mode windows."""
    f = np.zeros(len(t))
    for w in ((x.get("phases") or {}).get("hmode") or {}).get("windows", []):
        f[(t >= w["t0"]) & (t <= w["t1"])] = 1.0
    return f


def dalpha_base(z):
    """The brightest filter channel in 1 ms means, and its lower envelope over 5 ms: the level without bursts."""
    y = np.nanmax(z["dalpha"], axis=0).astype(np.float64)
    from scipy.ndimage import median_filter
    return y, median_filter(np.nan_to_num(y), size=5, mode="nearest")


def dalpha_rows(z, x):
    """Within one shot: ln of the level against ln line density, ln beam power and the H-mode flag, 1 ms apart, each taken from its value over the shot."""
    if "dalpha" not in z or "ne" not in z:
        return None
    t = z["t"]
    y, base = dalpha_base(z)
    ne = z["ne"].astype(np.float64)
    nbi = z["nbi"].astype(np.float64) if "nbi" in z else np.zeros(len(t))
    m = np.isfinite(ne) & (ne > 1e18) & (base > 1e-3)
    if m.sum() < 50:
        return None
    gas = np.gradient(np.nan_to_num(z["gas"].astype(np.float64)), t) if "gas" in z else np.zeros(len(t))
    gas = np.convolve(gas, np.ones(5) / 5, mode="same")
    r_out = np.interp(t, z["efit_t"], np.nan_to_num(z["efit_rpsi100_out"], nan=1.4)) if "efit_rpsi100_out" in z else np.full(len(t), 1.4)
    X = np.stack([np.log(ne[m] / 1e19), np.log(np.maximum(nbi[m], 1e5) / 1e6), h_flag(x, t)[m], np.log(np.maximum(gas[m], 1e19) / 1e20), r_out[m]], axis=1)
    return X, np.log(base[m])


def edge():
    """D-alpha: how its level moves with the plasma, and the shape of a burst. Mode tracks: their size against the catalog's field, and the quiet level. On the fit shots; the check is on the others."""
    d = shots()
    cat = catalog()
    Xs, Ys = [], []
    peaks, halfs = [], []
    amp_pairs = []
    quiet = []
    freq_pairs = []
    for s_ in d["fit"]:
        z = load(s_)
        x = cat.get(str(s_))
        if z is None or x is None:
            continue
        q = dalpha_rows(z, x)
        if q:
            Xs.append(q[0] - q[0].mean(axis=0))
            Ys.append(q[1] - q[1].mean())
        if "raw_dalpha" in z and (x.get("elms") or {}).get("t"):
            t = z["raw_dalpha_t"]
            y = np.nanmax(z["raw_dalpha"], axis=0).astype(np.float64)
            dt = float(np.median(np.diff(t)))
            for te in x["elms"]["t"]:
                i = int(np.searchsorted(t, te))
                n0, n1 = int(0.002 / dt), int(0.004 / dt)
                if i - n0 < 0 or i + n1 >= len(y):
                    continue
                b = np.median(y[i - n0:i])
                seg = y[i:i + n1]
                if b <= 1e-3 or not np.isfinite(seg).all():
                    continue
                pk = int(np.argmax(seg))
                ratio = seg[pk] / b
                if ratio < 1.5:
                    continue
                after = seg[pk:]
                below = np.flatnonzero(after <= b + 0.5 * (seg[pk] - b))
                peaks.append(ratio)
                halfs.append((below[0] if len(below) else len(after)) * dt)
        if "track_amp" in z:
            la = np.log10(np.maximum(z["track_amp"].astype(np.float64), 1e-6))
            coh = z["track_coh"] > 0
            if (~coh).sum() > 10:
                quiet.append(float(np.nanmedian(la[~coh])))
            for m_ in x.get("modes") or []:
                if m_.get("family") != "slow" or m_.get("n") != 1 or not m_.get("trace"):
                    continue
                tt = np.array([np.nan if a is None else a for a in m_["trace"]["t"]], float)
                bb = np.array([np.nan if a is None else a for a in m_["trace"]["field_Vs"]], float)
                ff = np.array([np.nan if a is None else a for a in m_["trace"]["f_kHz"]], float)
                idx = np.clip(np.round((tt - z["t"][0]) / 1e-3).astype(int), 0, len(z["t"]) - 1)
                ok = np.isfinite(bb) & (bb > 0) & np.isfinite(ff) & np.isfinite(la[idx])
                for a_, b_, f_, g_ in zip(la[idx][ok], bb[ok], ff[ok], z["track_freq"][idx][ok]):
                    amp_pairs.append((a_, np.log10(b_), np.log10(max(f_, 0.1))))
                    freq_pairs.append((f_, g_ / 1e3))
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    co = np.linalg.lstsq(X, Y, rcond=None)[0]
    print(f"D-alpha level, within a shot, on {len(Xs)} fit shots: ln D = {co[0]:.3f} ln n_line + {co[1]:.3f} ln P_beam + {co[2]:.3f} H + {co[3]:.3f} ln gas rate + {co[4]:.3f} R_out; explains {1 - np.var(Y - X @ co) / np.var(Y):.3f} of the variance within shots")
    peaks, halfs = np.array(peaks), np.array(halfs)
    print(f"D-alpha bursts, {len(peaks)} on the catalog's ELMs: peak over level median {np.median(peaks):.2f} (quartiles {np.percentile(peaks, 25):.2f} to {np.percentile(peaks, 75):.2f}), fall to half in {1e3 * np.median(halfs):.2f} ms (quartiles {1e3 * np.percentile(halfs, 25):.2f} to {1e3 * np.percentile(halfs, 75):.2f})")
    A = np.array(amp_pairs)
    M = np.stack([np.ones(len(A)), A[:, 1], A[:, 2]], axis=1)
    c2 = np.linalg.lstsq(M, A[:, 0], rcond=None)[0]
    pred = M @ c2
    print(f"mode tracks on {len(A)} frames of the catalog's slow modes: log10 amp = {c2[0]:.3f} + {c2[1]:.3f} log10 field + {c2[2]:.3f} log10 f_kHz; rms {np.std(A[:, 0] - pred):.3f} decades, correlation {np.corrcoef(pred, A[:, 0])[0, 1]:.3f}")
    fp = np.array(freq_pairs)
    print(f"   track frequency over the catalog's: median {np.median(fp[:, 1] / fp[:, 0]):.3f}, within 20% in {np.mean(np.abs(fp[:, 1] / fp[:, 0] - 1) < 0.2):.2f} of frames; quiet level log10 amp median {np.median(quiet):.3f} (quartiles {np.percentile(quiet, 25):.3f} to {np.percentile(quiet, 75):.3f})")
    json.dump(dict(dalpha=dict(ne=float(co[0]), P=float(co[1]), H=float(co[2]), gas=float(co[3]), r_out=float(co[4]), peak=float(np.median(peaks)), half_s=float(np.median(halfs))), modes=dict(a=float(c2[0]), b=float(c2[1]), c=float(c2[2]), quiet=float(np.median(quiet)))), open(os.path.join(OUT, "edge.json"), "w"), indent=1)
    # the check, on shots not used: the level from the first third, the last two thirds scored
    errs = []
    for s_ in d["check"]:
        z = load(s_)
        x = cat.get(str(s_))
        if z is None or x is None or "dalpha" not in z:
            continue
        q = dalpha_rows(z, x)
        if not q:
            continue
        a = forward_score(np.exp(q[0] @ co), np.exp(q[1]), gain=True)
        if a:
            errs.append(a)
    print(f"D-alpha level on {len(errs)} check shots, gain from the first third, last two thirds scored: error over spread median {np.median([a['of_spread'] for a in errs]):.3f} (holding {np.median([a['hold'] for a in errs]):.3f}), relative {np.median([a['rel'] for a in errs]):.3f} (holding {np.median([a['hold_rel'] for a in errs]):.3f}), correlation {np.nanmedian([a['corr'] for a in errs]):.3f}")


def read_sxr(emission, impact):
    """The chords from emission in each shell."""
    return np.array([path_weights(b) @ emission for b in impact])


def magnetics_refine():
    """Below the grid: each pickup coil's place to 5 mm, by the same fit on a finer patch of the map, on the fit shots."""
    from scipy.interpolate import RectBivariateSpline
    geo = json.load(open(os.path.join(OUT, "geometry.json")))
    fit = shots()["fit"]
    cand = {}
    for fam in PROBES:
        for c, g in geo[fam].items():
            rr = g["R"] + np.arange(-6, 7) * 0.005
            zz = g["Z"] + np.arange(-12, 13) * 0.005
            cand[(fam, c)] = dict(rr=rr, zz=zz, rr_=np.zeros((len(zz), len(rr))), zz_=np.zeros((len(zz), len(rr))), rz=np.zeros((len(zz), len(rr))), ry=np.zeros((len(zz), len(rr))), zy=np.zeros((len(zz), len(rr))), yy=0.0, n=0)
    for s_ in fit[:25]:
        z = load(s_)
        if z is None or "efit_psi" not in z:
            continue
        ok = np.where(np.isfinite(z["efit_psi"]).all(axis=(0, 1)) & (z["efit_t"] >= z["t"][0]) & (z["efit_t"] <= z["t"][-1]))[0]
        for k in ok[::2]:
            spl = RectBivariateSpline(z["efit_Z"], z["efit_R"], z["efit_psi"][:, :, k].astype(np.float64), kx=3, ky=3)
            t = np.array([z["efit_t"][k]])
            for (fam, c), a in cand.items():
                if fam not in z:
                    continue
                y = float(at_times(z, fam, t)[int(c), 0])
                if not np.isfinite(y):
                    continue
                dz = spl(a["zz"], a["rr"], dx=1)
                dr = spl(a["zz"], a["rr"], dy=1)
                BR, BZ = -dz / a["rr"][None, :], dr / a["rr"][None, :]
                a["rr_"] += BR ** 2
                a["zz_"] += BZ ** 2
                a["rz"] += BR * BZ
                a["ry"] += BR * y
                a["zy"] += BZ * y
                a["yy"] += y * y
                a["n"] += 1
        print(s_, flush=True)
    for (fam, c), a in cand.items():
        if a["n"] < 20:
            continue
        det = a["rr_"] * a["zz_"] - a["rz"] ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            ca = (a["ry"] * a["zz_"] - a["zy"] * a["rz"]) / det
            cb = (a["zy"] * a["rr_"] - a["ry"] * a["rz"]) / det
            res = a["yy"] - ca * a["ry"] - cb * a["zy"]
        res[~np.isfinite(res)] = np.inf
        j, i = np.unravel_index(np.argmin(res), res.shape)
        geo[fam][c].update(R=float(a["rr"][i]), Z=float(a["zz"][j]), a_R=float(ca[j, i]), a_Z=float(cb[j, i]), rms=float(np.sqrt(max(res[j, i], 0) / a["n"])))
    json.dump(geo, open(os.path.join(OUT, "geometry.json"), "w"), indent=0)
    for fam in PROBES:
        print(fam, "rms " + " ".join(f"{g['rms']:.4f}" for c, g in sorted(geo[fam].items(), key=lambda kv: int(kv[0])))[:300])
        print(fam, "a_Z " + " ".join(f"{g['a_Z']:.2f}" for c, g in sorted(geo[fam].items(), key=lambda kv: int(kv[0])))[:300])


def check():
    geo = json.load(open(os.path.join(OUT, "geometry.json")))
    d = shots()
    fit_some, chk = d["fit"][:15], d["check"]
    summary = {}
    for name, which in (("fit", fit_some), ("check", chk)):
        rows = check_magnetics(geo, which)
        for f, r in rows.items():
            if r:
                summary[f"{f} {name}"] = dict(shots=len({a['shot'] for a in r}), of_spread=float(np.median([a["of_spread"] for a in r])), of_spread_p90=float(np.percentile([a["of_spread"] for a in r], 90)), rel=float(np.median([a["rel"] for a in r])), corr=float(np.nanmedian([a["corr"] for a in r])))
                print(f"{f:5s} on {name:5s} shots: error over spread median {summary[f'{f} {name}']['of_spread']:.3f} (90th {summary[f'{f} {name}']['of_spread_p90']:.3f}), relative {summary[f'{f} {name}']['rel']:.4f}, correlation {summary[f'{f} {name}']['corr']:.4f}, {summary[f'{f} {name}']['shots']} shots", flush=True)
                fw = [a for a in r if "fw_of_spread" in a]
                if fw:
                    summary[f"{f} {name} forward"] = dict(of_spread=float(np.median([a["fw_of_spread"] for a in fw])), rel=float(np.median([a["fw_rel"] for a in fw])), hold=float(np.median([a["fw_hold"] for a in fw])), hold_rel=float(np.median([a["fw_hold_rel"] for a in fw])), corr=float(np.nanmedian([a["fw_corr"] for a in fw])))
                    v = summary[f"{f} {name} forward"]
                    print(f"      offset from the first third, last two thirds scored: error over spread {v['of_spread']:.3f} (holding {v['hold']:.3f}), relative {v['rel']:.4f} (holding {v['hold_rel']:.4f}), correlation {v['corr']:.4f}", flush=True)
    ch = np.load(os.path.join(OUT, "chords.npz"))
    impact, E, ne_f = ch["impact"], float(ch["E"]), float(ch["ne_factor"])
    paths = np.array([path_weights(b) for b in impact])
    for name, which in (("fit", fit_some), ("check", chk)):
        sx, inter = [], []
        for s_ in which:
            z = load(s_)
            if z is None:
                continue
            q = sxr_rows(z, E)
            if q:
                X, Y, _ = q
                P = X @ paths.T
                for c in range(54):
                    a = forward_score(P[:, c], Y[:, c].astype(np.float64), gain=True)
                    if a:
                        sx.append(dict(a, shot=s_, c=c))
            if "ne" in z and "nek" in z:
                P, Y = [], []
                for k in range(z["te_t"].shape[0]):
                    t = z["te_t"][k]
                    if t < z["t"][0] or t > z["t"][-1]:
                        continue
                    n, R = z["nek"][:, k].astype(np.float64), z["R_ts"]
                    m = np.isfinite(n) & (n > 1e17)
                    if m.sum() < 30:
                        continue
                    o = np.argsort(R[m])
                    P.append(ne_f * float(np.trapezoid(n[m][o], R[m][o])))
                    Y.append(float(at_times(z, "ne", np.array([t]))[0]))
                a = score_series(np.array(P), np.array(Y)) if len(P) > 8 else None
                if a:
                    inter.append(dict(a, shot=s_))
        if sx:
            print(f"soft X-ray on {name:5s} shots, gain from the first third, last two thirds scored: error over spread median {np.median([a['of_spread'] for a in sx]):.3f} (holding {np.median([a['hold'] for a in sx]):.3f}), relative {np.median([a['rel'] for a in sx]):.3f} (holding {np.median([a['hold_rel'] for a in sx]):.3f}), correlation {np.nanmedian([a['corr'] for a in sx]):.3f}, {len({a['shot'] for a in sx})} shots")
            core = [a for a in sx if impact[a['c']] <= 0.3]
            if core:
                print(f"   chords passing inside rho 0.3: error over spread {np.median([a['of_spread'] for a in core]):.3f} (holding {np.median([a['hold'] for a in core]):.3f}), relative {np.median([a['rel'] for a in core]):.3f} (holding {np.median([a['hold_rel'] for a in core]):.3f}), correlation {np.nanmedian([a['corr'] for a in core]):.3f}")
        if inter:
            print(f"interferometer on {name:5s} shots: error over spread median {np.median([a['of_spread'] for a in inter]):.3f} (90th {np.percentile([a['of_spread'] for a in inter], 90):.3f}), relative {np.median([a['rel'] for a in inter]):.3f}, correlation {np.nanmedian([a['corr'] for a in inter]):.3f}, {len(inter)} shots")
    json.dump(summary, open(os.path.join(OUT, "level1.json"), "w"), indent=1)


if __name__ == "__main__":
    {"place": lambda: place(), "refine": magnetics_refine, "chords": chords, "edge": edge, "check": check}[sys.argv[1]]()
