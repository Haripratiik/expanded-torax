"""MAST shots in the twin (H36, docs/digital-twin.md): the first accuracy test of TORAX against real profiles, on FAIR-MAST TokaMark v1 data.

Two sides, because the store is zarr and TORAX's venv has no zarr:
  select, convert   run in the zarr environment (XTORAX_FAIRMAST_PYTHON), read the store read-only and write into runs/twin/mast/
  backtest, report  run in TORAX's venv and read only runs/twin/mast/

Selection rule, fixed before any TORAX run on a selected shot (RULE below; `select` writes it and its result to selection.json): shots of manifest.json with Thomson, EFIT, summary and charge
exchange on the common 5 ms time base, none of TokaMark's QC flags set, and a window of at least 150 ms in which Ip, NBI power, line density, stored energy and the core T_e and T_i stay steady with the magnetic
axis within 3 cm of the midplane. Walk down by window length, skip a shot within 100 shot numbers of one already taken (one per session), stop at 8. How the rule got here: the first version asked for 200 ms and
steady Ip, NBI, density and W_dia. It passed development shot 21901, whose T_e doubled in the window while W_dia moved by 30%, so core T_e and T_i were added and the window lowered to 150 ms to keep enough
shots. Converting that list showed that plasmas shifted downwards (Z_mag down to -25 cm) put the Thomson chord at z = 0 up to psi_norm 0.19 away from the axis, so the core is not measured and EFIT's q(R) misses the
axis; hence the limit on Z_mag. Those fits were looked at; nothing was run on them. Shots 21901 and 20092 served for debugging and for the two calibrated settings below and are excluded from the selection.

Conversion (convert): each slice of the window on its own, no smoothing in time, so a profile at a cut carries nothing from after it.
  flux surfaces   EFIT's psi(R, Z) on a spline; contours give the integral of dl / (R |grad psi|) on each surface; F(psi_norm) comes from EFIT's midplane q
                  over psi_norm < 0.85 (cubic in F^2, tied to the vacuum value at the boundary) and q = F times that integral, because the midplane q is too
                  coarsely sampled near the separatrix (checked against EFIT's q95, axis field and the measured plasma current: "check" in conversion.json)
  rho_norm        sqrt(toroidal flux / flux at psi_norm 0.99): TORAX's EQDSK reader stops at that surface, so its rho_norm = 1 is the same surface here
  profiles        Thomson T_e and n_e and charge-exchange T_i (carbon) mapped through psi(R, z=0); a cubic spline per slice, outliers rejected, with shot.fit_profiles's knot rule, zero slope at the axis
  T_i             edge chords can read tens of keV in some slices, in clusters that outlier rejection cannot remove: chords above 1.5 times the median of those inside rho 0.4 are dropped, and a slice whose mean over the band
                  leaves the median of its neighbours (4 slices either side) by more than 25% is replaced by interpolation in time. T_i is kept only if at least 90% of slices have 30 chords reaching rho < 0.4 and rho > 0.6.
                  This cleaning was added after a first full run, whose T_i errors of about 1000% in two shots led to those chords; no transport setting changed with it
Shot npz: the fields of shot.Shot (T_e, T_i keV, n_e m^-3, Ip A, P_heat = injected NBI MW) plus geometry and checks. shot.Shot.load ignores the extras.

Backtest settings, all fixed in advance (FIXED and TRANSPORT below), none per shot:
  geometry        EFIT at the cut, held for the run, through TORAX's EQDSK reader (COCOS 11, 257 x 513 grid resampled from EFIT's 65 x 65)
  boundary        T_e, T_i, n_e pinned at rho_norm 0.91 (TORAX's pedestal model) and set at rho_norm 1, from the measured fits (trailing mean of 5 slices), also during the held-back part
                  (`--frozen` holds them at their value at the cut instead). The pin is outside the scored band 0.1 < rho < 0.9
  initial state   the mean of the last 5 slices up to the cut (20 ms span), nothing later; q from EFIT's psi(rho)
  NBI             measured power, 70% absorbed, Gaussian at rho 0 width 0.3, 60% to electrons, fuelling at 60 keV per particle: assumptions, none of them in the store (`--nbi_width` tests the shape).
                  No beam current drive, no radiation. Ohmic heating, bootstrap current, resistive current diffusion at the real resistivity
  plasma          D with C, Z_eff 1.5 (not in the store), Angioni-Sauter neoclassical transport on top of every model
  transport       qlknn: TORAX's default QLKNN_7_11 with the settings of its ITER hybrid example, unchanged; qlknn_clip: the same with inputs clipped to the training range; bgb: Bohm-gyroBohm, TORAX's default coefficients;
                  constant: chi = 1 m^2/s, D = 1 m^2/s, V = -0.33 m/s, TORAX's defaults; bgb_dev, constant_dev: the same two with one global number (multiplier on all four Bohm and gyroBohm terms, chi) chosen by `calibrate`
                  on the two development shots, which are not scored here (calibration.json: multiplier 6, chi 8 m^2/s)
  time            fixed 1 ms step, 1/5 to 1/50 of a confinement time (dev shots: a step of 0.25 ms or Newton's solver moves T_e by 1 to 7%). Cuts every 20 ms from the fifth slice of the window while 60 ms remain; each run goes to the window's end
A run is scored with shot.score and shot.trust_window against the fits averaged over 5 slices (20 ms span) centred on each time; against the bare per-slice fits as well ("raw"), whose scatter of several percent ends
most trust windows at once. Persistence, the profile at the cut held for ever, is scored the same way: on a steady flat top it is the number to beat.
tau_E = W_th / (0.7 P_NBI + V_loop Ip - dW_th/dt), median over the window, with W_th from the fits and EFIT's volume: the thermal energy, the only energy TORAX evolves. EFIT's stored energy W_dia is up to 4.5 times
larger (fast ions and magnetics error, inferred), so tau_E from W_dia is saved as well (tau_E_wdia_s).
Usage (the zarr environment, read-only on the store, from the repository; never writes to the store):
  $XTORAX_FAIRMAST_PYTHON core/mast.py select
  $XTORAX_FAIRMAST_PYTHON core/mast.py convert [--dev]
Usage (TORAX's environment, from the repository):
  python core/mast.py calibrate
  python core/mast.py closure
  python core/mast.py backtest [--shots 27552 ...] [--models qlknn bgb constant bgb_dev constant_dev qlknn_clip] [--frozen] [--geometry circular] [--nbi_width 0.5] [--tag name]
  python core/mast.py qlknn_range
  python core/mast.py report
Outputs in runs/twin/mast/: selection.json, conversion.json, calibration.json, closure.json, qlknn_range.json, shots/<shot>.npz, results/<shot>_<model>[_tag].json, runs/<shot>_<model>_<cut in ms>[_tag].npz, results.json, te_profiles.png, error_vs_lead.png.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.environ.get("XTORAX_HOME") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the workspace: runs/twin and external/ (README, "Where things live")

def torax_example(name):
    """TORAX's own example file: from a source checkout in external/torax if there is one, else from the installed package."""
    local = os.path.join(ROOT, "external", "torax", "torax", "examples", name)
    if os.path.exists(local):
        return local
    import importlib.util
    return os.path.join(list(importlib.util.find_spec("torax").submodule_search_locations)[0], "examples", name)

OUT = os.path.join(ROOT, "runs", "twin", "mast")
STORE = os.path.expanduser(os.environ.get("XTORAX_MAST_STORE", "~/data/mast/tokamark_v1"))   # FAIR-MAST's TokaMark v1 store (zarr)
META = os.path.expanduser("~/data/mast/meta")
MU0, QE = 4e-7 * np.pi, 1.602176634e-19
PSI_LAST, N_SURF = 0.99, 100
RHO = np.linspace(0.0, 1.0, 101)
PN = np.linspace(0.0, PSI_LAST, N_SURF)
X_MAX = 1.03                       # the profile fits run a little past the separatrix (sqrt(psi_norm) up to X_MAX), so rho_norm = 1 is read from data, not extrapolated

RULE = dict(
    qc_flags_false=["thomson_max_outlier", "thomson_nan_count_outlier", "ip_negative", "efit_beta_pol_outlier", "efit_x_point_outlier"],
    ip_min_A=3e5, ip_tol=0.05, pnbi_min_W=1e6, pnbi_tol=0.15, nel_tol=0.20, wdia_tol=0.20, core_tol=0.20, zmag_max_m=0.03, thomson_points_min=100, cx_chords_min=50,
    window_min_s=0.150, session_gap_shots=100, n_shots=8, n_dev=0, excluded_already_used_for_debugging=[21901, 20092],
    text="Longest run of consecutive 5 ms slices where every slice has Ip >= 300 kA, NBI >= 1 MW, >= 100 valid Thomson points, >= 50 valid CX chords, "
         "finite EFIT q95 and stored energy, the magnetic axis within 3 cm of the midplane (the Thomson chord is at z = 0 and otherwise misses the core), and where over the run Ip stays within +-5%, NBI within +-15%, line density and stored energy within "
         "+-20% of their medians (spread <= 2 tol median), and the Thomson T_e and CX T_i near the axis (median within 15 cm of it, then a 5-slice running median) within +-20%. Window >= 150 ms. None of the TokaMark QC flags set. Rank by window length, skip a shot "
         "within 100 shot numbers of one already taken, take 8. Shots 21901 and 20092 were used for debugging under an earlier version of the rule and are excluded.")

FIXED = dict(n_rho=50, dt=1e-3, rho_ped=0.91, f_abs=0.7, f_elec=0.6, E_beam_eV=60e3, nbi_center=0.0, nbi_width=0.3, Z_eff=1.5, cut_step_s=0.02, min_lead_s=0.06, trailing_slices=5, truth_slices=5,
             tol=0.10, leads_s=[0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3], leads_tau=[0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0])
LINEAR_SOLVER = dict(solver_type="linear", use_predictor_corrector=True, n_corrector_steps=1, chi_pereverzev=30, D_pereverzev=15, use_pereverzev=True)
TRANSPORT = dict(bgb=dict(model_name="bohm-gyrobohm"), constant=dict(model_name="constant", chi_i=1.0, chi_e=1.0, D_e=1.0, V_e=-0.33))     # qlknn comes from TORAX's example
DEV = [dict(shot=21901, t0=0.1690, t1=0.3740, campaign="M7", shape="", heating=""), dict(shot=20092, t0=0.1414, t1=0.3414, campaign="M7", shape="", heating="")]
BGB_TERMS = ("chi_e_bohm_multiplier", "chi_i_bohm_multiplier", "chi_e_gyrobohm_multiplier", "chi_i_gyrobohm_multiplier")


def _cell_mean(t_fast, v, t_grid):
    """Mean of a fast signal over each cell of the slow grid."""
    dt = np.median(np.diff(t_grid))
    idx = np.searchsorted(np.r_[t_grid - dt / 2, t_grid[-1] + dt / 2], t_fast) - 1
    ok = (idx >= 0) & (idx < len(t_grid)) & np.isfinite(v)
    n = np.bincount(idx[ok], minlength=len(t_grid))
    return np.where(n > 0, np.bincount(idx[ok], v[ok], len(t_grid)) / np.maximum(n, 1), np.nan)


def _running_median(v, half=2):
    """Median over 5 slices, to take the scatter of single Thomson and charge-exchange slices out of the steadiness test."""
    return np.array([np.nanmedian(v[max(0, i - half):i + half + 1]) if np.isfinite(v[max(0, i - half):i + half + 1]).any() else np.nan for i in range(len(v))])


def scalars(root):
    """Per-slice arrays on the Thomson time base, or None if a group is missing or not on that base."""
    for g in ("thomson_scattering", "equilibrium", "summary", "charge_exchange"):
        if g not in root:
            return None
    ts, eq, su, cx = root["thomson_scattering"], root["equilibrium"], root["summary"], root["charge_exchange"]
    if "power_nbi" not in su or "line_average_n_e" not in su:
        return None
    t = ts["time"][:]
    if eq["time"].shape[0] != len(t) or cx["time"].shape[0] != len(t) or not (np.allclose(eq["time"][:], t) and np.allclose(cx["time"][:], t)):
        return None
    te, ne, ti = ts["t_e"][:], ts["n_e"][:], cx["t_i"][:]
    tf = su["time"][:]
    Rmag = eq["magnetic_axis_r"][:]
    near = lambda R, v: np.array([np.nanmedian(v[np.abs(R - Rmag[k]) < 0.15, k]) if np.isfinite(Rmag[k]) and np.isfinite(v[np.abs(R - Rmag[k]) < 0.15, k]).any() else np.nan for k in range(len(t))])
    return dict(t=t, ip=_cell_mean(tf, su["ip"][:], t), pnbi=_cell_mean(tf, su["power_nbi"][:], t), nel=_cell_mean(tf, su["line_average_n_e"][:], t),
                wdia=eq["whmd"][:], q95=eq["q95"][:], zmag=eq["magnetic_axis_z"][:], nts=(np.isfinite(te) & np.isfinite(ne) & (te > 0) & (ne > 0)).sum(axis=0),
                ncx=(np.isfinite(ti) & (ti > 0)).sum(axis=0), te_core=_running_median(near(ts["major_radius"][:], te)), ti_core=_running_median(near(cx["major_radius"][:], ti)))


def steady_window(s, rule=RULE):
    """(i, j) of the longest run of slices that satisfies the rule, inclusive."""
    ok = ((np.nan_to_num(s["ip"]) >= rule["ip_min_A"]) & (np.nan_to_num(s["pnbi"]) >= rule["pnbi_min_W"]) & (s["nts"] >= rule["thomson_points_min"])
          & (s["ncx"] >= rule["cx_chords_min"]) & np.isfinite(s["q95"]) & (np.nan_to_num(s["wdia"]) > 0) & (np.nan_to_num(s["nel"]) > 0)
          & (np.nan_to_num(s["te_core"]) > 0) & (np.nan_to_num(s["ti_core"]) > 0) & (np.abs(np.nan_to_num(s["zmag"], nan=9.0)) <= rule["zmag_max_m"]))
    tol = dict(ip=rule["ip_tol"], pnbi=rule["pnbi_tol"], nel=rule["nel_tol"], wdia=rule["wdia_tol"], te_core=rule["core_tol"], ti_core=rule["core_tol"])
    best, i = (0, -1), 0
    for j in range(len(ok)):
        if not ok[j]:
            i = j + 1
            continue
        while i < j and not all(np.ptp(s[k][i:j + 1]) <= 2 * v * np.median(s[k][i:j + 1]) for k, v in tol.items()):
            i += 1
        if j - i > best[1] - best[0]:
            best = (i, j)
    return best


def select(out=OUT):
    import zarr
    os.makedirs(out, exist_ok=True)
    meta = {int(r["shot_id"]): r for r in csv.DictReader(open(os.path.join(META, "shot_summary.csv"), newline=""))}
    shots = [s["shot"] for s in json.load(open(os.path.join(STORE, "manifest.json")))["shots"]]
    eligible, dropped = [], dict(missing_groups=0, qc_flag=0, short_window=0)
    for shot in shots:
        s = scalars(zarr.open_group(os.path.join(STORE, f"{shot}.zarr"), mode="r"))
        if s is None:
            dropped["missing_groups"] += 1
            continue
        m = meta.get(shot, {})
        if shot in RULE["excluded_already_used_for_debugging"]:
            continue
        if any(m.get(k) == "True" for k in RULE["qc_flags_false"]):
            dropped["qc_flag"] += 1
            continue
        i, j = steady_window(s)
        length = float(s["t"][j] - s["t"][i]) if j > i else 0.0
        if length < RULE["window_min_s"] - 1e-9:
            dropped["short_window"] += 1
            continue
        a = slice(i, j + 1)
        eligible.append(dict(shot=shot, campaign=m.get("campaign"), shape=m.get("plasma_shape"), heating=m.get("heating"), t0=float(s["t"][i]), t1=float(s["t"][j]),
                             window_s=round(length, 4), ip_kA=float(np.median(s["ip"][a]) / 1e3), pnbi_MW=float(np.median(s["pnbi"][a]) / 1e6),
                             nel_1e19=float(np.median(s["nel"][a]) / 1e19), wdia_kJ=float(np.median(s["wdia"][a]) / 1e3), q95=float(np.median(s["q95"][a])),
                             objective=m.get("shot_objective", "")[:200], postshot=m.get("postshot_description", "").strip()[:200]))
    eligible.sort(key=lambda e: (-e["window_s"], e["shot"]))
    taken = []
    for e in eligible:
        if all(abs(e["shot"] - t["shot"]) > RULE["session_gap_shots"] for t in taken):
            taken.append(e)
        if len(taken) == RULE["n_shots"] + RULE["n_dev"]:
            break
    res = dict(rule=RULE, n_manifest=len(shots), dropped=dropped, n_eligible=len(eligible), shots=taken[:RULE["n_shots"]], dev=taken[RULE["n_shots"]:], eligible=eligible)
    json.dump(res, open(os.path.join(out, "selection.json"), "w"), indent=1)
    print(f"{len(shots)} shots, dropped {dropped}, eligible {len(eligible)}")
    for name, group in (("selected", res["shots"]), ("dev", res["dev"])):
        for e in group:
            print(name, e["shot"], e["campaign"], e["shape"], e["heating"], f"t {e['t0']:.3f}-{e['t1']:.3f} ({e['window_s'] * 1e3:.0f} ms)", f"Ip {e['ip_kA']:.0f} kA, NBI {e['pnbi_MW']:.2f} MW, "
                  f"nel {e['nel_1e19']:.2f}e19, W {e['wdia_kJ']:.0f} kJ, q95 {e['q95']:.2f}")


def flux_surfaces(R, Z, psi, axis, lcfs, q_mid, prof, rin, rout, q0, F_vac):
    """The flux surfaces of one EFIT slice (psi in Wb/rad, [z, R]), on PN. See the docstring at the top."""
    from scipy.integrate import cumulative_trapezoid
    from scipy.interpolate import RectBivariateSpline
    import contourpy
    spl = RectBivariateSpline(Z, R, psi, kx=3, ky=3)
    ax, bd = float(spl.ev(axis[1], axis[0])), float(np.median(spl.ev(lcfs[1], lcfs[0])))
    sgn = np.sign(bd - ax)
    total = sgn * 2 * np.pi * (bd - ax)
    Rf, Zf = np.linspace(R[0], R[-1], 257), np.linspace(Z[0], Z[-1], 513)
    wb = sgn * 2 * np.pi * (spl(Zf, Rf) - ax)
    RR, ZZ = np.meshgrid(Rf, Zf)
    inside = (RR > lcfs[0].min() - 0.01) & (RR < lcfs[0].max() + 0.01) & (ZZ > lcfs[1].min() - 0.01) & (ZZ < lcfs[1].max() + 0.01)
    gen = contourpy.contour_generator(Rf, Zf, np.ma.masked_where(~inside, wb))
    oint_q, oint_v = np.zeros(N_SURF), np.zeros(N_SURF)
    for i in range(1, N_SURF):
        lines = gen.create_contour(PN[i] * total)
        if not lines:                                  # no surface at this level in this slice (EFIT's axis off its own minimum): filled from its neighbours below
            oint_q[i], oint_v[i] = np.nan, np.nan
            continue
        c = max(lines, key=len)
        dl = np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1]))
        xm, ym = 0.5 * (c[1:, 0] + c[:-1, 0]), 0.5 * (c[1:, 1] + c[:-1, 1])
        grad = 2 * np.pi * np.hypot(spl.ev(ym, xm, dx=1), spl.ev(ym, xm, dy=1))
        oint_q[i], oint_v[i] = np.sum(dl / (xm * grad)), np.sum(2 * np.pi * xm * dl / grad)
    I_last = np.sum(grad / (2 * np.pi * xm) * dl) / MU0
    for v in (oint_q, oint_v):
        missing = ~np.isfinite(v)
        missing[0] = False
        if missing.any():
            v[missing] = np.interp(PN[missing], PN[1:][~missing[1:]], v[1:][~missing[1:]])
    oint_q[0], oint_v[0] = 2 * oint_q[1] - oint_q[2], 2 * oint_v[1] - oint_v[2]
    Rq = rin + prof * (rout - rin)
    pq = (spl.ev(0 * Rq, Rq) - ax) / (bd - ax)
    use = (pq > 0.02) & (pq < 0.85) & np.isfinite(q_mid)
    coef = np.polyfit(np.r_[pq[use], 1.0], np.r_[(q_mid[use] / np.interp(pq[use], PN, oint_q)) ** 2, F_vac ** 2], 3, w=np.r_[np.ones(use.sum()), 10.0])
    F = np.sqrt(np.polyval(coef, PN))
    q = F * oint_q
    q[0] = q0
    Phi = cumulative_trapezoid(q, PN * total, initial=0.0)
    return dict(spl=spl, ax=ax, bd=bd, total=total, q=q, F=F, Phi_b=float(Phi[-1]), rho=np.sqrt(Phi / Phi[-1]), V=cumulative_trapezoid(oint_v, PN * total, initial=0.0), I_last=float(I_last),
                pn=lambda r, z=0.0: (spl.ev(z + 0 * np.asarray(r), r) - ax) / (bd - ax))


def smooth_fit(x, y, grid, knots=6, lam=1e-3, x_max=X_MAX):
    """Cubic spline through one slice against x = sqrt(psi_norm): isolated outliers dropped against a 9-point running median, then shot.fit_profiles's knot rule, zero slope at the axis,
    Huber weights with a hard cut at 4 sigma, a weak curvature penalty for gaps."""
    from scipy.interpolate import BSpline
    from scipy.ndimage import median_filter
    order = np.argsort(x)
    x, y = x[order], y[order]
    r0 = y - median_filter(y, size=9, mode="mirror")
    keep = np.abs(r0) <= 5 * max(1.4826 * np.median(np.abs(r0)), 1e-3 * np.median(np.abs(y)))
    x, y = x[keep], y[keep]
    inner = x_max * (1 - (1 - np.linspace(0.0, 1.0, knots + 2)[1:-1]) ** 1.5)
    tk = np.r_[[0.0] * 4, inner, [x_max] * 4]
    A = BSpline.design_matrix(np.clip(x, 0.0, x_max), tk, 3).toarray()
    n = A.shape[1]
    D = np.diff(np.eye(n), 2, axis=0)
    axis = np.zeros((1, n))
    axis[0, :2] = -1.0, 1.0
    scale = max(np.median(np.abs(y)), 1e-30)
    w = np.ones(len(y))
    for _ in range(5):
        lhs = np.vstack([A * np.sqrt(w)[:, None], lam * np.sqrt(len(y)) * D, 1e3 * axis])
        c = np.linalg.lstsq(lhs, np.r_[y / scale * np.sqrt(w), np.zeros(n - 2), 0.0], rcond=None)[0] * scale
        r = y - A @ c
        s = max(1.4826 * np.median(np.abs(r)), 1e-6 * scale)
        w = np.minimum(1.0, 1.5 * s / np.maximum(np.abs(r), 1e-30)) * (np.abs(r) < 4 * s)
    return BSpline(tk, c, 3)(grid), float(np.sqrt(np.mean((r[w > 0] / scale) ** 2)))


def elm_count(root, t0, t1):
    """Spikes in D-alpha over the window, on the channel with the largest spread: a tag for the report, not part of the selection."""
    try:
        sp = root["spectrometer_visible"]
        t = sp["time"][:]
        i, j = np.searchsorted(t, [t0, t1])
        v = sp["filter_spectrometer_dalpha_voltage"][:, i:j]
        ch = int(np.nanargmax(np.nanstd(v, axis=1)))
        x = np.nan_to_num(v[ch])
        from scipy.signal import find_peaks, medfilt
        base = medfilt(x, 251)
        s = 1.4826 * np.median(np.abs(x - base))
        peaks, _ = find_peaks(x, height=base + np.maximum(5 * s, 0.2 * np.abs(base)), distance=25)
        return int(len(peaks))
    except Exception:
        return None


def tau_confinement(t, W, P_nbi_W, V_loop, Ip, f_abs=FIXED["f_abs"]):
    """Median over the window of W / (f_abs P_NBI + V_loop Ip - dW/dt); dW/dt from a straight line over +-25 ms."""
    dW = np.array([np.polyfit(t[max(0, i - 5):i + 6], W[max(0, i - 5):i + 6], 1)[0] for i in range(len(t))])
    return float(np.median(W / (f_abs * P_nbi_W + np.abs(V_loop) * Ip - dW))), float(np.median(f_abs * P_nbi_W + np.abs(V_loop) * Ip - dW))


def convert(entry, out):
    import zarr
    shot, t0, t1 = entry["shot"], entry["t0"], entry["t1"]
    root = zarr.open_group(os.path.join(STORE, f"{shot}.zarr"), mode="r")
    ts, eq, su = root["thomson_scattering"], root["equilibrium"], root["summary"]
    cx = root["charge_exchange"] if "charge_exchange" in root else None                # some shots have none: then T_i = T_e throughout, and the shot is not scored on T_i
    t_all = ts["time"][:]
    i0, i1 = int(np.searchsorted(t_all, t0 - 1e-6)), int(np.searchsorted(t_all, t1 + 1e-6))
    t = t_all[i0:i1]
    nt = len(t)
    R_ts, R_cx = ts["major_radius"][:].astype(float), cx["major_radius"][:].astype(float) if cx is not None else np.array([1.0])
    te_raw, ne_raw = ts["t_e"][:, i0:i1] / 1e3, ts["n_e"][:, i0:i1]
    ti_raw = cx["t_i"][:, i0:i1] / 1e3 if cx is not None else np.full((1, i1 - i0), np.nan)
    R, Z = eq["major_radius"][:].astype(float), eq["z"][:].astype(float)
    psi = eq["psi"][:, :, i0:i1]
    prof, q_mid = eq["profile_r"][:].astype(float), eq["q"][:, i0:i1]
    get = lambda k: eq[k][i0:i1].astype(float)
    R_mag, Z_mag, rin, rout, q0, q95 = get("magnetic_axis_r"), get("magnetic_axis_z"), get("rpsi100_in"), get("rpsi100_out"), get("q_axis"), get("q95")
    bvac, bphi, W_dia, V_loop = np.abs(get("bvac_rmag")), np.abs(get("bphi_rmag")), get("whmd"), get("vloop_static")
    lcfs_r, lcfs_z = eq["lcfs_r"][:, i0:i1], eq["lcfs_z"][:, i0:i1]
    tf = su["time"][:]
    Ip, P = _cell_mean(tf, su["ip"][:], t_all)[i0:i1], _cell_mean(tf, su["power_nbi"][:], t_all)[i0:i1]
    nel = _cell_mean(tf, su["line_average_n_e"][:], t_all)[i0:i1]
    T_e, n_e, T_i = np.zeros((nt, len(RHO))), np.zeros((nt, len(RHO))), np.full((nt, len(RHO)), np.nan)
    q_psi, F_psi, rho_psi, V_rho, psi_rho = np.zeros((nt, N_SURF)), np.zeros((nt, N_SURF)), np.zeros((nt, N_SURF)), np.zeros((nt, len(RHO))), np.zeros((nt, len(RHO)))
    rho_ts, rho_cx = np.full((nt, len(R_ts)), np.nan), np.full((nt, len(R_cx)), np.nan)
    rms = dict(T_e=np.zeros(nt), n_e=np.zeros(nt), T_i=np.full(nt, np.nan))
    check = dict(q95=np.zeros(nt), F_axis=np.zeros(nt), Ip=np.zeros(nt), psi_ax=np.zeros(nt), psi_b=np.zeros(nt), Phi_b=np.zeros(nt), in_out_Te=np.zeros(nt))
    ti_ok, te_ok = np.zeros(nt, bool), np.ones(nt, bool)
    for k in range(nt):
        ok = np.isfinite(lcfs_r[:, k]) & np.isfinite(lcfs_z[:, k])
        lc = (lcfs_r[ok, k].astype(float), lcfs_z[ok, k].astype(float))
        fs = flux_surfaces(R, Z, psi[:, :, k], (R_mag[k], Z_mag[k]), lc, q_mid[:, k], prof, rin[k], rout[k], q0[k], bvac[k] * R_mag[k])
        q_psi[k], F_psi[k], rho_psi[k] = fs["q"], fs["F"], fs["rho"]
        V_rho[k] = np.interp(RHO, fs["rho"], fs["V"])
        psi_rho[k] = np.interp(RHO ** 2, fs["rho"] ** 2, PN) * fs["total"]
        check["q95"][k], check["F_axis"][k], check["Ip"][k] = np.interp(0.95, PN, fs["q"]) / q95[k] - 1, fs["F"][0] / (bphi[k] * R_mag[k]) - 1, fs["I_last"] / Ip[k] - 1
        check["psi_ax"][k], check["psi_b"][k], check["Phi_b"][k] = fs["ax"], fs["bd"], fs["Phi_b"]
        grid = np.interp(RHO, fs["rho"], np.sqrt(PN))
        pn_ts = np.clip(fs["pn"](R_ts), 0, None)
        rho_ts[k] = np.where(pn_ts <= PSI_LAST, np.interp(pn_ts, PN, fs["rho"]), np.nan)
        good = (np.sqrt(pn_ts) <= X_MAX) & np.isfinite(te_raw[:, k]) & np.isfinite(ne_raw[:, k]) & (te_raw[:, k] > 0) & (ne_raw[:, k] > 0)
        if good.sum() < 20:                            # no Thomson profile in this slice: filled in time below
            te_ok[k] = False
        else:
            T_e[k], rms["T_e"][k] = smooth_fit(np.sqrt(pn_ts[good]), te_raw[good, k], grid)
            n_e[k], rms["n_e"][k] = smooth_fit(np.sqrt(pn_ts[good]), ne_raw[good, k], grid)
        out_side = good & (R_ts > R_mag[k]) & (rho_ts[k] > 0.3) & (rho_ts[k] < 0.8)
        in_side = good & (R_ts < R_mag[k]) & (rho_ts[k] > 0.3) & (rho_ts[k] < 0.8)
        check["in_out_Te"][k] = np.nanmedian(te_raw[in_side, k]) / np.nanmedian(te_raw[out_side, k]) - 1 if in_side.any() and out_side.any() else np.nan
        pn_cx = np.clip(fs["pn"](R_cx), 0, None)
        rho_cx[k] = np.where(pn_cx <= PSI_LAST, np.interp(pn_cx, PN, fs["rho"]), np.nan)
        good = (np.sqrt(pn_cx) <= X_MAX) & np.isfinite(ti_raw[:, k]) & (ti_raw[:, k] > -0.3)
        core = good & (rho_cx[k] < 0.4)
        if core.sum() >= 3:
            good &= ti_raw[:, k] <= 1.5 * np.median(ti_raw[core, k])
        if good.sum() >= 30 and (rho_cx[k][good] < 0.4).sum() >= 3 and (rho_cx[k][good] > 0.6).sum() >= 3:
            T_i[k], rms["T_i"][k] = smooth_fit(np.sqrt(pn_cx[good]), ti_raw[good, k], grid)
            ti_ok[k] = True
    if not te_ok.all():
        for j in range(len(RHO)):
            T_e[:, j], n_e[:, j] = np.interp(t, t[te_ok], T_e[te_ok, j]), np.interp(t, t[te_ok], n_e[te_ok, j])
    floor_T, floor_n = 0.02, 1e18
    T_e, n_e = np.maximum(T_e, floor_T), np.maximum(n_e, floor_n)
    band = (RHO > 0.1) & (RHO < 0.9)
    ti_mean = np.where(ti_ok, np.nan_to_num(T_i[:, band]).mean(axis=1), np.nan)
    ti_near = np.array([np.nanmedian(ti_mean[max(0, i - 4):i + 5]) if np.isfinite(ti_mean[max(0, i - 4):i + 5]).any() else np.nan for i in range(nt)])
    ti_ok &= np.abs(np.nan_to_num(ti_mean / ti_near - 1, nan=9.0)) <= 0.25
    use_ti = ti_ok.mean() >= 0.9
    if use_ti and not ti_ok.all():
        for j in range(len(RHO)):
            T_i[:, j] = np.interp(t, t[ti_ok], T_i[ti_ok, j])
    if use_ti:
        T_i = np.maximum(T_i, floor_T)
    tau_E_dia, P_loss = tau_confinement(t, W_dia, P, V_loop, Ip)
    n_i_over_e = (6.0 - FIXED["Z_eff"]) / 5.0
    W_th = np.array([1.5 * 1e3 * QE * np.trapezoid(n_e[k] * T_e[k] + n_i_over_e * n_e[k] * (T_i[k] if use_ti else T_e[k]), V_rho[k]) for k in range(nt)])
    R0, a = 0.5 * (np.nanmax(lcfs_r, axis=0) + np.nanmin(lcfs_r, axis=0)), 0.5 * (np.nanmax(lcfs_r, axis=0) - np.nanmin(lcfs_r, axis=0))
    B0 = bvac * R_mag / R0
    tau_E = tau_confinement(t, W_th, P, V_loop, Ip)[0]
    kappa_a = float(np.median(V_rho[:, -1] / (2 * np.pi ** 2 * R0 * a ** 2)))
    tau98 = float(0.0562 * (np.median(Ip) / 1e6) ** 0.93 * np.median(B0) ** 0.15 * (P_loss / 1e6) ** -0.69 * (np.median(nel) / 1e19) ** 0.41 * 2 ** 0.19
                  * np.median(R0) ** 1.97 * np.median(a / R0) ** 0.58 * kappa_a ** 0.78)
    i91, i50 = int(np.argmin(np.abs(RHO - FIXED["rho_ped"]))), int(np.argmin(np.abs(RHO - 0.5)))
    channels = (("T_e", T_e), ("n_e", n_e)) + ((("T_i", T_i),) if use_ti else ())
    jitter = {c: float(np.median(np.sqrt(np.mean((v[1:][:, band] - v[:-1][:, band]) ** 2, axis=1)) / np.mean(v[:, band]))) / np.sqrt(2) for c, v in channels}
    med = lambda v: float(np.median(v))
    info = dict(shot=shot, campaign=entry["campaign"], shape=entry["shape"], heating=entry["heating"], t0=float(t[0]), t1=float(t[-1]), n_slices=nt,
                tau_E_s=tau_E, tau_E_wdia_s=tau_E_dia, tau98_s=tau98, H98_thermal=tau_E / tau98, H98_wdia=tau_E_dia / tau98, P_loss_MW=P_loss / 1e6,
                Ip_kA=med(Ip) / 1e3, P_nbi_MW=med(P) / 1e6, P_ohm_MW=med(np.abs(V_loop) * Ip) / 1e6, nel_1e19=med(nel) / 1e19, W_dia_kJ=med(W_dia) / 1e3, W_th_kJ=med(W_th) / 1e3,
                R0=med(R0), a=med(a), A=med(R0 / a), B0=med(B0), kappa=med(get("elongation")), delta_u=med(get("triangularity_upper")), delta_l=med(get("triangularity_lower")), q95=med(q95), q0=med(q0),
                T_i_used=bool(use_ti), T_i_slices_ok=float(ti_ok.mean()), elm_spikes=elm_count(root, t0, t1), Te91_over_Te50=med(T_e[:, i91] / T_e[:, i50]), ne91_over_ne50=med(n_e[:, i91] / n_e[:, i50]),
                Te_core_keV=med(T_e[:, 0]), Ti_core_keV=med(T_i[:, 0]) if use_ti else None, fit_jitter_per_slice=jitter,
                fit_residual=dict(T_e=med(rms["T_e"]), n_e=med(rms["n_e"]), T_i=float(np.nanmedian(rms["T_i"])) if use_ti else None),
                check=dict(q95_rel=med(check["q95"]), F_axis_rel=med(check["F_axis"]), Ip_at_last_surface_rel=med(check["Ip"]), inboard_over_outboard_Te=float(np.nanmedian(check["in_out_Te"]))))
    arrays = dict(t=t, rho=RHO, T_e=T_e, n_e=n_e, psi=psi_rho, Ip=Ip, P_heat=P / 1e6, R0=R0, a=a, B0=B0, R_mag=R_mag, Z_mag=Z_mag, bvac_rmag=bvac, q95=q95, q0=q0, W_dia=W_dia, W_th=W_th,
                  V_loop=V_loop, nel=nel, kappa=get("elongation"), delta_u=get("triangularity_upper"), delta_l=get("triangularity_lower"), lcfs_r=lcfs_r, lcfs_z=lcfs_z, efit_R=R, efit_z=Z, efit_psi=psi,
                  pn=PN, q_psi=q_psi, F_psi=F_psi, rho_psi=rho_psi, V_rho=V_rho, psi_ax=check["psi_ax"], psi_b=check["psi_b"], R_ts=R_ts, ts_rho=rho_ts, ts_te=te_raw.T, ts_ne=ne_raw.T,
                  R_cx=R_cx, cx_rho=rho_cx, cx_ti=ti_raw.T, window=np.array([t0, t1]), tau_E=tau_E, info=json.dumps(info))
    if use_ti:
        arrays["T_i"] = T_i
    os.makedirs(os.path.join(out, "shots"), exist_ok=True)
    np.savez_compressed(os.path.join(out, "shots", f"{shot}.npz"), **arrays)
    print(json.dumps(info), flush=True)
    return info


def eqdsk_object(g, k):
    """COCOS 11 EQDSK of slice k, as a dict for TORAX's reader: psi in Wb growing outward, Ip > 0, B0 > 0."""
    from scipy.interpolate import PchipInterpolator, RectBivariateSpline
    R, Z, psi = g["efit_R"], g["efit_z"], g["efit_psi"][:, :, k]
    spl = RectBivariateSpline(Z, R, psi, kx=3, ky=3)
    Rf, Zf = np.linspace(R[0], R[-1], 257), np.linspace(Z[0], Z[-1], 513)
    sgn = np.sign(g["psi_b"][k] - g["psi_ax"][k])
    wb = sgn * 2 * np.pi * spl(Zf, Rf)
    ax, bd = sgn * 2 * np.pi * g["psi_ax"][k], sgn * 2 * np.pi * g["psi_b"][k]
    pn = np.linspace(0.0, 1.0, len(Rf))
    u = lambda p: -np.log(1.0 - p + 1e-3)
    qs, pns = g["q_psi"][k], g["pn"]
    q_last = qs[-1] + (qs[-1] - qs[-2]) / (u(pns[-1]) - u(pns[-2])) * (u(1.0) - u(pns[-1]))
    q = PchipInterpolator(u(np.r_[pns, 1.0]), np.r_[qs, q_last])(u(pn))
    F_edge = g["bvac_rmag"][k] * g["R_mag"][k]
    F = np.interp(pn, np.r_[g["pn"], 1.0], np.r_[g["F_psi"][k], F_edge])
    ok = np.isfinite(g["lcfs_r"][:, k]) & np.isfinite(g["lcfs_z"][:, k])
    lr, lz = g["lcfs_r"][ok, k].astype(float), g["lcfs_z"][ok, k].astype(float)
    if lr[0] != lr[-1] or lz[0] != lz[-1]:
        lr, lz = np.r_[lr, lr[0]], np.r_[lz, lz[0]]
    n = len(Rf)
    wall_r, wall_z = np.array([R[0], R[-1], R[-1], R[0], R[0]]), np.array([Z[0], Z[0], Z[-1], Z[-1], Z[0]])
    none = np.zeros(0)
    return dict(bcentre=float(g["bvac_rmag"][k]), cplasma=float(g["Ip"][k]), dxc=none, dzc=none, ffprime=np.zeros(n), fpol=F, Ic=none, name=f"EFIT {g['t'][k]:.3f}", nbdry=len(lr), ncoil=0, nlim=5, nx=n, nz=len(Zf),
                pprime=-np.ones(n), pressure=np.zeros(n), psi=wb.T, psibdry=float(bd), psimag=float(ax), xbdry=lr, xc=none, xcentre=float(g["R_mag"][k]), xdim=float(R[-1] - R[0]), xgrid1=float(R[0]),
                xlim=wall_r, xmag=float(g["R_mag"][k]), zbdry=lz, zc=none, zdim=float(Z[-1] - Z[0]), zlim=wall_z, zmag=float(g["Z_mag"][k]), zmid=float(0.5 * (Z[0] + Z[-1])), qpsi=q)


def transport_settings(model):
    if model == "qlknn_clip":
        return transport_settings("qlknn") | dict(clip_inputs=True)
    if model in ("bgb_dev", "constant_dev"):
        best = json.load(open(os.path.join(OUT, "calibration.json")))[model]["best"]
        return transport_settings(model[:-4]) | ({k: best for k in BGB_TERMS} if model == "bgb_dev" else dict(chi_i=best, chi_e=best))
    if model == "qlknn":
        import importlib.util
        spec = importlib.util.spec_from_file_location("ex", torax_example("iterhybrid_predictor_corrector.py"))
        ex = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ex)
        return dict(ex.CONFIG["transport"])
    return dict(TRANSPORT[model])


def base_config(g, k, model, geometry="efit", solver=None, dt=None, n_rho=None, transport_extra=None, nbi_width=None):
    """Everything the shot's profiles do not give: the settings in the docstring at the top. geometry 'efit' is the slice at the cut; 'circular' is TORAX's analytic shape with the same R0, a, B0,
    elongation."""
    t = g["t"]
    n_rho = n_rho or FIXED["n_rho"]
    P_trace = {float(tt): float(p * 1e6) for tt, p in zip(t, g["P_heat"])}
    S_trace = {float(tt): float(FIXED["f_abs"] * p * 1e6 / (FIXED["E_beam_eV"] * QE)) for tt, p in zip(t, g["P_heat"])}
    if geometry == "efit":
        geo = dict(geometry_type="eqdsk", cocos=11, eqdsk_object=eqdsk_object(g, k), n_rho=n_rho, Ip_from_parameters=True, last_surface_factor=PSI_LAST, n_surfaces=N_SURF)
    else:
        geo = dict(geometry_type="circular", R_major=float(g["R0"][k]), a_minor=float(g["a"][k]), B_0=float(g["B0"][k]), elongation_LCFS=float(g["kappa"][k]), n_rho=n_rho)
    return dict(
        plasma_composition=dict(main_ion="D", impurity="C", Z_eff=FIXED["Z_eff"]),
        profile_conditions=dict() if geometry == "efit" else dict(current_profile_nu=1.0, initial_j_is_total_current=True),
        numerics=dict(fixed_dt=dt or FIXED["dt"], evolve_ion_heat=True, evolve_electron_heat=True, evolve_current=True, evolve_density=True, resistivity_multiplier=1.0,
                      dW_dt_smoothing_time_scale=0.01),
        geometry=geo,
        neoclassical=dict(bootstrap_current={}, transport=dict(model_name="angioni_sauter")),
        sources=dict(generic_heat=dict(P_total=P_trace, absorption_fraction=FIXED["f_abs"], gaussian_location=FIXED["nbi_center"], gaussian_width=nbi_width or FIXED["nbi_width"],
                                       electron_heat_fraction=FIXED["f_elec"]),
                     generic_particle=dict(S_total=S_trace, deposition_location=FIXED["nbi_center"], particle_width=nbi_width or FIXED["nbi_width"]), ei_exchange={}, ohmic={}),
        pedestal=dict(model_name="set_T_ped_n_ped", set_pedestal=True, rho_norm_ped_top=FIXED["rho_ped"]),
        transport=transport_settings(model) | (transport_extra or {}),
        solver=dict(solver or LINEAR_SOLVER),
        time_step_calculator=dict(calculator_type="fixed"))


def scored_shot(shot):
    """What a run is scored against: the per-slice fits, each averaged with its neighbours in time (5 slices, 20 ms span). Consecutive fits scatter by several percent, and one first crossing of 10%
    ends a trust window."""
    import dataclasses
    n = FIXED["truth_slices"] // 2
    def mean(v):
        return None if v is None else np.array([v[max(0, i - n):i + n + 1].mean(axis=0) for i in range(len(v))])
    return dataclasses.replace(shot, T_e=mean(shot.T_e), n_e=mean(shot.n_e), T_i=mean(shot.T_i))


START_LINE = False
EDGE_CAP = False        # with EDGE_COVER: a slice's fitted temperature may not rise beyond its outermost Thomson point, and its edge density not fall under 0.2 of its mean (H86, known_shot)
EDGE_COVER = None       # rho_norm that a slice's Thomson points must reach for its fit to enter the start (H83): where they stop short, the spline runs on unchecked to rho 1 (up to 1 keV at the edge on 4 of the 11 H82 starts). None: every slice, as before


def edge_cover(g):
    """Per slice, the outermost rho_norm with a Thomson T_e point (0 where there is none)."""
    r, T = g["ts_rho"], g["ts_te"]
    out = np.zeros(len(r))
    for i in range(len(r)):
        ok = np.isfinite(r[i]) & np.isfinite(T[i]) & (T[i] > 0)
        out[i] = float(np.max(r[i][ok])) if ok.any() else 0.0
    return out


def known_shot(truth, k, frozen, cover=None):
    """What the twin is given at the cut k: fits averaged over the trailing slices, so the state at the cut carries nothing from after it. frozen also holds them at that value afterwards.
    cover (edge_cover, with EDGE_COVER set): only the trailing slices whose Thomson points reach EDGE_COVER enter; all of them if none does."""
    import dataclasses
    n = FIXED["trailing_slices"]
    def pick(i):
        idx = np.arange(max(0, i - n + 1), i + 1)
        if cover is not None and EDGE_COVER is not None:
            good = idx[cover[idx] >= EDGE_COVER]
            if len(good):
                return good
        return idx
    def capped(v, name):
        """EDGE_CAP (H86): beyond a slice's outermost Thomson point the fitted temperature may not rise above its value there. On 21428 the one slice whose points
        reach rho 0.9 (157 ms) ends at 95 eV at rho 0.91 and its fit climbs to 357 eV at the separatrix, which the run then held at its edge for the whole shot."""
        if not EDGE_CAP:
            return v
        if name == "n_e":                  # and no edge density under 0.2 of the profile's mean: the fits' floor (1e18 m^-3) left 24627's start at 0.11 of it, under an H-mode pedestal whose
            out = np.array(v, float, copy=True)          # beam-heated edge cells then ran to 600 eV and stopped TORAX; MAST's measured separatrix is 0.25 to 0.30 of the line average in L-mode (docs/lit/43)
            floor = 0.2 * np.nanmean(out, axis=1, keepdims=True)
            return np.maximum(out, floor)
        if cover is None or name != "T_e":          # Thomson's cover; T_i comes from charge exchange with its own
            return v
        out = np.array(v, float, copy=True)
        rho = np.asarray(truth.rho, float)
        for i in range(len(out)):
            c = float(cover[i]) if i < len(cover) else 0.0
            if 0.5 < c < 1.0:
                j = int(np.searchsorted(rho, c))
                if 0 < j < len(rho):
                    out[i, j:] = np.minimum(out[i, j:], out[i, j - 1])
        return out

    def trail(v, name=None):
        if v is None:
            return None
        v = capped(v, name)
        w = np.array([v[pick(i)].mean(axis=0) for i in range(len(v))])
        if START_LINE and k >= n - 1:       # the state at the cut from a straight line through the trailing slices, not their mean: the mean lags a plasma that is changing fast
            idx = pick(k)
            if len(idx) >= 3:
                x = truth.t[idx]
                y = v[idx]
                xm = x - x.mean()
                slope = (xm[:, None] * (y - y.mean(axis=0))).sum(axis=0) / (xm ** 2).sum()
                w[k] = np.maximum(y.mean(axis=0) + slope * (truth.t[k] - x.mean()), 0.5 * y.mean(axis=0))
        if frozen:
            w[k:] = w[k]
        return w
    return dataclasses.replace(truth, T_e=trail(truth.T_e, "T_e"), n_e=trail(truth.n_e, "n_e"), T_i=trail(truth.T_i, "T_i"))


def same_knots(cfg, t_cut, t_end, n=64):
    """Every time trace on the same n knots (about 4 ms apart at most, the data's own 5 ms) and the starting profiles at time 0, as constant extrapolation makes the time arbitrary: JAX compiles the step for the
    length of these arrays and for the times that key the profiles, so otherwise each cut compiles anew, 10 s or more."""
    knots = np.linspace(t_cut, t_end, n)
    pc, ped, src = cfg["profile_conditions"], cfg["pedestal"], cfg["sources"]
    for name in ("T_e", "T_i", "n_e", "psi"):
        if isinstance(pc.get(name), tuple):
            pc[name] = (np.array([0.0]), *pc[name][1:])
    traces = ((pc, "Ip"), (pc, "T_e_right_bc"), (pc, "T_i_right_bc"), (pc, "n_e_right_bc"), (ped, "T_e_ped"), (ped, "T_i_ped"), (ped, "n_e_ped"), (src["generic_heat"], "P_total"), (src["generic_particle"], "S_total"))
    for holder, key in traces:
        times = np.array(sorted(holder[key]))
        holder[key] = (knots, np.interp(knots, times, [holder[key][t] for t in times]))
    return cfg


def build_config(truth, g, k, model, frozen=False, **kw):
    import shot as shot_lib
    t_cut, t_end = float(truth.t[k]), float(truth.t[-1])
    known = known_shot(truth, k, frozen, cover=edge_cover(g) if EDGE_COVER is not None else None)
    cfg = shot_lib.continue_from(known, t_cut, t_end, base_config(g, k, model, **kw))
    later = truth.t[truth.t >= t_cut]
    at_ped = lambda name: {float(tt): float(np.interp(FIXED["rho_ped"], known.rho, known.at(name, tt))) for tt in later}
    cfg["pedestal"].update(T_e_ped=at_ped("T_e"), T_i_ped=at_ped("T_i" if known.T_i is not None else "T_e"), n_e_ped=at_ped("n_e"))
    if kw.get("geometry") == "circular":
        cfg["profile_conditions"].update(psi=None, initial_psi_mode="j")
    return same_knots(cfg, t_cut, t_end)


def persistence_tree(known, t_cut, times, rho):
    """The profile at the cut, held: scored by shot.score exactly as a TORAX run is."""
    import xarray as xr
    data = {c: (("time", "rho_norm"), np.array([np.interp(rho, known.rho, known.at(c, t_cut))] * len(times))) for c in ("T_e", "T_i", "n_e") if getattr(known, c) is not None}
    return {"profiles": xr.DataTree(xr.Dataset(data, coords=dict(time=times, rho_norm=rho)))}


def _trust(sc, t_cut, tol, channels):
    import shot as shot_lib
    return shot_lib.trust_window({c: sc[c] for c in channels if c in sc} | dict(t=sc["t"]), t_cut, tol)


def at_leads(sc, t_cut, leads):
    """Error per channel at the requested lead times; None past the end of the run."""
    lead = sc["t"] - t_cut
    return {c: [float(np.interp(L, lead, sc[c])) if L <= lead[-1] + 1e-9 else None for L in leads] for c in ("T_e", "T_i", "n_e") if c in sc}


def summarize(sc, t_cut, tau_E, tol=FIXED["tol"]):
    channels = [c for c in ("T_e", "T_i", "n_e") if c in sc]
    settle, horizon = _trust(sc, t_cut, tol, channels)
    res = dict(lead_end_s=float(sc["t"][-1] - t_cut), settle_s=settle, horizon_s=horizon, err_at_s=at_leads(sc, t_cut, FIXED["leads_s"]),
               err_at_tau=at_leads(sc, t_cut, [x * tau_E for x in FIXED["leads_tau"]]), mean_err={c: float(np.mean(sc[c])) for c in channels})
    for c in channels:
        s, h = _trust(sc, t_cut, tol, [c])
        res[f"settle_{c}_s"], res[f"horizon_{c}_s"] = s, h
    res["inside_whole_run"] = bool(horizon is not None and horizon >= res["lead_end_s"] - 1e-9 and settle == 0.0)
    return res


def signed_bias(shot, tree, t_cut, leads):
    """Signed mean error over the scored band at the given lead times, relative to the mean of the data: is the model too high or too low?"""
    p = tree["profiles"].to_dataset()
    t, rho = p["time"].values, p["rho_norm"].values
    band = (rho > 0.1) & (rho < 0.9)
    out = {}
    for c in ("T_e", "T_i", "n_e"):
        if getattr(shot, c) is None:
            continue
        row = []
        for L in leads:
            tt = t_cut + L
            if tt > t[-1] + 1e-9:
                row.append(None)
                continue
            i = int(np.argmin(np.abs(t - tt)))
            true = np.interp(rho[band], shot.rho, shot.at(c, t[i]))
            row.append(float(np.mean(p[c].values[i][band] - true) / np.mean(true)))
        out[c] = row
    return out


def run_cut(truth, g, k, model, tau_E, frozen=False, **kw):
    import ntm
    import shot as shot_lib
    import torax
    t_cut = float(truth.t[k])
    cfg = build_config(truth, g, k, model, frozen=frozen, **kw)
    t0 = time.perf_counter()
    tree, recs = ntm.run(torax.ToraxConfig.from_dict(cfg), [])
    wall = time.perf_counter() - t0
    p, s = tree["profiles"].to_dataset(), tree["scalars"].to_dataset()
    sc = shot_lib.score(scored_shot(truth), tree, t_cut)
    res = summarize(sc, t_cut, tau_E)
    res["raw"] = summarize(shot_lib.score(truth, tree, t_cut), t_cut, tau_E)
    res["bias_at_s"] = signed_bias(scored_shot(truth), tree, t_cut, FIXED["leads_s"])
    t_sim = p["time"].values
    end_ok = bool(abs(t_sim[-1] - truth.t[-1]) < 1e-6)
    fs_rho = g["rho_psi"][k]
    q_sim = p["q"].values[0]
    rho_face = p["rho_face_norm"].values if "rho_face_norm" in p else np.linspace(0, 1, len(q_sim))
    q_data = np.interp(rho_face, fs_rho, g["q_psi"][k])
    inner = rho_face < 0.9
    idx = np.searchsorted(t_sim, t_cut + np.array([0.0, 0.05, 0.1]) - 1e-9).clip(0, len(t_sim) - 1)
    numerics = tree["numerics"].to_dataset()
    res.update(cut=t_cut, wall_s=round(wall, 1), steps=len(t_sim) - 1, reached_end=end_ok, sim_error=int(numerics["sim_error"].values.max()) if "sim_error" in numerics else None,
               q_init_rms_rel=float(np.sqrt(np.mean((q_sim[inner] / q_data[inner] - 1) ** 2))), q0_sim=float(q_sim[0]), q_data0=float(q_data[0]),
               W_th_sim_kJ=[float(s["W_thermal_total"].values[i] / 1e3) for i in idx], P_ohmic_sim_MW=[float(s["P_ohmic_e"].values[i] / 1e6) for i in idx], P_aux_sim_MW=float(s["P_aux_total"].values[0] / 1e6))
    arrays = dict(t=t_sim, rho=p["rho_norm"].values, T_e=p["T_e"].values, T_i=p["T_i"].values, n_e=p["n_e"].values, q=p["q"].values, rho_face=rho_face, sc_t=sc["t"],
                  **{f"sc_{c}": sc[c] for c in ("T_e", "T_i", "n_e") if c in sc})
    for name in ("chi_turb_e", "chi_turb_i", "D_turb_e"):
        if name in p:
            arrays[name] = p[name].values
    return res, arrays


def cut_indices(t):
    last = t[-1] - FIXED["min_lead_s"]
    step = int(round(FIXED["cut_step_s"] / np.median(np.diff(t))))
    return [k for k in range(FIXED["trailing_slices"] - 1, len(t), step) if t[k] <= last + 1e-9]


def load_shot(path):
    import shot as shot_lib
    g = dict(np.load(path, allow_pickle=False))
    return shot_lib.Shot.load(path), g, json.loads(str(g["info"]))


def backtest(shots, models, out, frozen=False, tag="", cuts=None, **kw):
    import shot as shot_lib
    os.makedirs(os.path.join(out, "results"), exist_ok=True)
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    for shot_no in shots:
        truth, g, info = load_shot(os.path.join(out, "shots", f"{shot_no}.npz"))
        tau_E = info["tau_E_s"]
        ks = cuts if cuts is not None else cut_indices(truth.t)
        persist = {}
        for k in ks:
            t_cut = float(truth.t[k])
            sim_t = np.arange(t_cut, truth.t[-1] + 1e-9, FIXED["dt"])
            tree = persistence_tree(known_shot(truth, k, frozen), t_cut, sim_t, truth.rho)
            persist[f"{t_cut:.3f}"] = summarize(shot_lib.score(scored_shot(truth), tree, t_cut), t_cut, tau_E) | dict(cut=t_cut, raw=summarize(shot_lib.score(truth, tree, t_cut), t_cut, tau_E))
        if not tag:
            json.dump(dict(shot=shot_no, model="persistence", tag="", tau_E_s=tau_E, runs=persist), open(os.path.join(out, "results", f"{shot_no}_persistence.json"), "w"), indent=1)
        for model in models:
            rows = {}
            for k in ks:
                t_cut = float(truth.t[k])
                try:
                    res, arrays = run_cut(truth, g, k, model, tau_E, frozen=frozen, **kw)
                except Exception as e:
                    res, arrays = dict(cut=t_cut, failed=repr(e)[:500]), None
                rows[f"{t_cut:.3f}"] = res
                if arrays is not None:
                    every = max(1, int(round(0.005 / FIXED["dt"])))
                    timed = ("t", "T_e", "T_i", "n_e", "q", "chi_turb_e", "chi_turb_i", "D_turb_e", "sc_t", "sc_T_e", "sc_T_i", "sc_n_e")
                    keep = {n: (v[::every] if n in timed else v) for n, v in arrays.items()}
                    np.savez_compressed(os.path.join(out, "runs", f"{shot_no}_{model}_{int(round(t_cut * 1e3))}{suffix}.npz"), **keep)
                shown = ("settle_s", "horizon_s", "lead_end_s", "wall_s", "steps", "reached_end", "q_init_rms_rel", "failed", "mean_err")
                print(shot_no, model, f"cut {t_cut:.3f}", json.dumps({n: (round(v, 4) if isinstance(v, float) else v) for n, v in res.items() if n in shown}), flush=True)
            extra = dict(frozen=frozen, **{n: str(v) for n, v in kw.items()})
            json.dump(dict(shot=shot_no, model=model, tag=tag, tau_E_s=tau_E, runs=rows, fixed=FIXED, extra=extra), open(os.path.join(out, "results", f"{shot_no}_{model}{suffix}.json"), "w"), indent=1)


def calibrate(out):
    """One global number per empirical model, from the two development shots only: the value with the smallest error, averaged over cuts and channels and over the whole of each run."""
    scan = dict(bgb_dev=[1, 2, 3, 4, 6, 8, 12, 16], constant_dev=[1, 2, 3, 4, 6, 8, 12, 16])
    res = {}
    for model, values in scan.items():
        table = {}
        for v in values:
            errs = []
            for d in DEV:
                truth, g, info = load_shot(os.path.join(out, "shots", f"{d['shot']}.npz"))
                for k in cut_indices(truth.t)[::2]:
                    extra = {t: float(v) for t in BGB_TERMS} if model == "bgb_dev" else dict(chi_i=float(v), chi_e=float(v))
                    r, _ = run_cut(truth, g, k, model[:-4], info["tau_E_s"], transport_extra=extra)
                    errs.append(np.mean(list(r["mean_err"].values())))
            table[str(v)] = float(np.mean(errs))
            print(model, v, round(table[str(v)], 4), flush=True)
        res[model] = dict(best=float(min(table, key=table.get)), mean_error=table, shots=[d["shot"] for d in DEV])
    json.dump(res, open(os.path.join(out, "calibration.json"), "w"), indent=1)
    print(json.dumps(res))


def qlknn_range(shots, out):
    """Where the measured profiles at the cuts sit against QLKNN_7_11's input ranges, with TORAX's own rescaling of the radius (its x is multiplied by the machine's inverse aspect ratio over 1/3)."""
    import dataclasses
    import torax
    from torax._src.orchestration import run_simulation as run_simulation_lib
    from torax._src.transport_model import qlknn_transport_model as qtm
    res = {}
    for shot_no in shots:
        truth, g, info = load_shot(os.path.join(out, "shots", f"{shot_no}.npz"))
        rows = []
        for k in cut_indices(truth.t):
            state, _, step_fn = run_simulation_lib.prepare_simulation(torax.ToraxConfig.from_dict(build_config(truth, g, k, "qlknn")))
            tm = step_fn.solver.models.transport_model
            rp = step_fn.runtime_params_provider(state.t)
            qi = tm._prepare_qualikiz_inputs(transport=rp.transport, geo=state.geometry, core_profiles=state.core_profiles,
                                             poloidal_velocity_multiplier=rp.neoclassical.poloidal_velocity_multiplier)
            eps_lcfs = float(np.asarray(qi.epsilon)[-1])
            qi = dataclasses.replace(qi, x=qi.x * qi.epsilon[-1] / qtm._EPSILON_NN)
            model = qtm.get_model(tm.path, tm.name)
            X = np.asarray(model.get_model_inputs_from_qualikiz_inputs(qi))
            rho_face = np.asarray(state.geometry.rho_face_norm)
            band = (rho_face > 0.1) & (rho_face < 0.9)
            names = list(model.inputs_and_ranges)
            row = dict(eps_lcfs=eps_lcfs, inputs={})
            for j, name in enumerate(names):
                rng, v = model.inputs_and_ranges[name], X[band, j]
                outside = (v < rng.get("min", -np.inf)) | (v > rng.get("max", np.inf))
                row["inputs"][name] = dict(min=rng.get("min"), max=rng.get("max"), frac_outside=float(np.mean(outside)), median=float(np.median(v)), lo=float(v.min()), hi=float(v.max()))
            x = X[band, names.index("x")]
            over = x > model.inputs_and_ranges["x"].get("max", np.inf)
            row["rho_where_x_leaves"] = float(rho_face[band][np.argmax(over)]) if over.any() else None
            rows.append(row)
        left = [r["rho_where_x_leaves"] for r in rows if r["rho_where_x_leaves"] is not None]
        res[str(shot_no)] = dict(eps_lcfs=float(np.median([r["eps_lcfs"] for r in rows])), rho_where_x_leaves=float(np.median(left)) if left else None,
                                 inputs={n: dict(min=rows[0]["inputs"][n]["min"], max=rows[0]["inputs"][n]["max"], frac_outside=float(np.mean([r["inputs"][n]["frac_outside"] for r in rows])),
                                                 median=float(np.median([r["inputs"][n]["median"] for r in rows])), lo=float(min(r["inputs"][n]["lo"] for r in rows)),
                                                 hi=float(max(r["inputs"][n]["hi"] for r in rows))) for n in names})
        print(shot_no, json.dumps(res[str(shot_no)]), flush=True)
    json.dump(res, open(os.path.join(out, "qlknn_range.json"), "w"), indent=1)


def closure(out):
    """The machinery on a MAST plasma: a TORAX run from a development shot is the truth; cut it 40 ms in and run on with the same physics. The second run must follow the first, as shot.py's
    closure test does on ITER."""
    import ntm
    import shot as shot_lib
    import torax
    truth, g, info = load_shot(os.path.join(out, "shots", f"{DEV[0]['shot']}.npz"))
    k = cut_indices(truth.t)[2]
    res = {}
    for model in ("bgb", "qlknn"):
        tree, _ = ntm.run(torax.ToraxConfig.from_dict(build_config(truth, g, k, model)), [])
        sim = shot_lib.Shot.from_torax(tree, name="closure")
        t2 = float(sim.t[np.searchsorted(sim.t, sim.t[0] + 0.04)])
        cfg = shot_lib.continue_from(sim, t2, float(sim.t[-1]), base_config(g, k, model))
        at_ped = lambda name: {float(tt): float(np.interp(FIXED["rho_ped"], sim.rho, sim.at(name, tt))) for tt in sim.t[sim.t >= t2]}
        cfg["pedestal"].update(T_e_ped=at_ped("T_e"), T_i_ped=at_ped("T_i"), n_e_ped=at_ped("n_e"))
        tree2, _ = ntm.run(torax.ToraxConfig.from_dict(cfg), [])
        sc = shot_lib.score(sim, tree2, t2)
        res[model] = dict(cut=t2, max_error={c: float(np.max(sc[c])) for c in ("T_e", "T_i", "n_e")}, error_at_end={c: float(sc[c][-1]) for c in ("T_e", "T_i", "n_e")},
                          trust=shot_lib.trust_window(sc, t2), steps=len(sc["t"]) - 1)
        print(model, json.dumps(res[model]), flush=True)
    json.dump(res, open(os.path.join(out, "closure.json"), "w"), indent=1)


def load_results(out):
    """{name: {shot: result file}}, name = model, with [tag] for the sensitivity runs."""
    groups = {}
    for path in sorted(glob.glob(os.path.join(out, "results", "*.json"))):
        d = json.load(open(path))
        groups.setdefault(d["model"] + (f"[{d['tag']}]" if d.get("tag") else ""), {})[str(d["shot"])] = d
    return groups


def lead_table(group, field, raw=False):
    """Mean over the cuts of each shot, then mean and spread over shots, per channel and lead: shots count once however many cuts they allow."""
    out = {}
    for c in ("T_e", "T_i", "n_e"):
        per_shot = {}
        for shot, d in group.items():
            src = [(r["raw"] if raw else r) for r in d["runs"].values() if "failed" not in r]
            rows = [r[field][c] for r in src if c in r.get(field, {})]
            if rows:
                per_shot[shot] = np.nanmean(np.array(rows, float), axis=0)
        if per_shot:
            a = np.array(list(per_shot.values()))
            out[c] = dict(mean=np.nanmean(a, axis=0).tolist(), lo=np.nanmin(a, axis=0).tolist(), hi=np.nanmax(a, axis=0).tolist(), n_shots=np.sum(np.isfinite(a), axis=0).tolist(),
                          per_shot={k: v.tolist() for k, v in per_shot.items()})
    return out


def trust_table(group, channel=None, raw=False):
    """The trust window per shot and overall, in ms and in confinement times. A cut that never comes inside counts as 0; to_end counts cuts inside the tolerance from the start to the end of the data."""
    key_s, key_h = (f"settle_{channel}_s", f"horizon_{channel}_s") if channel else ("settle_s", "horizon_s")
    per_shot, all_h, all_tau, n_never, n_end, n_cuts = {}, [], [], 0, 0, 0
    for shot, d in group.items():
        tau = d["tau_E_s"]
        rows = [(r["raw"] if raw else r) for r in d["runs"].values() if "failed" not in r]
        h = [r[key_h] or 0.0 for r in rows]
        settle = [r[key_s] for r in rows if r[key_s] is not None]
        never = sum(r[key_h] is None for r in rows)
        end = sum(r[key_h] is not None and r[key_h] >= r["lead_end_s"] - 1e-9 and r[key_s] == 0.0 for r in rows)
        per_shot[shot] = dict(horizon_ms=float(np.median(h) * 1e3), horizon_tau=float(np.median(h) / tau), settle_ms=float(np.median(settle) * 1e3) if settle else None, cuts=len(rows), never_inside=int(never),
                              inside_to_end=int(end), mean_run_ms=float(np.mean([r["lead_end_s"] for r in rows]) * 1e3))
        all_h += h
        all_tau += [x / tau for x in h]
        n_never += never
        n_end += end
        n_cuts += len(rows)
    return dict(per_shot=per_shot, median_horizon_ms=float(np.median(all_h) * 1e3), median_horizon_tau=float(np.median(all_tau)), mean_of_shot_medians_ms=float(np.mean([v["horizon_ms"] for v in per_shot.values()])),
                cuts=n_cuts, never_inside=int(n_never), inside_to_end=int(n_end))


def fraction_inside(out, group):
    """Share of the held-back time inside the tolerance, per channel, from the saved scores (5 ms apart). The trust window ends at the first crossing; this shows whether a run that left came back."""
    import shot as shot_lib
    res = {c: {} for c in ("all", "T_e", "T_i", "n_e")}
    for shot_no, d in group.items():
        truth = None
        rows = {c: [] for c in res}
        for r in d["runs"].values():
            if "failed" in r:
                continue
            t_cut = r["cut"]
            if d["model"] == "persistence":
                truth = truth or load_shot(os.path.join(out, "shots", f"{shot_no}.npz"))[0]
                k = int(np.argmin(np.abs(truth.t - t_cut)))
                tree = persistence_tree(known_shot(truth, k, False), t_cut, np.arange(t_cut, truth.t[-1] + 1e-9, 0.005), truth.rho)
                sc = shot_lib.score(scored_shot(truth), tree, t_cut)
            else:
                suffix = f"_{d['tag']}" if d.get("tag") else ""
                run = np.load(os.path.join(out, "runs", f"{shot_no}_{d['model']}_{int(round(t_cut * 1e3))}{suffix}.npz"))
                sc = {c: run[f"sc_{c}"] for c in ("T_e", "T_i", "n_e") if f"sc_{c}" in run.files}
            bad = np.zeros(len(next(iter(sc[c] for c in sc if c != "t"))), bool)
            for c in ("T_e", "T_i", "n_e"):
                if c in sc:
                    rows[c].append(float(np.mean(sc[c] <= FIXED["tol"])))
                    bad |= sc[c] > FIXED["tol"]
            rows["all"].append(float(np.mean(~bad)))
        for c in res:
            if rows[c]:
                res[c][shot_no] = float(np.mean(rows[c]))
    return {c: dict(mean_over_shots=float(np.mean(list(v.values()))), per_shot=v) for c, v in res.items() if v}


def report(out):
    groups = load_results(out)
    conv = json.load(open(os.path.join(out, "conversion.json")))
    selected = [e["shot"] for e in json.load(open(os.path.join(out, "selection.json")))["shots"]]
    res = dict(settings=FIXED, shots={str(s): conv[str(s)] for s in selected}, models={})
    for key in ("calibration", "qlknn_range", "closure"):
        path = os.path.join(out, f"{key}.json")
        if os.path.exists(path):
            res[key] = json.load(open(path))
    for name, group in groups.items():
        failures = [dict(shot=sh, cut=r.get("cut"), why=r["failed"]) for sh, d in group.items() for r in d["runs"].values() if "failed" in r]
        short = [dict(shot=sh, cut=r.get("cut"), lead_end_s=r["lead_end_s"], sim_error=r.get("sim_error")) for sh, d in group.items() for r in d["runs"].values()
                 if "failed" not in r and r.get("reached_end") is False]
        runs = [r for d in group.values() for r in d["runs"].values()]
        res["models"][name] = dict(err_at_s=lead_table(group, "err_at_s"), err_at_tau=lead_table(group, "err_at_tau"), raw_err_at_s=lead_table(group, "err_at_s", raw=True),
                                   bias_at_s=lead_table(group, "bias_at_s") if name != "persistence" else None, trust_all=trust_table(group), trust_raw=trust_table(group, raw=True),
                                   trust={c: trust_table(group, c) for c in ("T_e", "T_i", "n_e")}, fraction_inside=fraction_inside(out, group), failures=failures, stopped_early=short, wall_s=float(np.sum([r.get("wall_s", 0.0) for r in runs])),
                                   q_init_rms_rel=float(np.median([r["q_init_rms_rel"] for r in runs if "q_init_rms_rel" in r])) if name != "persistence" else None)
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1)
    leads, taus = FIXED["leads_s"], FIXED["leads_tau"]
    fmt = lambda v: "   -" if v is None or not np.isfinite(v) else f"{100 * v:4.0f}"
    print("relative RMS error (%), mean over shots of the per-shot mean over cuts, against the 5-slice averaged fits. Lead in ms: " + " ".join(f"{1e3 * L:4.0f}" for L in leads))
    for name, m in res["models"].items():
        for c in ("T_e", "T_i", "n_e"):
            if c in m["err_at_s"]:
                print(f"  {name:24s} {c:4s} " + " ".join(fmt(v) for v in m["err_at_s"][c]["mean"]))
    print("the same against lead in confinement times: " + " ".join(f"{x:4.2g}" for x in taus))
    for name, m in res["models"].items():
        for c in ("T_e", "T_i", "n_e"):
            if c in m["err_at_tau"]:
                print(f"  {name:24s} {c:4s} " + " ".join(fmt(v) for v in m["err_at_tau"][c]["mean"]))
    print("signed mean error (%), model minus data, same leads in ms")
    for name, m in res["models"].items():
        if m["bias_at_s"]:
            for c in ("T_e", "T_i", "n_e"):
                if c in m["bias_at_s"]:
                    print(f"  {name:24s} {c:4s} " + " ".join("   -" if v is None or not np.isfinite(v) else f"{100 * v:+4.0f}" for v in m["bias_at_s"][c]["mean"]))
    print("trust window at 10%: median horizon over cuts in ms and confinement times, cuts never inside, cuts inside to the end of the data, all channels together, then per channel")
    for name, m in res["models"].items():
        t = m["trust_all"]
        extra = "  ".join(f"{c} {m['trust'][c]['median_horizon_ms']:.0f} ms ({m['trust'][c]['median_horizon_tau']:.2f}), to end {m['trust'][c]['inside_to_end']}/{m['trust'][c]['cuts']}" for c in ("T_e", "T_i", "n_e"))
        flags = (f"  FAILED {len(m['failures'])}" if m["failures"] else "") + (f"  STOPPED EARLY {len(m['stopped_early'])}" if m["stopped_early"] else "")
        print(f"  {name:24s} {t['median_horizon_ms']:6.1f} ms {t['median_horizon_tau']:5.2f} tau_E  never {t['never_inside']}/{t['cuts']}  to end {t['inside_to_end']}/{t['cuts']} | raw fits {m['trust_raw']['median_horizon_ms']:.1f} ms | {extra}{flags}")
    print("share of the held-back time inside 10%, mean over shots, all channels together, then T_e, T_i, n_e")
    for name, m in res["models"].items():
        print(f"  {name:24s} " + " ".join(f"{m['fraction_inside'][c]['mean_over_shots']:5.2f}" for c in ("all", "T_e", "T_i", "n_e")))
    print("trust window per shot: median horizon in ms (confinement times), all channels")
    names = [n for n in res["models"] if "[" not in n]
    print("  shot   tau_E(ms) " + " ".join(f"{n:>16s}" for n in names))
    for s in selected:
        cells = [res["models"][n]["trust_all"]["per_shot"].get(str(s)) for n in names]
        print(f"  {s} {conv[str(s)]['tau_E_s'] * 1e3:8.1f}  " + " ".join(f"{c['horizon_ms']:7.1f} ({c['horizon_tau']:5.2f})" if c else " " * 16 for c in cells))
    figures(out, res, selected, conv)


INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
STYLE = {"qlknn": ("#2a78d6", "-", 2.0, "QLKNN"), "bgb": ("#eb6834", "-", 2.0, "Bohm-gyroBohm, TORAX default"), "constant": ("#1baf7a", "-", 2.0, "constant chi = 1"),
         "bgb_dev": ("#eb6834", (0, (4, 2)), 1.6, "Bohm-gyroBohm, multiplier from dev shots"), "constant_dev": ("#1baf7a", (0, (4, 2)), 1.6, "constant chi from dev shots"),
         "persistence": (MUTED, "-", 1.6, "profile at the cut, held")}


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)


def figures(out, res, selected, conv):
    """Measured against simulated T_e for the first shot of the selection at its first cut, and error against lead over all shots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "axes.labelcolor": INK2, "text.color": INK})
    shot_no = selected[0]
    truth, g, info = load_shot(os.path.join(out, "shots", f"{shot_no}.npz"))
    scored = scored_shot(truth)
    k = cut_indices(truth.t)[0]
    t_cut, tau = float(truth.t[k]), info["tau_E_s"]
    wanted = [x for x in (0.5, 2.0, 5.0, 10.0) if x * tau <= truth.t[-1] - t_cut]
    known = known_shot(truth, k, False)
    fig, axes = plt.subplots(1, len(wanted), figsize=(3.3 * len(wanted), 3.8), sharey=True, facecolor=SURFACE)
    top = 3.0 * max(scored.at("T_e", t_cut + x * tau).max() for x in wanted)
    for ax, x in zip(np.atleast_1d(axes), wanted):
        tt = t_cut + x * tau
        i = int(np.argmin(np.abs(truth.t - tt)))
        off_scale = []
        ok = np.isfinite(g["ts_rho"][i]) & np.isfinite(g["ts_te"][i])
        ax.plot(g["ts_rho"][i][ok], g["ts_te"][i][ok], ".", color=AXIS, ms=4, label="Thomson points", zorder=1)
        ax.plot(truth.rho, known.at("T_e", t_cut), color=MUTED, lw=1.6, label=STYLE["persistence"][3], zorder=2)
        for m in ("qlknn", "bgb", "constant", "bgb_dev", "constant_dev"):
            path = os.path.join(out, "runs", f"{shot_no}_{m}_{int(round(t_cut * 1e3))}.npz")
            if os.path.exists(path):
                d = np.load(path)
                j = int(np.argmin(np.abs(d["t"] - tt)))
                ax.plot(d["rho"], d["T_e"][j], color=STYLE[m][0], ls=STYLE[m][1], lw=STYLE[m][2], label=STYLE[m][3], zorder=3)
                if d["T_e"][j].max() > top:
                    off_scale.append(f"{STYLE[m][3]}: {d['T_e'][j].max():.1f} keV")
        ax.plot(scored.rho, scored.at("T_e", tt), color=INK, lw=2.0, label="measured, smooth fit", zorder=4)
        ax.set_title(f"{1e3 * x * tau:.0f} ms after the cut ({x:g} tau_E)", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("rho_norm", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, top)
        if off_scale:
            ax.text(0.98, 0.97, "off scale, peak\n" + "\n".join(off_scale), transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK2)
        _style(ax)
    first = np.atleast_1d(axes)[0]
    first.set_ylabel("T_e (keV)", fontsize=9)
    handles, labels = first.get_legend_handles_labels()
    fig.suptitle(f"MAST {shot_no}: T_e run on from a cut at t = {t_cut:.3f} s, tau_E = {1e3 * tau:.0f} ms", fontsize=11, color=INK, x=0.01, ha="left")
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    fig.savefig(os.path.join(out, "te_profiles.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), sharey=True, facecolor=SURFACE)
    taus = np.array(FIXED["leads_tau"])
    for ax, c in zip(axes, ("T_e", "T_i", "n_e")):
        ax.axhline(FIXED["tol"], color=INK2, lw=1.0)
        if c == "T_i":
            ax.text(taus[-1], FIXED["tol"] * 0.93, "10% tolerance", fontsize=8, color=INK2, ha="right", va="top")
        for m in ("persistence", "qlknn", "bgb", "constant", "bgb_dev", "constant_dev"):
            if m in res["models"] and c in res["models"][m]["err_at_tau"]:
                row = res["models"][m]["err_at_tau"][c]
                y = np.where(np.array(row["n_shots"]) >= min(3, len(res["shots"])), np.array(row["mean"], float), np.nan)
                ax.plot(taus, y, color=STYLE[m][0], ls=STYLE[m][1], lw=STYLE[m][2], label=STYLE[m][3], solid_capstyle="round")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(taus)
        ax.set_xticklabels([f"{x:g}" for x in taus])
        ax.minorticks_off()
        ax.set_yticks([0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
        ax.set_yticklabels(["3%", "10%", "30%", "100%", "300%", "1000%"])
        ax.set_xlabel("lead time (confinement times)", fontsize=9)
        ax.set_title(c, fontsize=10, color=INK, loc="left")
        _style(ax)
    axes[0].set_ylabel("relative RMS error, 0.1 < rho < 0.9", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    n = len(res["shots"])
    fig.suptitle(f"Error against lead, {n} MAST shots: mean over shots of the per-shot mean over cuts, leads reached by at least 3 shots", fontsize=11, color=INK, x=0.01, ha="left")
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    fig.savefig(os.path.join(out, "error_vs_lead.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["select", "convert", "calibrate", "closure", "backtest", "qlknn_range", "report"])
    ap.add_argument("--shots", type=int, nargs="*", default=None)
    ap.add_argument("--dev", action="store_true", help="convert: the development shots")
    ap.add_argument("--models", nargs="*", default=["qlknn", "bgb", "constant", "bgb_dev", "constant_dev", "qlknn_clip"])
    ap.add_argument("--frozen", action="store_true", help="backtest: boundary and pedestal held at their value at the cut")
    ap.add_argument("--geometry", default="efit", choices=["efit", "circular"])
    ap.add_argument("--nbi_width", type=float, default=None, help="backtest: sensitivity to the heating profile")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    OUT = args.out
    if args.command == "select":
        select(args.out)
    elif args.command == "convert":
        entries = json.load(open(os.path.join(args.out, "selection.json")))["shots"] + DEV
        chosen = args.shots or [e["shot"] for e in (DEV if args.dev else entries[:-len(DEV)])]
        infos = [convert(next(e for e in entries if e["shot"] == s), args.out) for s in chosen]
        path = os.path.join(args.out, "conversion.json")
        old = json.load(open(path)) if os.path.exists(path) else {}
        json.dump({**old, **{str(i["shot"]): i for i in infos}}, open(path, "w"), indent=1)
    else:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        shots = args.shots or [e["shot"] for e in json.load(open(os.path.join(args.out, "selection.json")))["shots"]]
        if args.command == "calibrate":
            calibrate(args.out)
        elif args.command == "closure":
            closure(args.out)
        elif args.command == "backtest":
            backtest(shots, args.models, args.out, frozen=args.frozen, tag=args.tag, geometry=args.geometry, nbi_width=args.nbi_width)
        elif args.command == "qlknn_range":
            qlknn_range(shots, args.out)
        elif args.command == "report":
            report(args.out)


if __name__ == "__main__":
    main()
