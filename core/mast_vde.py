"""MAST's disruptions in 2D with FreeGSNKE (H78, C43; docs/digital-twin.md, "MAST's current quench in 2D"): the plasma, the vessel and the coils through the
thermal and current quench, from a real shot's state just before it, checked against that shot's own probes, flux loops and current.

Why: DREAM (1D) gets MAST's median current quench only with the energy going the wrong way (87 to 90% radiated against 5 to 10% measured), and Thornton
et al. 2012 found single-null MAST plasmas lose their current at a rate set by losing vertical control, gas or no gas. A 2D free-boundary model with the
vessel's currents is what can say whether the current quench is the plasma moving, and so what a chip's gas after the onset could change.

  extract   WSL, read only: for development disruptions with the thermal quench timed (runs/twin/disruptions/mast.json), EFIT before the collapse,
            the coil traces, and the probes, flux loops, current and soft X-ray cameras around it -> runs/twin/disruptions/vde/shots/<shot>.npz
            Usage: python core/mast_vde.py extract [--n 40] [--shots ...]
Shots held out for testing are never read: the events come from mast.json, and when the workspace keeps runs/twin/dev_shots.json, events off that list are skipped.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = os.path.abspath(os.environ.get("XTORAX_HOME") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the workspace: runs/twin and external/ (README, "Where things live")
OUT = os.path.join(ROOT, "runs", "twin", "disruptions", "vde")
SHOTS = os.path.join(OUT, "shots")
STORE = os.path.expanduser(os.environ.get("XTORAX_MAST_STORE", "~/data/mast/tokamark_v1"))   # FAIR-MAST's TokaMark v1 store (zarr)
EVENTS = os.path.join(ROOT, "runs", "twin", "disruptions", "mast.json")
WINDOW = (-0.030, 0.015)                    # s around the thermal quench for the fast signals
SXR_WINDOW = (-0.005, 0.003)


def development():
    """runs/twin/dev_shots.json's shots if the workspace keeps that list, else None (any shot)."""
    path = os.path.join(ROOT, "runs", "twin", "dev_shots.json")
    return set(json.load(open(path))["development"]) if os.path.exists(path) else None


def separatrices(R, Z, psi, axis, xr, xz):
    """psi_n at each X-point and the gap between the two separatrices at the outboard midplane (dRsep, m; NaN when an X-point is missing).
    Double null when both X-points sit on the boundary."""
    from scipy.interpolate import RectBivariateSpline
    spl = RectBivariateSpline(Z, R, psi, kx=3, ky=3)
    p_ax = float(spl.ev(axis[1], axis[0]))
    out = []
    rr = np.linspace(axis[0], R[-1] - 0.05, 400)
    line = spl.ev(np.full_like(rr, axis[1]), rr)
    for r, z in zip(xr, xz):
        if not (np.isfinite(r) and np.isfinite(z)):
            out.append((np.nan, np.nan))
            continue
        p_x = float(spl.ev(z, r))
        k = np.flatnonzero(np.diff(np.sign(line - p_x)))
        out.append((p_x, float(rr[k[0]]) if k.size else np.nan))
    p_b = [x[0] for x in out]
    r_mid = [x[1] for x in out]
    return p_ax, p_b, (r_mid[0] - r_mid[1]) if len(r_mid) == 2 else np.nan


def extract(args):
    import zarr
    ok = development()
    events = [e for e in json.load(open(EVENTS)) if e.get("t_tq") is not None and e.get("cq_80_20_ms") is not None and (ok is None or int(e["shot"]) in ok)]
    events.sort(key=lambda e: e["shot"])
    if args.shots:
        pick = [e for e in events if e["shot"] in set(args.shots)]
    else:
        step = max(1, len(events) // args.n)
        pick = events[::step][: args.n]
    os.makedirs(SHOTS, exist_ok=True)
    listing = []
    for e in pick:
        shot, t_tq = int(e["shot"]), float(e["t_tq"])
        try:
            root = zarr.open_group(os.path.join(STORE, f"{shot}.zarr"), mode="r")
            eq, pf, mg = root["equilibrium"], root["pf_active"], root["magnetics"]
            t_eq = eq["time"][:].astype(float)
            R, Z = eq["major_radius"][:].astype(float), eq["z"][:].astype(float)
            psi_all, j_all = eq["psi"][:].astype(np.float32), eq["j_tor"][:].astype(np.float32)
            if psi_all.shape[0] != len(Z) or psi_all.shape[1] != len(R):
                psi_all, j_all = np.transpose(psi_all, (1, 0, 2)), np.transpose(j_all, (1, 0, 2))
            lr, lz = eq["lcfs_r"][:].astype(float), eq["lcfs_z"][:].astype(float)
            good = [k for k in range(len(t_eq)) if t_eq[k] <= t_tq - args.before and np.isfinite(psi_all[:, :, k]).all() and np.isfinite(j_all[:, :, k]).all() and np.isfinite(lr[:, k]).sum() > 20]
            if not good:
                print(shot, "no EFIT slice before the collapse", flush=True)
                continue
            k0 = good[-1]
            keep = [k for k in range(max(k0 - 2, 0), min(k0 + 4, len(t_eq)))]
            d = dict(shot=shot, t_tq=t_tq, t_cq80=float(e["t_cq80"]), cq_80_20_ms=float(e["cq_80_20_ms"]), spike_pct=float(e.get("spike_pct") or np.nan), k0=k0, R=R, Z=Z, t_eq=t_eq,
                     keep=np.array(keep), psi=psi_all[:, :, keep], j_keep=j_all[:, :, keep], j_tor=j_all[:, :, : k0 + 1], lcfs_r=lr[:, keep], lcfs_z=lz[:, keep])
            for name in ("magnetic_axis_r", "magnetic_axis_z", "bvac_rmag", "elongation", "q95", "li", "beta_pol", "minor_radius", "rpsi100_in", "rpsi100_out", "triangularity_upper", "triangularity_lower", "whmd"):
                if name in eq:
                    d[name] = eq[name][:].astype(float)
            d["x_point_r"], d["x_point_z"] = eq["x_point_r"][:].astype(float), eq["x_point_z"][:].astype(float)
            tp = pf["time"][:].astype(float)
            names = [str(x) for x in pf["current_channel"][:]]
            cur = pf["coil_current"][:].astype(float)
            mean = lambda key: np.mean([cur[names.index(a)] for a in names if key in a], axis=0)
            d.update(t_pf=tp, P1=pf["solenoid_current"][:].astype(float), P2=mean("P2"), P3=mean("P3"), P4=mean("P4"), P5=mean("P5"), pf_voltage=pf["coil_voltage"][:].astype(float),
                     pf_voltage_channel=np.array([str(x) for x in pf["voltage_channel"][:]]), P2_sections=np.stack([cur[names.index(a)] for a in names if "P2" in a]))
            tm = mg["time"][:].astype(float)
            w = (tm >= t_tq + WINDOW[0]) & (tm <= t_tq + WINDOW[1])
            d["t_mag"], d["ip"] = tm[w], mg["ip"][:].astype(float)[w]
            for fam, key in (("ccbv", "b_field_pol_probe_ccbv_field"), ("obr", "b_field_pol_probe_obr_field"), ("obv", "b_field_pol_probe_obv_field"), ("flux", "flux_loop_flux")):
                x = mg[key][:].astype(float)
                d[fam] = x[:, w] if x.shape[-1] == len(tm) else x[w].T
            if "summary" in root:
                sm = root["summary"]
                ts = sm["time"][:].astype(float)
                w = (ts >= t_tq + WINDOW[0]) & (ts <= t_tq + WINDOW[1])
                d["t_sum"] = ts[w]
                for name in ("power_radiated", "ip", "power_nbi"):
                    if name in sm:
                        d["sum_" + name] = sm[name][:].astype(float)[w]
            if "soft_x_rays" in root:
                sx = root["soft_x_rays"]
                ts = sx["time"][:].astype(float)
                w = (ts >= t_tq + SXR_WINDOW[0]) & (ts <= t_tq + SXR_WINDOW[1])
                d["t_sxr"] = ts[w]
                for name in ("horizontal_cam_upper", "horizontal_cam_lower"):
                    if name in sx:
                        x = sx[name][:].astype(np.float32)
                        d["sxr_" + name.split("_")[-1]] = x[:, w] if x.shape[-1] == len(ts) else x[w].T
            axis = (float(d["magnetic_axis_r"][k0]), float(d["magnetic_axis_z"][k0]))
            p_ax, p_x, drsep = separatrices(R, Z, psi_all[:, :, k0].astype(float), axis, d["x_point_r"][:, k0], d["x_point_z"][:, k0])
            lc = np.c_[lr[:, k0], lz[:, k0]]
            lc = lc[np.isfinite(lc).all(axis=1)]
            from scipy.interpolate import RectBivariateSpline
            p_b = float(np.median(RectBivariateSpline(Z, R, psi_all[:, :, k0].astype(float), kx=3, ky=3).ev(lc[:, 1], lc[:, 0])))
            pn_x = [(p - p_ax) / (p_b - p_ax) if np.isfinite(p) else np.nan for p in p_x]
            d["drsep"], d["pn_x"] = drsep, np.array(pn_x)
            np.savez_compressed(os.path.join(SHOTS, f"{shot}.npz"), **d)
            row = dict(shot=shot, t_tq=t_tq, t_efit=float(t_eq[k0]), before_ms=1e3 * (t_tq - t_eq[k0]), cq_80_20_ms=d["cq_80_20_ms"], spike_pct=d["spike_pct"], Ip_kA=float(abs(np.sum(j_all[:, :, k0]) * (R[1] - R[0]) * (Z[1] - Z[0])) / 1e3),
                       Z_axis=axis[1], R_axis=axis[0], kappa=float(d["elongation"][k0]), q95=float(d["q95"][k0]), li=float(d["li"][k0]) if "li" in d else None, drsep_cm=100 * drsep if np.isfinite(drsep) else None,
                       pn_x=[round(x, 3) if np.isfinite(x) else None for x in pn_x], x_point_z=[round(float(x), 3) if np.isfinite(x) else None for x in d["x_point_z"][:, k0]])
            listing.append(row)
            print(shot, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "shot"}, flush=True)
        except Exception as err:
            print(shot, "failed", repr(err)[:200], flush=True)
    json.dump(listing, open(os.path.join(OUT, "shots.json"), "w"), indent=0)
    print(len(listing), "shots written to", SHOTS, flush=True)


def config(args):
    """WSL, read only: every development disruption's configuration from EFIT just before it (dRsep, the axis's height, shape, li) beside its measured
    current quench: does the shape before say how fast the current goes? -> runs/twin/disruptions/vde/config.json"""
    import zarr
    from scipy.interpolate import RectBivariateSpline
    ok = development()
    events = [e for e in json.load(open(EVENTS)) if e.get("cq_80_20_ms") is not None and (ok is None or int(e["shot"]) in ok)]
    rows = []
    for e in events:
        shot = int(e["shot"])
        t_ref = float(e["t_tq"]) if e.get("t_tq") is not None else float(e["t_cq80"]) - 0.002
        try:
            eq = zarr.open_group(os.path.join(STORE, f"{shot}.zarr"), mode="r")["equilibrium"]
            t_eq = eq["time"][:].astype(float)
            R, Z = eq["major_radius"][:].astype(float), eq["z"][:].astype(float)
            lr, lz = eq["lcfs_r"][:].astype(float), eq["lcfs_z"][:].astype(float)
            ks = [k for k in np.flatnonzero(t_eq <= t_ref - args.before) if np.isfinite(lr[:, k]).sum() > 20]
            if not ks:
                continue
            k0 = int(ks[-1])
            psi = eq["psi"][:, :, k0].astype(float)
            if psi.shape != (len(Z), len(R)):
                psi = psi.T
            if not np.isfinite(psi).all():
                continue
            axis = (float(eq["magnetic_axis_r"][k0]), float(eq["magnetic_axis_z"][k0]))
            xr, xz = eq["x_point_r"][:, k0].astype(float), eq["x_point_z"][:, k0].astype(float)
            p_ax, p_x, drsep = separatrices(R, Z, psi, axis, xr, xz)
            lc = np.c_[lr[:, k0], lz[:, k0]]
            lc = lc[np.isfinite(lc).all(axis=1)]
            p_b = float(np.median(RectBivariateSpline(Z, R, psi, kx=3, ky=3).ev(lc[:, 1], lc[:, 0])))
            pn_x = [(p - p_ax) / (p_b - p_ax) if np.isfinite(p) else np.nan for p in p_x]
            val = lambda name: float(eq[name][k0]) if name in eq else None
            rows.append(dict(shot=shot, cq_80_20_ms=float(e["cq_80_20_ms"]), spike_pct=e.get("spike_pct"), timed=e.get("t_tq") is not None, before_ms=1e3 * (t_ref - t_eq[k0]),
                             drsep_cm=100 * drsep if np.isfinite(drsep) else None, pn_x=[x if np.isfinite(x) else None for x in pn_x], Z_axis=axis[1], R_axis=axis[0],
                             Z_top=float(np.max(lc[:, 1])), Z_bot=float(np.min(lc[:, 1])), kappa=val("elongation"), li=val("li"), q95=val("q95"), beta_pol=val("beta_pol"),
                             Ip_kA=float(e.get("ip_before_kA") or np.nan), delta_u=val("triangularity_upper"), delta_l=val("triangularity_lower")))
        except Exception as err:
            print(shot, "failed", repr(err)[:160], flush=True)
    json.dump(rows, open(os.path.join(OUT, "config.json"), "w"), indent=0)
    print(len(rows), "of", len(events), "disruptions with EFIT before them", flush=True)


def motion(args):
    """WSL, read only: which way each development disruption's current moves once the thermal quench starts, from the centre column's B_Z probes:
    the moment sum(Z B) / sum(|B|) over the array (channel heights from runs/twin/sensors/geometry.json). The coils' currents are the same above and
    below, so their part is even in Z and moves the moment little; a current centroid moving up moves it up. Adds to config.json, per event: the moment
    1 ms before, its extreme within 3 ms after, and when."""
    import zarr
    geo = json.load(open(os.path.join(ROOT, "runs", "twin", "sensors", "geometry.json")))
    zc = np.full(40, np.nan)
    for c, g in geo["ccbv"].items():
        zc[int(c)] = g["Z"]
    events = {int(e["shot"]): e for e in json.load(open(EVENTS))}
    rows = json.load(open(os.path.join(OUT, "config.json")))
    for r in rows:
        e = events[r["shot"]]
        t_ref = float(e["t_tq"]) if e.get("t_tq") is not None else float(e["t_cq80"]) - 0.0015
        try:
            mg = zarr.open_group(os.path.join(STORE, f"{r['shot']}.zarr"), mode="r")["magnetics"]
            t = mg["time"][:].astype(float)
            w = (t >= t_ref - 0.003) & (t <= t_ref + 0.004)
            B = mg["b_field_pol_probe_ccbv_field"][:].astype(float)
            B = B[:, w] if B.shape[-1] == len(t) else B[w].T
            ip = np.abs(mg["ip"][:].astype(float)[w])
            tt = t[w] - t_ref
            ok = np.isfinite(zc) & np.isfinite(B).all(axis=1)
            M = 100 * (zc[ok, None] * B[ok]).sum(0) / np.abs(B[ok]).sum(0)
            before = float(np.median(M[(tt >= -0.0015) & (tt <= -0.0005)]))
            after = (tt >= 0.0) & (tt <= 0.003) & (ip >= 0.3 * float(np.median(ip[tt < -0.0005])))     # while a third of the current is left: a moment of a tiny current means little
            if after.sum() < 2:
                continue
            k = int(np.argmax(np.abs(M[after] - before)))
            r.update(moment_before_cm=before, moment_shift_cm=float(M[after][k] - before), moment_shift_ms=1e3 * float(tt[after][k]), motion_ref="t_tq" if e.get("t_tq") is not None else "t_cq80 - 1.5 ms")
        except Exception as err:
            print(r["shot"], "failed", repr(err)[:160], flush=True)
    json.dump(rows, open(os.path.join(OUT, "config.json"), "w"), indent=0)
    print(sum("moment_shift_cm" in r for r in rows), "of", len(rows), "with a moment", flush=True)


R_BIG = 1.0e4           # ohm: a measured coil driven as a current source, V = R_BIG * I_measured; its own L / R_BIG is microseconds, and what the plasma induces moves it by EMF / R_BIG, amps
COILS = ("P1", "P2", "P3", "P4", "P5")


def machine3():
    """The twin's MAST (mast_machine2: coils with turns fitted from EFIT, MAST-U's vessel description, a box for the limiter) plus P6 from MAST-U's
    description, which kept MAST's P6: the vertical position coil, its two halves wired opposite. The store has no P6 current, so its current at the
    start is found to hold EFIT's boundary, and its supply then holds its voltage."""
    import pickle
    import shutil
    src = os.path.join(ROOT, "runs", "twin", "freeboundary", "mast_machine2")
    folder = os.path.join(ROOT, "runs", "twin", "freeboundary", "mast_machine3")
    paths = dict(active_coils_path=os.path.join(folder, "active.pickle"), passive_coils_path=os.path.join(folder, "passive.pickle"), limiter_path=os.path.join(folder, "limiter.pickle"), wall_path=os.path.join(folder, "wall.pickle"))
    if not os.path.exists(paths["active_coils_path"]):
        os.makedirs(folder, exist_ok=True)
        active = pickle.load(open(os.path.join(src, "active.pickle"), "rb"))
        mu = pickle.load(open(os.path.join(ROOT, "external", "freegsnke", "machine_configs", "MAST-U", "MAST-U_like_active_coils.pickle"), "rb"))
        active["P6"] = mu["P6"]
        pickle.dump(active, open(paths["active_coils_path"], "wb"))
        for name in ("passive", "limiter", "wall"):
            shutil.copy(os.path.join(src, name + ".pickle"), paths[name + "_coils_path" if name == "passive" else name + "_path"])
    return paths


def jeon_fit(R, Z, psi, j, lcfs, axis):
    """FreeGSNKE's ConstrainBetapIp current, j = l1 (R / R_axis) f + l2 (R_axis / R) f with f = (1 - psi_n^alpha_m)^alpha_n (Jeon 2015), fitted to EFIT's
    current density inside the boundary: alpha_m and alpha_n by a grid search, l1 and l2 linearly. Returns alpha_m, alpha_n and the rms left over the rms."""
    from scipy.interpolate import RectBivariateSpline
    spl = RectBivariateSpline(Z, R, psi, kx=3, ky=3)
    ax = float(spl.ev(axis[1], axis[0]))
    bd = float(np.median(spl.ev(lcfs[:, 1], lcfs[:, 0])))
    RR, _ = np.meshgrid(R, Z)
    pn = (psi - ax) / (bd - ax)
    m = (np.abs(j) > 0) & (pn >= 0) & (pn <= 1)
    x, r, y = pn[m], RR[m], np.abs(j[m])
    best = (np.inf, None, None)
    for am in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0):
        for an in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0):
            f = (1 - x ** am) ** an
            A = np.stack([r / axis[0] * f, axis[0] / r * f], axis=1)
            c = np.linalg.lstsq(A, y, rcond=None)[0]
            rms = float(np.sqrt(np.mean((y - A @ c) ** 2)))
            if rms < best[0]:
                best = (rms, am, an)
    return best[1], best[2], best[0] / float(np.sqrt(np.mean(y ** 2)))


def lao(x, pp, ffp):
    """p' and FF' against normalized flux as Lao85 coefficients with p'(1) = FF'(1) = 0, as FreeGSNKE's evolving solver takes them."""
    A = np.stack([x ** k - x ** 4 for k in range(4)], axis=1)
    return np.linalg.lstsq(A, pp, rcond=None)[0], np.linalg.lstsq(A, ffp, rcond=None)[0]


def shot_state(shot):
    """An extracted disruption (extract): EFIT's last slice before the thermal quench, the coil traces, EFIT's current density up to then, beta_pol."""
    d = dict(np.load(os.path.join(SHOTS, f"{shot}.npz")))
    k0 = int(d["k0"])
    kk = list(d["keep"]).index(k0)
    lc = np.c_[d["lcfs_r"][:, kk], d["lcfs_z"][:, kk]]
    on = [i for i in range(2) if np.isfinite(d["pn_x"][i]) and abs(d["pn_x"][i] - 1.0) < 0.01]
    return dict(d=d, R=d["R"].astype(float), Z=d["Z"].astype(float), psi0=d["psi"][:, :, kk].astype(float), j0=d["j_keep"][:, :, kk].astype(float), lcfs=lc[np.isfinite(lc).all(axis=1)],
                axis=(float(d["magnetic_axis_r"][k0]), float(d["magnetic_axis_z"][k0])), bvac_rmag=float(d["bvac_rmag"][k0]), betap=float(d["beta_pol"][k0]), t0=float(d["t_tq"]),
                t_pf=d["t_pf"], coils={n: d[n] for n in COILS}, t_eq=d["t_eq"][: k0 + 1], j_hist=d["j_tor"],
                nulls=dict(null_R=[float(d["x_point_r"][i, k0]) for i in on], null_Z=[float(d["x_point_z"][i, k0]) for i in on]) if on else {})


def natural_state(shot, t, betap):
    """For the harness: a shot of the hands-off run's set (runs/twin/mast_natural2), EFIT's slice nearest before t, its coil traces and current density,
    and beta_pol from the run (TORAX), not from EFIT. No X-point constraints: those files do not hold them."""
    g = dict(np.load(os.path.join(ROOT, "runs", "twin", "mast_natural2", "shots", f"{shot}.npz"), allow_pickle=False))
    c = dict(np.load(os.path.join(ROOT, "runs", "twin", "mast_natural2", "coils", f"{shot}.npz")))
    tg = np.asarray(g["t"], float)
    k = int(np.flatnonzero(tg <= t + 1e-9)[-1]) if np.any(tg <= t + 1e-9) else 0
    ke = int(np.argmin(np.abs(np.asarray(c["t_eq"], float) - tg[k])))
    lc = np.c_[g["lcfs_r"][:, k], g["lcfs_z"][:, k]]
    return dict(d=None, R=np.asarray(g["efit_R"], float), Z=np.asarray(g["efit_z"], float), psi0=np.asarray(g["efit_psi"][:, :, k], float), j0=np.nan_to_num(np.asarray(c["j_tor"][:, :, ke], float)),
                lcfs=lc[np.isfinite(lc).all(axis=1)], axis=(float(g["R_mag"][k]), float(g["Z_mag"][k])), bvac_rmag=float(g["bvac_rmag"][k]), betap=float(betap), t0=float(t),
                t_pf=c["t"], coils={n: c[n] for n in COILS}, t_eq=np.asarray(c["t_eq"], float)[: ke + 1], j_hist=np.nan_to_num(c["j_tor"][:, :, : ke + 1]), nulls={})


def start(shot, args):
    """An extracted disruption's state at its thermal quench in FreeGSNKE, and the evolving solver built on it (see run)."""
    return start_state(shot_state(shot), args, label=shot)


def start_state(st0, args, label="", build=True):
    """FreeGSNKE's evolving solver built on a state (shot_state or natural_state): the current refitted (ConstrainBetapIp, or Lao85), the coils at t0, the
    vessel's currents from their circuit run up from the shot's start, P6 and the far coil found to hold EFIT's boundary. Returns what the stepping needs."""
    sys.path.insert(0, os.path.join(ROOT, "external", "freegsnke"))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import freegsnke_server as fs
    import mast_machine as mm
    from freegsnke import nonlinear_solve
    from freegsnke.jtor_update import Lao85
    shot = label
    if True:
        d = st0["d"]
        R, Z = st0["R"], st0["Z"]
        dA = (R[1] - R[0]) * (Z[1] - Z[0])
        psi0, j0, lc, axis = st0["psi0"], st0["j0"], st0["lcfs"], st0["axis"]
        t_tq = st0["t0"]
        coils_at = lambda t: {n: float(np.interp(t, st0["t_pf"], st0["coils"][n])) for n in COILS}
        s = fs.Server()
        s.build("MAST", Rmin=float(R[0]), Rmax=float(R[-1]), Zmin=float(Z[0]), Zmax=float(Z[-1]), nx=len(R), ny=len(Z), paths=machine3())
        names = s.active_names()
        grid, pp, ffp, left = mm.fit_profiles(R, Z, psi0, j0, lc, axis)
        sign = float(np.sign((j0 * dA).sum()))
        Ip = abs(float((j0 * dA).sum()))
        fvac = abs(st0["bvac_rmag"] * axis[0])
        alpha, beta = lao(grid, sign * pp, sign * ffp)
        s.table = dict(Ip=Ip, fvac=fvac, psi_n=grid, pprime=sign * pp, ffprime=sign * ffp, Raxis=1.0)
        if getattr(args, "profile", "betap") == "lao":
            s.profiles = Lao85(eq=s.eq, Ip=Ip, fvac=fvac, alpha=alpha, beta=beta, alpha_logic=True, beta_logic=True, Raxis=1.0, Ip_logic=True)
            shape = None
        else:
            from freegsnke.jtor_update import ConstrainBetapIp
            am, an, left = jeon_fit(R, Z, psi0, j0, lc, axis)
            betap = st0["betap"]
            s.profiles = ConstrainBetapIp(eq=s.eq, betap=betap, Ip=Ip, fvac=fvac, alpha_m=am, alpha_n=an, Raxis=axis[0])
            shape = dict(alpha_m=am, alpha_n=an, betap=betap)
        s.currents({**coils_at(t_tq), "BV": 0.0, "P6": 0.0})
        s.vessel_start(R, Z, st0["t_pf"], st0["coils"], st0["t_eq"], st0["j_hist"], t_tq, resist=args.resist)
        s.vessel.apply()
        s.guess(psi0.T)
        pick = lc[:: max(1, len(lc) // 16)]
        res = fs.limited(s.inverse, dict(R=pick[:, 0], Z=pick[:, 1], tol=1e-4, max_iter=60, l2_reg=1e-10, frozen=COILS, **st0["nulls"]), 180.0)      # FreeGSNKE's inverse solve can loop for ever with no X-point
        start = dict(BV=float(res["currents_active"]["BV"]), P6=float(res["currents_active"]["P6"]), Z_axis=float(res["axis"][1]), Z_axis_efit=axis[1], R_axis=float(res["axis"][0]), R_axis_efit=axis[0],
                     dist_cm=100 * float(np.median([np.min(np.hypot(res["lcfs"][:, 0] - a, res["lcfs"][:, 1] - b)) for a, b in lc[::4]])) if res["lcfs"] is not None else None, Ip=float(res["Ip"]), profile_fit_left=left)
        print(shot, "start:", {k: round(v, 4) if isinstance(v, float) else v for k, v in start.items()}, flush=True)
        resist = np.asarray(s.eq.tokamak.coil_resist, float).copy()
        resist[resist <= 0] = 1e-7
        idx = {n: names.index(n) for n in names}
        if args.drive == "current":
            for n in COILS:
                resist[idx[n]] = R_BIG
        dt0 = getattr(args, "dt_tq", args.dt) if getattr(args, "quiet", 0) == 0 else args.dt          # built at the step it takes first, so no reset before the collapse
        nl = None
        if build:                                                  # build=False: the start only, for static solves
            nl = nonlinear_solve.nl_solver(eq=s.eq, profiles=s.profiles, GSStaticSolver=s.solver, full_timestep=dt0, max_internal_timestep=dt0, plasma_resistivity=args.eta_hot, max_mode_frequency=args.mode_freq, custom_coil_resist=resist, target_relative_tolerance_linearization=1e-6, n_linearization_workers=1)
            nl.initialize_from_ICs(s.eq, s.profiles)
        held = {n: resist[idx[n]] * float(s.eq.tokamak.getCurrents()[n]) for n in names}
        a_tq = alpha * args.p_left
        b_flat = np.array([float(beta[0]), 0.0, 0.0, 0.0])          # FF' on the axis kept, 1 - x^4 across: a flat current
        b_tq = (1 - args.flat) * beta + args.flat * b_flat
        if shape is not None:
            start.update(shape)

        def collapse(w):
            """Profile parameters a fraction w of the way through the collapse."""
            if shape is None:
                return dict(alpha=(1 - w) * alpha + w * a_tq, beta=(1 - w) * beta + w * b_tq)
            return dict(alpha_m=shape["alpha_m"], alpha_n=shape["alpha_n"] * (1 - w + w * args.an_after), betap=shape["betap"] * (1 - w + w * args.p_left))
    return dict(d=d, s=s, nl=nl, names=names, held=held, alpha=alpha, beta=beta, a_tq=a_tq, b_tq=b_tq, start=start, R=R, Z=Z, Ip=Ip, t_tq=t_tq, coils_at=coils_at, resist=resist, collapse=collapse)


def static_shift(args):
    """Where the collapse moves the equilibrium point (H78): from each extracted disruption's start (coils, vessel currents, P6 and the far coil as in
    run), static free-boundary solves with the coils' and the vessel's currents frozen and the current held, along the collapse to w = 1. The point moves by
    (dR, dZ). A vertically unstable plasma held back by the vessel ends on the far side of the point it would balance at, so it should move the other way:
    predicted direction -sign(dZ). Compared with the direction measured on the centre column (config.json, motion). -> runs/twin/disruptions/vde/static.json"""
    import time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import freegsnke_server as fs
    measured = {r["shot"]: r for r in json.load(open(os.path.join(OUT, "config.json")))}
    shots = args.shots or [r["shot"] for r in json.load(open(os.path.join(OUT, "shots.json")))]
    path = os.path.join(OUT, "static.json")
    rows = json.load(open(path)) if os.path.exists(path) and not args.fresh else []
    done = {r["shot"] for r in rows}
    for shot in shots:
        if shot in done:
            continue
        t0 = time.perf_counter()
        try:
            st = start_state(shot_state(shot), args, label=shot, build=False)
            s, collapse = st["s"], st["collapse"]
            row = dict(shot=shot, start=st["start"], path=[])
            for w in (0.0, 0.25, 0.5, 1.0):
                for k, v in collapse(w).items():
                    setattr(s.profiles, k, v)
                fs.limited(s.solver.forward_solve, dict(eq=s.eq, profiles=s.profiles, target_relative_tolerance=1e-5, max_solving_iterations=80, verbose=False), 120.0)
                o = s.eq.opt[0]
                row["path"].append(dict(w=w, R_axis=float(o[0]), Z_axis=float(o[1])))
            p0, p1 = row["path"][0], row["path"][-1]
            row.update(dR_cm=100 * (p1["R_axis"] - p0["R_axis"]), dZ_cm=100 * (p1["Z_axis"] - p0["Z_axis"]))
            m = measured.get(shot, {})
            row.update(measured_shift_cm=m.get("moment_shift_cm"), cq_80_20_ms=m.get("cq_80_20_ms"), drsep_cm=m.get("drsep_cm"), wall_s=time.perf_counter() - t0)
            rows.append(row)
            print(shot, f"dR {row['dR_cm']:+.1f} cm, dZ {row['dZ_cm']:+.2f} cm: predicted {'down' if row['dZ_cm'] > 0 else 'up'}; measured shift {row['measured_shift_cm']}, quench {row['cq_80_20_ms']} ms; P6 {st['start']['P6']:.0f} A; {row['wall_s']:.0f} s", flush=True)
        except Exception as err:
            print(shot, "failed", repr(err)[:200], flush=True)
        json.dump(rows, open(path, "w"), indent=0, default=float)
    ok = [r for r in rows if r.get("measured_shift_cm") is not None and abs(r["measured_shift_cm"]) > 10]
    agree = sum((r["dZ_cm"] > 0) == (r["measured_shift_cm"] < 0) for r in ok)
    print(f"direction: predicted against measured on {len(ok)} disruptions that moved more than 10 cm: {agree} agree", flush=True)


def quench(nl, names, held, collapse, coils_at, Ip, t_tq, args, shot=""):
    """The collapse and the current quench on a built FreeGSNKE evolving solver: --quiet steps with nothing changed, then the collapse at --dt_tq
    (collapse(w) gives the profile parameters a fraction w through it; the resistivity rises from --eta_hot to --eta), a relinearisation about the
    collapsed plasma, then --dt steps to --after. A step that raises, or moves the current by 30% or the axis by 10 cm, is refused: back to the last
    good state with the linear model rebuilt there, and half the change (collapse) or half the step (after), up to --retries times. Supplies hold
    their voltages (held), or P1 to P5 follow coils_at(t) (--drive current). A step over --step_limit seconds of wall time is refused too. Returns
    rows, flux maps and the refusals."""
    import time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import freegsnke_server as fs
    rows, maps, log = [], [], []
    t = t_tq
    state = dict(dt=float(nl.dt_step), w=0.0, dw=1.0 / args.ramp, quiet=args.quiet, dt_c=args.dt_tq)      # dt_c: the collapse's own step, halved with the change

    def set_dt(dt):
        if abs(dt - state["dt"]) > 1e-15:
            nl.reset_timestep(full_timestep=dt, max_internal_timestep=dt)
            state["dt"] = dt

    def snap():
        return nl.eq1.create_auxiliary_equilibrium(), nl.profiles1.copy(), {k: float(v) for k, v in nl.eq1.tokamak.getCurrents().items()}

    def back(sn):
        """Back to a saved state, with the linear model rebuilt there (FreeGSNKE's initialize_from_ICs does both)."""
        eq, pr, cur = sn
        for k, v in cur.items():
            eq.tokamak.set_coil_current(coil_label=k, current_value=v)
        nl.initialize_from_ICs(eq, pr)
        nl.linearised_sol.calculate_linear_growth_rate()           # FreeGSNKE computes these only when the solver is built

    def now():
        eq1, pr1 = nl.eq1, nl.profiles1
        cur = eq1.tokamak.getCurrents()
        opt = eq1.opt
        return dict(t_ms=1e3 * (t - t_tq), Ip=float(eq1.plasmaCurrent()), R_axis=float(opt[0][0]) if len(opt) else np.nan, Z_axis=float(opt[0][1]) if len(opt) else np.nan,
                    limited=bool(getattr(pr1, "flag_limiter", False)), w=state["w"], **{f"I_{n}": float(cur[n]) for n in names}, vessel=float(sum(cur[n] for n in eq1.tokamak.coils_list[len(names):])))

    last = now()
    good = snap()
    done = False
    while t < t_tq + args.after - 1e-12 and not done:
        collapsing = state["quiet"] <= 0 and state["w"] < 1.0 - 1e-9
        if collapsing:
            set_dt(state["dt_c"])
        elif state["quiet"] <= 0 and state.get("relinearised") is None:
            t_rel = time.perf_counter()
            back(snap())                                        # the linear model about the collapsed plasma, not the one before
            state["relinearised"] = True
            print(shot, f"relinearised after the collapse in {time.perf_counter() - t_rel:.0f} s; growth rates {np.round(nl.linearised_sol.growth_rates, 1).tolist()} per s", flush=True)
            good = snap()
        t_step = time.perf_counter()
        for attempt in range(args.retries + 1):
            if collapsing:
                w_next = min(1.0, state["w"] + state["dw"])
                params, eta = collapse(w_next), args.eta_hot * (args.eta / args.eta_hot) ** w_next
            else:
                set_dt(min(args.dt, state["dt"] * (2.0 if attempt == 0 and state["dt"] < args.dt else 1.0)) if not collapsing else args.dt_tq)
                params, eta = None, None
            t_next = t + state["dt"]
            V = np.array([R_BIG * coils_at(t_next)[n] if (n in COILS and args.drive == "current") else held[n] for n in names])
            try:
                fs.limited(nl.nlstepper, dict(active_voltage_vec=V, profiles_parameters=params, plasma_resistivity=eta, target_relative_tol_currents=0.005, target_relative_tol_GS=0.003, verbose=0, max_solving_iterations=50), args.step_limit)      # a step past the limit is refused like any other: on two runs single steps sat for over an hour
                t_try = t
                t = t_next
                row = now()
                t = t_try
                ok = np.isfinite(row["Ip"]) and np.isfinite(row["Z_axis"]) and abs(row["Ip"] - last["Ip"]) < 0.3 * abs(last["Ip"]) and abs(row["Z_axis"] - last["Z_axis"]) < 0.10 and abs(row["R_axis"] - last["R_axis"]) < 0.10
                why = "" if ok else f"jump: Ip {last['Ip'] / 1e3:.0f} to {row['Ip'] / 1e3:.0f} kA, Z {100 * last['Z_axis']:+.1f} to {100 * row['Z_axis']:+.1f} cm"
            except Exception as err:
                ok, why = False, repr(err)[:120]
            if ok:
                break
            log.append(dict(t_ms=1e3 * (t - t_tq), attempt=attempt, why=why, dw=state["dw"], dt=state["dt"]))
            back(good)
            gr = np.round(np.sort(np.real(np.atleast_1d(nl.linearised_sol.growth_rates)))[::-1][:3], 0).tolist()
            log[-1]["growth_per_s"] = gr
            print(shot, f"t {1e3 * (t - t_tq):.3f} ms: step refused ({why}); back to the last good state at w {state['w']:.3f}, relinearised (growth rates there {gr} per s), " + ("half the change" if collapsing else "half the step"), flush=True)
            if collapsing:
                state["dw"] /= 2
                state["dt_c"] /= 2                                       # the plasma may be moving fast: a shorter step as well as a smaller change
            else:
                set_dt(state["dt"] / 2)
        else:
            print(shot, f"gave up at {1e3 * (t - t_tq):.3f} ms after {args.retries} retries", flush=True)
            break
        t = t_next
        if collapsing:
            state["w"] = w_next
            state["dw"] = min(1.0 / args.ramp, 1.5 * state["dw"])        # past a hard stretch the change and the step grow back
            state["dt_c"] = min(args.dt_tq, 1.5 * state["dt_c"])
        elif state["quiet"] > 0:
            state["quiet"] -= 1
        row = now()
        rows.append(row)
        maps.append(np.asarray(nl.eq1.psi(), np.float32))
        last, good = row, snap()
        if len(rows) % args.print_every == 0 or collapsing:
            print(shot, f"t {row['t_ms']:6.3f} ms w {row['w']:.2f} Ip {row['Ip'] / 1e3:7.1f} kA Z_axis {100 * row['Z_axis']:+6.1f} cm R_axis {row['R_axis']:.3f} limited {row['limited']} P6 {row['I_P6'] / 1e3:+.2f} kA P5 {row['I_P5'] / 1e3:+.2f} kA, {time.perf_counter() - t_step:.1f} s", flush=True)
        if abs(row["Ip"]) < 0.05 * Ip:
            done = True
    return rows, maps, log


def run(args):
    """FreeGSNKE's evolving solver through one real disruption: start at the measured thermal quench with EFIT's last plasma before it; the vessel's
    currents come from their circuit run up from the shot's start; P6 and the far coil hold EFIT's boundary at the start. Then every supply holds its
    voltage (--drive hold) or P1 to P5 follow their measured currents (--drive current). The pressure goes (p' to --p_left of itself), the current flattens
    (FF' toward 1 - x^4 by the weight --flat) and the resistivity rises to --eta, over --ramp steps of --dt_tq. Saves the run's current, axis, coils and
    flux maps."""
    import time
    out_dir = os.path.join(OUT, "runs", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    for shot in args.shots:
        t_wall = time.perf_counter()
        st = start(shot, args)
        d, s, nl, names, held = st["d"], st["s"], st["nl"], st["names"], st["held"]
        alpha, beta, a_tq, b_tq, Ip, t_tq, coils_at, R, Z, start_info = st["alpha"], st["beta"], st["a_tq"], st["b_tq"], st["Ip"], st["t_tq"], st["coils_at"], st["R"], st["Z"], st["start"]
        rows, maps, log = quench(nl, names, held, st["collapse"], coils_at, Ip, t_tq, args, shot=shot)
        np.savez_compressed(os.path.join(out_dir, f"{shot}.npz"), R=R, Z=Z, psi=np.stack(maps, axis=-1) if maps else np.zeros((len(Z), len(R), 0), np.float32), **{k: np.array([r[k] for r in rows]) for k in (rows[0] if rows else {})})
        json.dump(dict(shot=shot, start=start_info, args=vars(args), steps=len(rows), refused=log, wall_s=time.perf_counter() - t_wall), open(os.path.join(out_dir, f"{shot}.json"), "w"), indent=0, default=float)
        print(shot, f"{len(rows)} steps in {time.perf_counter() - t_wall:.0f} s", flush=True)


def summarise(rows, Ip0):
    """A quench's numbers: the spike, the current quench 80 to 20% of the current before, the axis's largest vertical shift and its sign, when the plasma
    first touched the limiter."""
    t = np.array([r["t_ms"] for r in rows])
    ip = np.abs(np.array([r["Ip"] for r in rows]))
    z = np.array([r["Z_axis"] for r in rows])
    out = dict(steps=len(rows), t_end_ms=float(t[-1]) if len(t) else None, spike_pct=100 * (float(ip.max()) / Ip0 - 1) if len(ip) else None)
    k80, k20 = np.flatnonzero(ip < 0.8 * Ip0), np.flatnonzero(ip < 0.2 * Ip0)
    out["cq_80_20_ms"] = float(t[k20[0]] - t[k80[0]]) if k80.size and k20.size else None
    if len(z):
        k = int(np.argmax(np.abs(z - z[0])))
        out.update(z_shift_cm=100 * float(z[k] - z[0]), z_shift_at_ms=float(t[k]))
    lim = [r["t_ms"] for r in rows if r.get("limited")]
    out["limited_at_ms"] = float(lim[0]) if lim else None
    return out


# MAST's measured current quench by which way the plasma moved (236 development disruptions, config.json and motion; docs/digital-twin.md)
MEASURED_CQ_MS = dict(up=(2.2, 1.8, 2.4), down=(0.8, 0.6, 1.0), little=(5.4, 4.7, 6.1))


def static_at(args):
    """The fast quench for the harness (mode static): the start as in run, then static solves with the coils' and vessel's currents frozen, before the
    collapse and after it. Where the equilibrium point moves 1 to 30 cm, the plasma is predicted to move the other way (checked on MAST's disruptions,
    mast_vde_static_table.py); under 1 cm the model has no preferred direction; over 30 cm there is no equilibrium near. The current quench's length is
    MAST's measured median for the predicted direction, labelled as measured, not simulated. --growth 1 also builds the linear model for the growth rate."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import freegsnke_server as fs
    st = start_state(natural_state(args.shot, args.t, args.betap), args, label=args.shot, build=bool(getattr(args, "growth", 0)))
    s, collapse = st["s"], st["collapse"]
    path = []
    for w in (0.0, 1.0):
        for k, v in collapse(w).items():
            setattr(s.profiles, k, v)
        fs.limited(s.solver.forward_solve, dict(eq=s.eq, profiles=s.profiles, target_relative_tolerance=1e-5, max_solving_iterations=80, verbose=False), 120.0)
        o = s.eq.opt[0]
        path.append((float(o[0]), float(o[1])))
    dR, dZ = 100 * (path[1][0] - path[0][0]), 100 * (path[1][1] - path[0][1])
    if abs(dZ) > 30:
        group, direction = "no equilibrium near", ("down" if dZ < 0 else "up")
    elif abs(dZ) >= 1:
        group, direction = "clear shift", ("down" if dZ > 0 else "up")
    else:
        group, direction = "no clear shift", None
    out = dict(shot=args.shot, t=args.t, betap=args.betap, start=st["start"], dR_cm=dR, dZ_cm=dZ, group=group, direction=direction,
               cq_80_20_ms_measured_on_mast=MEASURED_CQ_MS.get(direction) if direction else None,
               note="direction from FreeGSNKE static equilibria (physics); current quench length: MAST measured median for that direction (data, MAST only)")
    if st["nl"] is not None:
        out["growth_per_s"] = [float(x) for x in np.real(np.atleast_1d(st["nl"].linearised_sol.growth_rates))]
    return out


def quench_at(args):
    """For the harness (twin.py's quench command): a MAST shot of the hands-off set at time --t, beta_pol --betap from the run, through the collapse and
    the current quench (see run). Writes runs/twin/disruptions/vde/harness/<tag>.npz and .json and prints the summary as one line of JSON."""
    import time
    t_wall = time.perf_counter()
    folder = os.path.join(OUT, "harness")
    os.makedirs(folder, exist_ok=True)
    if getattr(args, "mode", "static") == "static":
        summ = static_at(args)
        summ["wall_s"] = time.perf_counter() - t_wall
        json.dump(dict(summary=summ, args=vars(args)), open(os.path.join(folder, f"{args.tag}.json"), "w"), indent=0, default=float)
        print("QUENCH " + json.dumps(summ, default=float), flush=True)
        return
    st = start_state(natural_state(args.shot, args.t, args.betap), args, label=args.shot)
    rows, maps, log = quench(st["nl"], st["names"], st["held"], st["collapse"], st["coils_at"], st["Ip"], st["t_tq"], args, shot=args.shot)
    np.savez_compressed(os.path.join(folder, f"{args.tag}.npz"), R=st["R"], Z=st["Z"], psi=np.stack(maps, axis=-1) if maps else np.zeros((len(st["Z"]), len(st["R"]), 0), np.float32), **{k: np.array([r[k] for r in rows]) for k in (rows[0] if rows else {})})
    summ = dict(shot=args.shot, t=args.t, betap=args.betap, start=st["start"], **(summarise(rows, st["Ip"]) if rows else {}), refused=len(log), wall_s=time.perf_counter() - t_wall)
    json.dump(dict(summary=summ, refused=log, args=vars(args)), open(os.path.join(folder, f"{args.tag}.json"), "w"), indent=0, default=float)
    print("QUENCH " + json.dumps(summ, default=float), flush=True)


def moment(B, zc):
    """The centre column's B_Z moment, cm: sum(Z B) / sum(|B|) over channels with a place, for B [channel, time]."""
    ok = np.isfinite(zc) & np.isfinite(B).all(axis=1)
    return 100 * (zc[ok, None] * B[ok]).sum(0) / np.abs(B[ok]).sum(0)


def compare(args):
    """A run against its shot: the current and the centre column's moment, the model's read through the probe model of the sensors work
    (sensors.read_magnetics on its flux maps) and the real one from the shot's probes, at the same times from the thermal quench."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import sensors
    geo = json.load(open(os.path.join(ROOT, "runs", "twin", "sensors", "geometry.json")))
    zc = np.full(40, np.nan)
    for c, g in geo["ccbv"].items():
        zc[int(c)] = g["Z"]
    folder = os.path.join(OUT, "runs", args.tag)
    rows = []
    for path in sorted(p for p in os.listdir(folder) if p.endswith(".npz")):
        shot = int(path[:-4])
        r = dict(np.load(os.path.join(folder, path)))
        d = dict(np.load(os.path.join(SHOTS, f"{shot}.npz")))
        if r["psi"].shape[-1] == 0:
            continue
        R, Z = r["R"], r["Z"]
        Bm = np.stack([sensors.read_magnetics(r["psi"][:, :, k].astype(float), R, Z, geo)["ccbv"] for k in range(r["psi"].shape[-1])], axis=1)
        Mm = moment(Bm, zc)
        tr = 1e3 * (d["t_mag"] - d["t_tq"])
        Mr = moment(d["ccbv"], zc)
        ip_r = np.abs(d["ip"])
        i0 = float(np.median(ip_r[(tr >= -1.5) & (tr <= -0.5)]))
        m0 = float(np.median(Mr[(tr >= -1.5) & (tr <= -0.5)]))
        tp = 1e3 * (d["t_pf"] - d["t_tq"])
        coil0 = {n: float(np.interp(0.0, tp, d[n])) for n in COILS[1:]}
        print(f"{shot}: t ms | model Ip/Ip0, moment shift cm, Z_axis cm | real Ip/Ip0, moment shift cm | coils' change, kA, model (real): " + " ".join(COILS[1:]))
        for tm in (0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0):
            k = int(np.argmin(np.abs(r["t_ms"] - tm)))
            if abs(r["t_ms"][k] - tm) > 0.06:
                continue
            j = int(np.argmin(np.abs(tr - tm)))
            dc = " ".join(f"{(r['I_' + n][k] - r['I_' + n][0]) / 1e3:+.2f} ({(float(np.interp(tm, tp, d[n])) - coil0[n]) / 1e3:+.2f})" for n in COILS[1:])
            print(f"   {tm:4.1f} | {abs(r['Ip'][k]) / abs(r['Ip'][0]):.2f} {Mm[k] - Mm[0]:+6.1f} {100 * r['Z_axis'][k]:+6.1f} | {ip_r[j] / i0:.2f} {Mr[j] - m0:+6.1f} | {dc}")
        rows.append(dict(shot=shot, t_ms=r["t_ms"].tolist(), model_ip=(np.abs(r["Ip"]) / abs(r["Ip"][0])).tolist(), model_shift=(Mm - Mm[0]).tolist(), model_Z_axis=r["Z_axis"].tolist()))
    json.dump(rows, open(os.path.join(folder, "compare.json"), "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["extract", "config", "motion", "run", "compare", "quench_at", "static"])
    ap.add_argument("--fresh", action="store_true", help="static: start the table again")
    ap.add_argument("--shot", type=int, help="quench_at: a shot of the hands-off set")
    ap.add_argument("--t", type=float, help="quench_at: s, when the collapse starts")
    ap.add_argument("--betap", type=float, help="quench_at: beta_pol from the run")
    ap.add_argument("--mode", choices=["static", "full"], default="static", help="quench_at: static (a minute: direction from static equilibria) or full (the time-dependent run, hours)")
    ap.add_argument("--growth", type=int, default=0, help="quench_at static: 1 adds growth rates from the linear model (about 100 s)")
    ap.add_argument("--tag", default="first")
    ap.add_argument("--dt", type=float, default=5e-5)
    ap.add_argument("--after", type=float, default=0.006, help="s simulated from the thermal quench")
    ap.add_argument("--eta", type=float, default=1e-4, help="ohm m after the thermal quench (Spitzer at about 5 eV with Z_eff 2)")
    ap.add_argument("--eta_hot", type=float, default=1e-7)
    ap.add_argument("--p_left", type=float, default=1e-3, help="share of the pressure left after the thermal quench (beta_pol, or p' for --profile lao)")
    ap.add_argument("--profile", choices=["betap", "lao"], default="betap", help="betap: FreeGSNKE's ConstrainBetapIp fitted to EFIT (robust); lao: Lao85 polynomials (the collapse loses the plasma)")
    ap.add_argument("--an_after", type=float, default=0.3, help="betap profile: alpha_n after the collapse as a share of its fitted value; smaller is flatter")
    ap.add_argument("--flat", type=float, default=1.0, help="0: FF' keeps its shape; 1: FF' becomes 1 - x^4")
    ap.add_argument("--resist", type=float, default=1.0, help="vessel resistance factor")
    ap.add_argument("--mode_freq", type=float, default=1e4, help="vessel modes kept up to this rate, 1/s")
    ap.add_argument("--print_every", type=int, default=5)
    ap.add_argument("--ramp", type=int, default=10, help="steps of --dt_tq over which the collapse is applied")
    ap.add_argument("--dt_tq", type=float, default=1e-5, help="s, the step during the collapse")
    ap.add_argument("--relin", type=int, default=1, help="1: relinearise once the collapse is done")
    ap.add_argument("--retries", type=int, default=6, help="a refused step goes back to the last good state, relinearises and halves the change (collapse) or the step (after), this many times")
    ap.add_argument("--step_limit", type=float, default=120.0, help="s of wall time a step may take before it is refused")
    ap.add_argument("--quiet", type=int, default=0, help="steps with nothing changed before the collapse (a check that the start holds)")
    ap.add_argument("--drive", choices=["current", "hold"], default="hold", help="hold: every supply holds its voltage, so the coils answer inductively; current: P1 to P5 follow their measured currents (slow, and an artefact mode in the linear model)")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--shots", type=int, nargs="*")
    ap.add_argument("--before", type=float, default=0.001, help="s: the last EFIT slice at least this long before the collapse")
    args = ap.parse_args()
    if args.command == "extract":
        extract(args)
    elif args.command == "config":
        os.makedirs(OUT, exist_ok=True)
        config(args)
    elif args.command == "motion":
        motion(args)
    elif args.command == "run":
        run(args)
    elif args.command == "compare":
        compare(args)
    elif args.command == "quench_at":
        quench_at(args)
    elif args.command == "static":
        static_shift(args)


if __name__ == "__main__":
    main()
