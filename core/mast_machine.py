"""A FreeGSNKE description of the original MAST (H46). FreeGSNKE ships MAST-U, ITER and SPARC. MAST-U kept MAST's P4, P5, P6 and much of its vessel, and replaced its solenoid and P2, P3 with divertor coils. FreeGS has MAST's coil centres without turns. So the turns of P1 and P2 are read from EFIT itself: EFIT's flux minus its plasma's own flux is the flux of the coils and the vessel, and it is linear in the measured coil currents, with the effective turns as the slopes. WSL, read-only on the store; the 19 held-out shots stay out.
Usage:
  python core/mast_machine.py fit          (WSL)
  $XTORAX_FREEGSNKE_PYTHON core/mast_machine.py build                                  (writes runs/twin/freeboundary/mast_machine/)
Outputs runs/twin/freeboundary/mast_turns.json.
"""
import json
import os
import pickle
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = os.path.abspath(os.environ.get("XTORAX_HOME") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the workspace: runs/twin and external/ (README, "Where things live")
STORE = os.path.expanduser(os.environ.get("XTORAX_MAST_STORE", "~/data/mast/tokamark_v1"))   # FAIR-MAST's TokaMark v1 store (zarr)
OUT = os.path.join(ROOT, "runs", "twin", "freeboundary")
MU0 = 4e-7 * np.pi
# centres from FreeGS's machine.MAST() (R, Z), solenoid from there too; P4, P5 agree with MAST-U's description
GROUPS = {"P1": [(0.15, z) for z in np.linspace(-1.4, 1.4, 29)], "P2": [(0.49, 1.76), (0.49, -1.76)], "P3": [(1.1, 1.1), (1.1, -1.1)], "P4": [(1.51, 1.095), (1.51, -1.095)], "P5": [(1.66, 0.52), (1.66, -0.52)]}


def green(R, Z, Rc, Zc):
    """Poloidal flux per radian at (R, Z) per amp in a loop at (Rc, Zc)."""
    from scipy.special import ellipe, ellipk
    k2 = 4 * R * Rc / ((R + Rc) ** 2 + (Z - Zc) ** 2)
    k2 = np.clip(k2, 1e-9, 1 - 1e-9)
    k = np.sqrt(k2)
    return MU0 / (2 * np.pi) * np.sqrt(R * Rc) * ((2 - k2) * ellipk(k2) - 2 * ellipe(k2)) / k


def fit():
    import zarr
    cat = json.load(open(os.path.join(ROOT, "runs", "twin", "mast_catalog", "catalog.json")))["shots"]
    natural = json.load(open(os.path.join(ROOT, "runs", "twin", "mast_catalog", "natural_test_shots.json")))
    held = {s["shot"] for s in natural["shots"] if s["split"] != "fit"}
    rng = np.random.default_rng(0)
    shots = [int(k) for k, x in cat.items() if int(k) not in held and x["phases"].get("flat_top")]
    rng.shuffle(shots)
    A, y, raw = [], [], []
    used = 0
    rows = []
    for shot in shots:
        if used >= 60:
            break
        try:
            r = zarr.open_group(os.path.join(STORE, f"{shot}.zarr"), mode="r")
            eq, pf = r["equilibrium"], r["pf_active"]
            R, Z = eq["major_radius"][:].astype(float), eq["z"][:].astype(float)
            t = eq["time"][:].astype(float)
            ft = cat[str(shot)]["phases"]["flat_top"]
            names = [str(x) for x in pf["current_channel"][:]]
            cur, sol, tp = pf["coil_current"], pf["solenoid_current"], pf["time"][:].astype(float)
            for tt in rng.uniform(ft["t0"], ft["t1"], 3):
                i = int(np.argmin(np.abs(t - tt)))
                psi = eq["psi"][:, :, i].astype(float)
                j = eq["j_tor"][:, :, i].astype(float) if eq["j_tor"].ndim == 3 else None
                if j is None or not np.isfinite(psi).all() or not np.isfinite(j).all():
                    continue
                if psi.shape != (len(Z), len(R)):
                    psi = psi.T
                if j.shape != (len(Z), len(R)):
                    j = j.T
                RR, ZZ = np.meshgrid(R, Z)
                dA = (R[1] - R[0]) * (Z[1] - Z[0])
                mask = np.abs(j) > 0
                if mask.sum() < 20:
                    continue
                ip_grid = float((j * dA).sum())
                pick = (RR > 0.28) & (RR < 1.45) & (np.abs(ZZ) < 1.5)
                pts_R, pts_Z = RR[pick][::3], ZZ[pick][::3]
                pl = np.array([np.sum(green(a, b, RR[mask], ZZ[mask]) * j[mask] * dA) for a, b in zip(pts_R, pts_Z)])
                sign = 1.0
                ext = psi[pick][::3] - pl
                k = int(np.argmin(np.abs(tp - t[i])))
                I = {n: float(cur[names.index(n), k]) for n in names}
                groups = {"P1": float(sol[k]), "P2": sum(I[n] for n in names if "P2" in n) / 4, "P3": sum(I[n] for n in names if "P3" in n) / 2, "P4": sum(I[n] for n in names if "P4" in n) / 2, "P5": sum(I[n] for n in names if "P5" in n) / 2}
                cols = []
                for g in ("P1", "P2", "P3", "P4", "P5"):
                    cols.append(sum(green(pts_R, pts_Z, a, b) for a, b in GROUPS[g]) * groups[g])
                A.append(np.stack(cols, axis=1))
                y.append(ext)
                rows.append(dict(shot=shot, t=float(t[i]), Ip_grid=ip_grid, groups=groups))
                raw.append((pts_R.astype(np.float32), pts_Z.astype(np.float32), ext.astype(np.float32), np.array([groups[g] for g in ("P1", "P2", "P3", "P4", "P5")])))
            used += 1
            if used % 10 == 0:
                print(used, shot, flush=True)
        except Exception as err:
            print(shot, "failed", repr(err)[:200], flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "mast_ext.npz"), R=np.concatenate([x[0] for x in raw]), Z=np.concatenate([x[1] for x in raw]), ext=np.concatenate([x[2] for x in raw]), I=np.stack([x[3] for x in raw]), n=np.array([len(x[0]) for x in raw]))
    A, y = np.vstack(A), np.concatenate(y)
    print("points", A.shape, "slices", len(rows), flush=True)
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ b
    names = ["P1", "P2", "P3", "P4", "P5"]
    out = dict(turns={n: float(v) for n, v in zip(names, b)}, rms_ext=float(np.sqrt(np.mean(y ** 2))), rms_left=float(np.sqrt(np.mean(res ** 2))), slices=len(rows), shots=used, Ip_grid_median=float(np.median([abs(x["Ip_grid"]) for x in rows])))
    print(json.dumps(out, indent=1), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "mast_turns.json"), "w"), indent=1)


def scan():
    """Coil geometry against EFIT's external flux: how much of what is left comes from the coils' assumed places? A search over the solenoid's radius and length and the coils' centres, from the data saved by fit. The measure includes radial derivatives, since the radial force on the plasma goes with the vertical field."""
    d = np.load(os.path.join(OUT, "mast_ext.npz"))
    R, Z, ext, I, n = d["R"].astype(float), d["Z"].astype(float), d["ext"].astype(float), d["I"], d["n"]
    ends = np.cumsum(n)
    start = ends - n
    slices = np.concatenate([np.full(a, i) for i, a in enumerate(n)])
    near = (R > 0.3) & (R < 1.4) & (np.abs(Z) < 1.2)

    def design(g):
        cols = []
        for name in ("P1", "P2", "P3", "P4", "P5"):
            basis = sum(green(R, Z, a, b) for a, b in g[name])
            cols.append(basis * I[slices, ("P1", "P2", "P3", "P4", "P5").index(name)])
        return np.stack(cols, axis=1)

    def score(g):
        A = design(g)
        b, *_ = np.linalg.lstsq(A[near], ext[near], rcond=None)
        res = ext - A @ b
        return float(np.sqrt(np.mean(res[near] ** 2))), b

    base = {k: list(v) for k, v in GROUPS.items()}
    rms, b = score(base)
    print("base", round(rms, 5), np.round(b, 2).tolist(), flush=True)
    best = dict(base)
    best_rms = rms
    results = []
    for Rs in (0.10, 0.12, 0.14, 0.16, 0.18, 0.20):
        for L in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0):
            g = dict(best)
            g["P1"] = [(Rs, z) for z in np.linspace(-L, L, 29)]
            r, c = score(g)
            results.append((r, Rs, L))
            if r < best_rms:
                best_rms, best = r, g
                best_b = c
    results.sort()
    print("solenoid radius, half length, best five:", [(round(a, 5), b_, c_) for a, b_, c_ in results[:5]], flush=True)
    for name, scan_R, scan_Z in (("P2", (0.35, 0.45, 0.5, 0.55, 0.65, 0.75), (1.5, 1.6, 1.7, 1.8, 1.9, 2.0)), ("P4", (1.35, 1.45, 1.5, 1.55, 1.65), (0.95, 1.05, 1.1, 1.15, 1.25)), ("P5", (1.5, 1.6, 1.65, 1.7, 1.8), (0.4, 0.5, 0.55, 0.6, 0.7))):
        results = []
        for a in scan_R:
            for z in scan_Z:
                g = dict(best)
                g[name] = [(a, z), (a, -z)]
                r, c = score(g)
                results.append((r, a, z))
                if r < best_rms:
                    best_rms, best = r, g
        results.sort()
        print(name, "R, Z, best three:", [(round(a, 5), b_, c_) for a, b_, c_ in results[:3]], flush=True)
    rms, b = score(best)
    out = dict(rms=rms, rms_base=score({k: list(v) for k, v in GROUPS.items()})[0], rms_ext=float(np.sqrt(np.mean(ext[near] ** 2))), turns=dict(zip(("P1", "P2", "P3", "P4", "P5"), b.tolist())), groups={k: [list(x) for x in v] for k, v in best.items()})
    print(json.dumps({k: v for k, v in out.items() if k != "groups"}), {k: out["groups"][k][0] for k in out["groups"]}, flush=True)
    json.dump(out, open(os.path.join(OUT, "mast_geometry.json"), "w"), indent=0)


def build():
    """The machine's pickles for FreeGSNKE, from the fitted turns. Each coil a filament at FreeGS's centre carrying its turns as a current multiplier; the solenoid a stack of filaments. Passive elements from FreeGSNKE's MAST-U description, which shares MAST's vessel but has its divertor structures too: first version. Limiter: centre column and a box, MAST's own not being public in this data."""
    turns = json.load(open(os.path.join(OUT, "mast_turns.json")))["turns"]
    eta = 1.55e-8
    # the solenoid drawn fine: at the turns' fitted spacing of 10 cm its filaments, 6 cm from the plasma's inner edge, put a ripple in the flux there, and TORAX's reader refuses the wiggling surfaces
    fine = np.linspace(-1.4, 1.4, 141)
    active = {"P1": dict(R=[0.15] * len(fine), Z=fine.tolist(), dR=0.05, dZ=0.02, resistivity=eta, polarity=1, multiplier=turns["P1"] * len(GROUPS["P1"]) / len(fine))}
    for g, (dR, dZ) in dict(P2=(0.1, 0.1), P3=(0.1, 0.1), P4=(0.1, 0.13), P5=(0.1, 0.13)).items():
        active[g] = {}
        for label, (r, z) in zip(("upper", "lower"), GROUPS[g]):
            active[g][label] = dict(R=[r], Z=[z], dR=dR, dZ=dZ, resistivity=eta, polarity=1, multiplier=turns[g])
    active["BV"] = dict(R=[6.0], Z=[0.0], dR=0.1, dZ=0.1, resistivity=eta, polarity=1, multiplier=100.0)      # not a coil of MAST's: a far loop, 1 mT per 95 A, the coil the position feedback moves
    folder = os.path.join(OUT, "mast_machine2")
    os.makedirs(folder, exist_ok=True)
    pickle.dump(active, open(os.path.join(folder, "active.pickle"), "wb"))
    src = os.path.join(ROOT, "external", "freegsnke", "machine_configs", "MAST-U")
    passive = pickle.load(open(os.path.join(src, "MAST-U_like_passive_coils.pickle"), "rb"))
    pickle.dump(passive, open(os.path.join(folder, "passive.pickle"), "wb"))
    box = [dict(R=r, Z=z) for r, z in zip([0.2, 0.2, 1.75, 1.75, 0.2], [-1.85, 1.85, 1.85, -1.85, -1.85])]
    pickle.dump(box, open(os.path.join(folder, "limiter.pickle"), "wb"))
    pickle.dump(box, open(os.path.join(folder, "wall.pickle"), "wb"))
    print("written", folder, {k: (v["multiplier"] if "multiplier" in v else {a: b["multiplier"] for a, b in v.items()}) for k, v in active.items()})


def extract():
    """WSL: slices of the fit shots for checking the machine against EFIT: flux, current density, boundary, coil currents. One slice every 40 ms in each shot's window."""
    import zarr
    natural = json.load(open(os.path.join(ROOT, "runs", "twin", "mast_catalog", "natural_test_shots.json")))
    rows = {}
    n = 0
    for s in natural["shots"]:
        if s["split"] != "fit":
            continue
        shot = s["shot"]
        r = zarr.open_group(os.path.join(STORE, f"{shot}.zarr"), mode="r")
        eq, pf = r["equilibrium"], r["pf_active"]
        t = eq["time"][:].astype(float)
        names = [str(x) for x in pf["current_channel"][:]]
        tp = pf["time"][:].astype(float)
        R, Z = eq["major_radius"][:].astype(float), eq["z"][:].astype(float)
        idx = np.where((t >= s["window"][0]) & (t <= s["window"][1]))[0][::8]
        for i in idx:
            psi, j = eq["psi"][:, :, i].astype(float), eq["j_tor"][:, :, i].astype(float)
            if psi.shape != (len(Z), len(R)):
                psi, j = psi.T, j.T
            k = int(np.argmin(np.abs(tp - t[i])))
            cur = pf["coil_current"]
            I = {a: float(cur[names.index(a), k]) for a in names}
            lc = np.c_[eq["lcfs_r"][:, i].astype(float), eq["lcfs_z"][:, i].astype(float)]
            lc = lc[np.isfinite(lc).all(axis=1)]
            key = f"{shot}_{int(i)}"
            rows[key + "_psi"], rows[key + "_j"], rows[key + "_lcfs"] = psi.astype(np.float32), j.astype(np.float32), lc.astype(np.float32)
            rows[key + "_info"] = np.array([shot, t[i], float(pf["solenoid_current"][k]), sum(I[a] for a in names if "P2" in a) / 4, sum(I[a] for a in names if "P3" in a) / 2, sum(I[a] for a in names if "P4" in a) / 2, sum(I[a] for a in names if "P5" in a) / 2, float(eq["magnetic_axis_r"][i]), float(eq["magnetic_axis_z"][i]), abs(float(eq["bvac_rmag"][i]) * float(eq["magnetic_axis_r"][i])), float(eq["elongation"][i]), float(eq["rpsi100_in"][i]), float(eq["rpsi100_out"][i])])
            n += 1
        rows["R"], rows["Z"] = R, Z
        print(shot, n, flush=True)
    np.savez_compressed(os.path.join(OUT, "mast_slices.npz"), **rows)


def fit_profiles(R, Z, psi, j, lcfs, axis):
    """p' and FF' against normalized flux from EFIT's current density: J = R p' + FF' / (mu0 R), cubic in psi_n for each. Returns grid, p', FF' (per unit psi_n, up to a common scale), and psi_n at the grid."""
    from scipy.interpolate import RectBivariateSpline
    spl = RectBivariateSpline(Z, R, psi, kx=3, ky=3)
    ax = float(spl.ev(axis[1], axis[0]))
    bd = float(np.median(spl.ev(lcfs[:, 1], lcfs[:, 0])))
    RR, ZZ = np.meshgrid(R, Z)
    pn = (psi - ax) / (bd - ax)
    mask = (np.abs(j) > 0) & (pn >= 0) & (pn <= 1)
    x, r, y = pn[mask], RR[mask], j[mask]
    basis = [x ** k * (1 - x) for k in range(4)]
    A = np.stack([b * r for b in basis] + [b / (MU0 * r) for b in basis], axis=1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    grid = np.linspace(0.0, 1.0, 41)
    B = np.stack([grid ** k * (1 - grid) for k in range(4)], axis=1)
    fitted = A @ c
    return grid, B @ c[:4], B @ c[4:], float(np.sqrt(np.mean((y - fitted) ** 2)) / np.sqrt(np.mean(y ** 2)))


def current_on_map(R, Z, psi, lcfs, axis, grid, pp, ffp, Ip):
    """The toroidal current density [z, R] of a plasma with these p' and FF' against normalized flux, on a given flux map, scaled to Ip."""
    from scipy.interpolate import RectBivariateSpline
    spl = RectBivariateSpline(Z, R, psi, kx=3, ky=3)
    ax = float(spl.ev(axis[1], axis[0]))
    bd = float(np.median(spl.ev(lcfs[:, 1], lcfs[:, 0])))
    RR, ZZ = np.meshgrid(R, Z)
    pn = (psi - ax) / (bd - ax)
    inside = (pn >= 0) & (pn <= 1) & (ZZ >= lcfs[:, 1].min()) & (ZZ <= lcfs[:, 1].max()) & (RR >= lcfs[:, 0].min()) & (RR <= lcfs[:, 0].max())
    j = np.zeros_like(psi)
    j[inside] = RR[inside] * np.interp(pn[inside], grid, pp) + np.interp(pn[inside], grid, ffp) / (MU0 * RR[inside])
    return j * Ip / (j.sum() * (R[1] - R[0]) * (Z[1] - Z[0]))


def passive_fit(srv, psi, j, R, Z, ridge=1e-3):
    """Vessel currents at this moment: what EFIT's flux needs beyond its plasma and the measured coils, as a smooth set of currents in FreeGSNKE's passive elements. Sets them. Returns the rms left, over the rms before, and the total."""
    tok = srv.eq.tokamak
    names = list(tok.coils_list[tok.n_active_coils:])
    RR, ZZ = np.meshgrid(R, Z)
    dA = (R[1] - R[0]) * (Z[1] - Z[0])
    mask = np.abs(j) > 0
    pick = (RR > 0.28) & (RR < 1.65) & (np.abs(ZZ) < 1.7)
    pR, pZ = RR[pick][::2], ZZ[pick][::2]
    pl = np.array([np.sum(green(a, b, RR[mask], ZZ[mask]) * j[mask] * dA) for a, b in zip(pR, pZ)])
    coils = np.zeros_like(pR)
    for name in ("P1", "P2", "P3", "P4", "P5"):
        c = tok.coils_dict[name]
        coords = c["coords"]
        I = tok.getCurrents()[name] if hasattr(tok, "getCurrents") else 0.0
        coils += I * np.sum([m * green(pR, pZ, a, b) for a, b, m in zip(coords[0], coords[1], np.resize(c["multiplier"], len(coords[0])))], axis=0)
    res = psi[pick][::2] - pl - coils
    basis = []
    for n in names:
        c = tok.coils_dict[n]
        coords = c["coords"]
        mult = np.resize(c["multiplier"], len(coords[0]))
        basis.append(np.sum([m * green(pR, pZ, a, b) for a, b, m in zip(coords[0], coords[1], mult)], axis=0))
    B = np.stack(basis, axis=1)
    scale = np.sqrt((B ** 2).mean(axis=0))
    Bn = B / scale
    x = np.linalg.solve(Bn.T @ Bn + ridge * len(res) * np.eye(Bn.shape[1]), Bn.T @ res)
    left = res - Bn @ x
    I = x / scale
    for n, v in zip(names, I):
        tok.set_coil_current(coil_label=n, current_value=float(v))
    return float(np.sqrt(np.mean(left ** 2)) / max(np.sqrt(np.mean(res ** 2)), 1e-9)), float(np.sqrt(np.mean(res ** 2))), float(I.sum())


def check():
    """Machine against EFIT, no TORAX: the measured coil currents and EFIT's plasma, solved free-boundary with FreeGSNKE, boundary compared. No passive currents: EFIT's are not in the data."""
    sys.path.insert(0, os.path.join(ROOT, "external", "freegsnke"))
    import time
    import freegsnke_server as fs
    d = dict(np.load(os.path.join(OUT, "mast_slices.npz")))
    R, Z = d["R"], d["Z"]
    keys = sorted({k.rsplit("_", 1)[0] for k in d if k.endswith("_info")})
    folder = os.path.join(OUT, "mast_machine")
    with_passive = len(sys.argv) > 3 and sys.argv[3] == "passive"
    s = fs.Server()
    s.build("MAST", Rmin=float(R[0]), Rmax=float(R[-1]), Zmin=float(Z[0]), Zmax=float(Z[-1]), nx=65, ny=65, paths=dict(active_coils_path=os.path.join(folder, "active.pickle"), passive_coils_path=os.path.join(folder, "passive.pickle"), limiter_path=os.path.join(folder, "limiter.pickle"), wall_path=os.path.join(folder, "wall.pickle")))
    out = []
    for key in keys[:: int(sys.argv[2]) if len(sys.argv) > 2 else 1]:
        info = d[key + "_info"]
        psi, j, lc = d[key + "_psi"].astype(float), d[key + "_j"].astype(float), d[key + "_lcfs"].astype(float)
        if not np.isfinite(info).all() or len(lc) < 10 or not np.isfinite(psi).all() or not np.isfinite(j).all():
            continue
        dA = (R[1] - R[0]) * (Z[1] - Z[0])
        Ip = abs(float((j * dA).sum()))
        if Ip < 2e5:
            continue
        t0 = time.perf_counter()
        try:
            grid, pp, ffp, left = fit_profiles(R, Z, psi, j, lc, info[7:9])
            sign = np.sign((j * dA).sum())
            s.profiles_set(Ip=Ip, fvac=info[9], psi_n=grid, pprime=sign * pp, ffprime=sign * ffp, Raxis=1.0)
            s.currents(dict(P1=info[2], P2=info[3], P3=info[4], P4=info[5], P5=info[6]))
            pf = passive_fit(s, psi, j, R, Z) if with_passive else (None, None, 0.0)
            s.guess(psi.T)                 # EFIT's flux as the start: this asks whether its solution holds under the measured coils
            r = s.forward(tol=1e-4)
            lcfs = r["lcfs"]
            Zm = r["axis"][1]
            row = dict(passive_left=pf[0], passive_rms=pf[1], passive_total=pf[2], key=key, shot=int(info[0]), t=float(info[1]), Ip=Ip, fit_left=left, wall=time.perf_counter() - t0, R_axis=float(r["axis"][0]), R_axis_efit=float(info[7]), Z_axis=float(Zm), Z_axis_efit=float(info[8]), kappa=r["kappa"], kappa_efit=float(info[10]), R_out=float(lcfs[:, 0].max()), R_out_efit=float(lc[:, 0].max()), R_in=float(lcfs[:, 0].min()), R_in_efit=float(lc[:, 0].min()), Z_top=float(lcfs[:, 1].max()), Z_top_efit=float(lc[:, 1].max()), Z_bot=float(lcfs[:, 1].min()), Z_bot_efit=float(lc[:, 1].min()), dist=float(np.median([np.min(np.hypot(lcfs[:, 0] - a, lcfs[:, 1] - b)) for a, b in lc[::4]])))
            out.append(row)
            print(key, f"passive {pf[2] / 1e3:.0f} kA; Ip {Ip / 1e3:.0f} kA, {row['wall']:.1f} s; boundary median distance {100 * row['dist']:.1f} cm; R_out {row['R_out']:.3f} ({row['R_out_efit']:.3f}), R_in {row['R_in']:.3f} ({row['R_in_efit']:.3f}), Z axis {row['Z_axis']:.3f} ({row['Z_axis_efit']:.3f}), kappa {row['kappa']:.2f} ({row['kappa_efit']:.2f})", flush=True)
        except Exception as err:
            print(key, "failed", repr(err)[:160], flush=True)
    json.dump(out, open(os.path.join(OUT, "mast_machine_check.json"), "w"), indent=0)
    if out:
        f = lambda a, b: 100 * np.median([abs(x[a] - x[b]) for x in out])
        print(f"{len(out)} slices: boundary distance median {100 * np.median([x['dist'] for x in out]):.1f} cm; |error| median, cm: R_out {f('R_out', 'R_out_efit'):.1f}, R_in {f('R_in', 'R_in_efit'):.1f}, R axis {f('R_axis', 'R_axis_efit'):.1f}, Z axis {f('Z_axis', 'Z_axis_efit'):.1f}, Z top {f('Z_top', 'Z_top_efit'):.1f}, Z bottom {f('Z_bot', 'Z_bot_efit'):.1f}; kappa {np.median([abs(x['kappa'] - x['kappa_efit']) for x in out]):.2f}", flush=True)


def track():
    """Static free-boundary solves along a shot, EFIT's plasma and the measured coils, no TORAX: the floor under the coupled run. Vessel currents fitted at the first slice and then held, or refitted at each slice (--refit). Each solve starts from the last one, as in a run, and a solve with no closed surface goes back to the last good one.
    Usage (FreeGSNKE's venv): mast_machine.py track --shots 28320 24853 [--symmetric] [--refit] [--every 2] [--start_slice 4] [--tag name]"""
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, nargs="+", required=True)
    ap.add_argument("--symmetric", action="store_true")
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--start_slice", type=int, default=4)
    ap.add_argument("--tag", default="track")
    ap.add_argument("--vessel", choices=["held", "circuit"], default="held")
    ap.add_argument("--comp", action="store_true")      # circuit: what the fit at the first slice wants beyond the circuit's currents stays, as a fixed set: what the machine's description gets wrong
    ap.add_argument("--no_fit", action="store_true")    # circuit: no fit to EFIT's flux at all
    ap.add_argument("--resist", type=float, default=1.0)
    ap.add_argument("--radial", choices=["none", "efit", "hold"], default="none")      # the outer edge held by a far coil, at EFIT's edge at each slice, or at its edge at the first slice
    args = ap.parse_args(sys.argv[2:])
    sys.path.insert(0, os.path.join(ROOT, "external", "freegsnke"))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import freegsnke_server as fs
    import vessel as vessel_lib
    shots_dir = os.path.join(ROOT, "runs", "twin", "mast_natural2", "shots")
    coils_dir = os.path.join(ROOT, "runs", "twin", "mast_natural2", "coils")
    folder = os.path.join(OUT, "mast_machine2")
    s = None
    out = []
    for shot in args.shots:
        g = dict(np.load(os.path.join(shots_dir, f"{shot}.npz"), allow_pickle=False))
        c = np.load(os.path.join(coils_dir, f"{shot}.npz"))
        R, Z = g["efit_R"].astype(float), g["efit_z"].astype(float)
        if s is None:
            s = fs.Server()
            s.build("MAST", Rmin=float(R[0]), Rmax=float(R[-1]), Zmin=float(Z[0]), Zmax=float(Z[-1]), nx=len(R), ny=len(Z), paths=dict(active_coils_path=os.path.join(folder, "active.pickle"), passive_coils_path=os.path.join(folder, "passive.pickle"), limiter_path=os.path.join(folder, "limiter.pickle"), wall_path=os.path.join(folder, "wall.pickle")))
        s.currents(dict(BV=0.0))
        dA = (R[1] - R[0]) * (Z[1] - Z[0])
        rows, rejected, a0 = [], 0, None
        t0_wall = time.perf_counter()
        coils_at = lambda t: {name: float(np.interp(t, c["t"], c[name])) for name in ("P1", "P2", "P3", "P4", "P5")}
        j_all, t_eq = np.nan_to_num(c["j_tor"]).astype(float), c["t_eq"].astype(float)

        def j_at(t):
            i = int(np.clip(np.searchsorted(t_eq, t), 1, len(t_eq) - 1))
            w = float(np.clip((t - t_eq[i - 1]) / max(t_eq[i] - t_eq[i - 1], 1e-9), 0.0, 1.0))
            return (1 - w) * j_all[:, :, i - 1] + w * j_all[:, :, i]

        ves, comp, t_ves = None, 0.0, None
        if args.vessel == "circuit":
            ves = vessel_lib.Vessel(s.eq.tokamak, R, Z, factor=args.resist)
            if shot == args.shots[0]:
                times = ves.times()
                print("vessel L/R, ms:", np.round(1e3 * times[:6], 1), "toroidal resistance, all elements in parallel, micro-ohm", round(1e6 / (1.0 / ves.R_p).sum(), 1), flush=True)
            t_ves = float(g["t"][args.start_slice])
            ves.run_up(c["t"], {name: c[name] for name in ("P1", "P2", "P3", "P4", "P5")}, t_eq, j_all, t_ves)
        for n, k in enumerate(range(args.start_slice, len(g["t"]), args.every)):
            t = float(g["t"][k])
            psi = g["efit_psi"][:, :, k].astype(float)
            j = c["j_tor"][:, :, int(np.argmin(np.abs(c["t_eq"] - t)))].astype(float)
            lc = np.c_[g["lcfs_r"][:, k], g["lcfs_z"][:, k]]
            lc = lc[np.isfinite(lc).all(axis=1)]
            if len(lc) < 10 or not np.isfinite(psi).all() or not np.isfinite(j).all():
                continue
            Ip = abs(float((j * dA).sum()))
            grid, pp, ffp, left = fit_profiles(R, Z, psi, j, lc, (float(g["R_mag"][k]), float(g["Z_mag"][k])))
            sign = np.sign((j * dA).sum())
            s.profiles_set(Ip=Ip, fvac=float(g["bvac_rmag"][k] * g["R_mag"][k]), psi_n=grid, pprime=sign * pp, ffprime=sign * ffp, Raxis=1.0)
            s.currents(coils_at(t))
            if ves is not None and t > t_ves:
                ves.step(t - t_ves, coils_at(t), j_at(t))
                t_ves = t
            if n == 0 and ves is not None and not args.no_fit:
                pf = passive_fit(s, psi, j, R, Z)
                fit = ves.now()
                comp = fit - ves.I if args.comp else 0.0
                print(shot, f"t {t:.3f}: fit leaves {pf[0]:.2f}, vessel current fitted {fit.sum() / 1e3:.0f} kA, from the circuit {ves.I.sum() / 1e3:.0f} kA, correlation by element {np.corrcoef(fit, ves.I)[0, 1]:.2f}", flush=True)
            elif (n == 0 or args.refit) and ves is None:
                passive_fit(s, psi, j, R, Z)
            if ves is not None:
                ves.apply(comp)
            if n == 0:
                s.guess(psi.T)
            else:
                s.save()
            if n == 0 or args.radial == "none":
                r = s.forward(tol=1e-3, max_iter=30, symmetric=args.symmetric)
            else:
                r = s.inverse(R=[float(lc[:, 0].max()) if args.radial == "efit" else rows[0]["R_out_efit"], rows[-1]["R_in"]], Z=[0.0, 0.0], tol=1e-3, max_iter=40, l2_reg=1e-10, frozen=("P1", "P2", "P3", "P4", "P5"), symmetric=args.symmetric)
                r["feedback"] = r["currents_active"]["BV"]
            lcfs = r["lcfs"]
            ok = lcfs is not None and r["axis"] is not None
            if ok:
                R_out, R_in = float(lcfs[:, 0].max()), float(lcfs[:, 0].min())
                a = 0.5 * (R_out - R_in)
                a0 = a if a0 is None else a0
                ok = a > 0.5 * a0 and abs(float(r["axis"][1])) < 0.5
            if not ok:
                rejected += 1
                if n:
                    s.back()
                continue
            kappa = (float(lcfs[:, 1].max()) - float(lcfs[:, 1].min())) / (2 * a)
            rows.append(dict(t=t, R_out=R_out, R_in=R_in, kappa=kappa, Z_axis=float(r["axis"][1]), R_axis=float(r["axis"][0]), R_out_efit=float(lc[:, 0].max()), R_in_efit=float(lc[:, 0].min()), kappa_efit=float((lc[:, 1].max() - lc[:, 1].min()) / (lc[:, 0].max() - lc[:, 0].min())), Z_axis_efit=float(g["Z_mag"][k]), R_axis_efit=float(g["R_mag"][k]), Ip=Ip, feedback=r.get("feedback", 0.0), q95=float(np.interp(0.95, r["pn"], r["q"])) if r["q"] is not None else float("nan"), q95_efit=float(g["q95"][k]), delta=0.5 * ((0.5 * (R_out + R_in) - float(lcfs[np.argmax(lcfs[:, 1]), 0])) + (0.5 * (R_out + R_in) - float(lcfs[np.argmin(lcfs[:, 1]), 0]))) / a, delta_efit=0.5 * float(g["delta_u"][k] + g["delta_l"][k])))
        out.append(dict(shot=shot, rejected=rejected, wall=time.perf_counter() - t0_wall, rows=rows))
        print(shot, f"{len(rows)} solved, {rejected} rejected, {out[-1]['wall']:.0f} s", " ".join(f"{100 * (x['R_out'] - x['R_out_efit']):+.0f}" for x in rows[:: max(1, len(rows) // 12)]), "(R_out minus EFIT, cm)", flush=True)
        json.dump(out, open(os.path.join(OUT, f"mast_{args.tag}.json"), "w"))
    leads = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    for key in ("R_out", "R_in", "kappa", "delta", "q95", "Z_axis", "R_axis"):
        run, held = {L: [] for L in leads}, {L: [] for L in leads}
        for x in out:
            rows = x["rows"]
            if not rows:
                continue
            tt = np.array([a["t"] for a in rows])
            for L in leads:
                i = int(np.argmin(np.abs(tt - tt[0] - L)))
                if abs(tt[i] - tt[0] - L) < 0.006:
                    run[L].append(abs(rows[i][key] - rows[i][key + "_efit"]))
                    held[L].append(abs(rows[0][key + "_efit"] - rows[i][key + "_efit"]))
        scale = 1.0 if key in ("kappa", "delta", "q95") else 100.0
        print(f"  {key:7s} |error| at {[int(1e3 * L) for L in leads]} ms: run " + ", ".join(f"{scale * np.median(run[L]):.1f}" if run[L] else "-" for L in leads) + "; held " + ", ".join(f"{scale * np.median(held[L]):.1f}" if held[L] else "-" for L in leads) + f"; shots {[len(run[L]) for L in leads]}", flush=True)
    print("rejected", sum(x["rejected"] for x in out), "solved", sum(len(x["rows"]) for x in out), flush=True)


def main():
    if sys.argv[1] == "track":
        track()
        return
    if sys.argv[1] == "fit":
        fit()
    elif sys.argv[1] == "scan":
        scan()
    elif sys.argv[1] == "build":
        build()
    elif sys.argv[1] == "extract":
        extract()
    elif sys.argv[1] == "check":
        check()


if __name__ == "__main__":
    main()
