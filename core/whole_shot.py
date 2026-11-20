"""The whole shot from its first frame (H82, docs/digital-twin.md): hands-off runs of the fourth fresh set (tags w82a to w82c) scored against what was
measured, as error against time since the run's start, for every effect the chip project's surrogates forecast (surrogates/effects_v1) and the rest.

Continuous effects, at every 5 ms: the twin's value against the measured one at the same time, and holding the first frame's measured value:
  plasma current        Ip_now                  against the store's plasma current (summary, 250 us)            relative
  line density          n_line                  against Thomson's profile averaged over the minor radius           relative
  core temperature      T_e0                    against Thomson's T_e at the axis (the shot file's fit)          relative
  stored energy         W_thermal + beam_W_fast against EFIT's whmd                                              relative
  thermal energy        W_thermal               against Thomson's kinetic thermal energy (W_th)                  relative
  q95                   q95                     against EFIT's                                                   relative
  elongation            the free boundary's     against EFIT's                                                   absolute
  triangularity         the free boundary's     against EFIT's mean of upper and lower                           absolute
  minor radius          the free boundary's     against EFIT's                                                   relative
  axis radius           the free boundary's     against EFIT's                                                   cm
  vertical position     the free boundary's     against EFIT's axis height                                       cm
  radiated power        P_rad                   against the store's radiated power                               relative
  neutrons              beam_neutrons           against the store's neutron rate                                 relative
Events, against the catalog: ELMs and sawtooth crashes as counts against time; islands (an n = 1 mode on) and H-mode as the share of time on; how the
shot ends. Profiles: T_e and n_e against Thomson at the backtest's leads (the run's own record), beside holding.
Usage: python -I core/whole_shot.py [--tags w82a,w82b,w82c] [--name h82]
       python -I core/whole_shot.py --compare h82,b1,v7      (named summaries side by side, and h83_compare.png)
Writes runs/twin/whole82/summary.json (summary_<name>.json for another name) and runs/twin/whole82/figures/h82_*.png (h83_<name>_*.png).
Measured frames whose Thomson points stop short of rho 0.9 are left out of what Thomson gives (line density, thermal energy, profiles): the fit runs on unchecked past its last point (H83).
Profiles come from the runs' saved profiles where every run has them (relative RMS over 0.1 < rho < 0.9, as shot.score), otherwise from the run's own score.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.dont_write_bytecode = True
ROOT = os.path.abspath(os.environ.get("XTORAX_HOME") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the workspace: runs/twin and external/ (README, "Where things live")
NAT = os.path.join(ROOT, "runs", "twin", "mast_natural2")
EXTRAS = os.path.join(ROOT, "runs", "twin", "mast_natural", "extras")
REAL = os.path.join(ROOT, "runs", "twin", "whole82", "real")
FIG = os.path.join(ROOT, "runs", "twin", "whole82", "figures")
TAGS = ("w82a", "w82b", "w82c")
NAME = "h82"
COVER = 0.9          # rho_norm a Thomson slice must reach to count (mast.edge_cover)


def configure(argv):
    global TAGS, NAME
    if "--tags" in argv:
        TAGS = tuple(argv[argv.index("--tags") + 1].split(","))
    if "--name" in argv:
        NAME = argv[argv.index("--name") + 1]


def out_summary(name=None):
    name = name or NAME
    return os.path.join(ROOT, "runs", "twin", "whole82", "summary.json" if name == "h82" else f"summary_{name}.json")


def prefix():
    return "h82_" if NAME == "h82" else f"h83_{NAME}_"


def cover_of(g):
    """Per slice, the outermost rho_norm with a Thomson T_e point (as mast.edge_cover)."""
    r, T = g["ts_rho"], g["ts_te"]
    out = np.zeros(len(r))
    for i in range(len(r)):
        ok = np.isfinite(r[i]) & np.isfinite(T[i]) & (T[i] > 0)
        out[i] = float(np.max(r[i][ok])) if ok.any() else 0.0
    return out
GRID = np.arange(0.0, 0.3751, 0.005)
TWIN, REALC, HOLD, BAR = "#1f6fb2", "#222222", "#8c8c8c", "#c0392b"


def load_runs():
    """Runs of TAGS in order; a later tag's run of the same shot and seed replaces an earlier one (a rerun after a fix, as v9fix replaces v9dc's 18886, H86)."""
    runs = {}
    for tag in TAGS:
        path = os.path.join(NAT, tag, "results.jsonl")
        if not os.path.exists(path):
            continue
        for line in open(path):
            r = json.loads(line)
            r["tag"] = tag
            f = os.path.join(NAT, tag, f"{r['shot']}_{r['seed']}.npz")
            if os.path.exists(f):
                r["rows"] = dict(np.load(f))
                runs[(r["shot"], r["seed"])] = r
    return list(runs.values())


def real_series(shot):
    """The measured side, each as (t, value)."""
    g = dict(np.load(os.path.join(NAT, "shots", f"{shot}.npz"), allow_pickle=False))
    e = dict(np.load(os.path.join(EXTRAS, f"{shot}.npz")))
    q = dict(np.load(os.path.join(REAL, f"{shot}.npz")))
    t = g["t"]
    lc = g["lcfs_r"]
    none = (np.array([]), np.array([]))
    ex = lambda k: (e["t_sum"], np.asarray(e[k], float)) if k in e else none        # a signal the store lacks for this shot is left out, not guessed
    # the energy EFIT's own beta_pol gives, W = (3/4) V beta_p mu0 Ip^2 / L^2, with EFIT's volume and boundary length from the shot file: on these shots whmd is 1.5 to 3.2 times it
    b = dict(np.load(os.path.join(REAL, f"{shot}_beta.npz")))
    bp = np.interp(t, b["t_eq"], b["beta_pol"], left=np.nan, right=np.nan)
    L = np.array([perimeter(g["lcfs_r"][:, k], g["lcfs_z"][:, k]) for k in range(len(t))])
    W_bp = 0.75 * g["V_rho"][:, -1] * bp * 4e-7 * np.pi * g["Ip"] ** 2 / L ** 2
    bad = cover_of(g) < COVER
    masked = lambda v: np.where(bad, np.nan, v)
    return {
        "plasma current": (e["t_sum"], np.abs(e["ip"])) if "ip" in e else none,
        "line density": (t, masked(np.nanmean(g["n_e"], axis=1))),          # from Thomson's fitted profile, averaged over the minor radius as TORAX averages the twin's: the chip project's density is the interferometer's line integral, which a synthetic chord through the twin would read
        "thermal energy": (t, masked(g["W_th"])),
        "core temperature": (t, g["T_e"][:, 0]),
        "stored energy": (q["t_eq"], q["whmd"]),
        "pressure energy": (t, W_bp),
        "q95": (t, g["q95"]),
        "elongation": (t, g["kappa"]),
        "triangularity": (t, 0.5 * (g["delta_u"] + g["delta_l"])),
        "minor radius": (t, g["a"]),
        "axis radius": (t, g["R_mag"]),
        "outer edge": (t, np.nanmax(lc, axis=0)),
        "vertical position": (t, g["Z_mag"]),
        "radiated power": ex("power_radiated"),
        "neutrons": ex("neutron_rates_total"),
    }


def has_whmd(r):
    """Whether the run's free-boundary log carries the synthetic whmd (freeboundary.synthetic_whmd, H84)."""
    return any(x.get("whmd") is not None for x in (r.get("free", {}).get("log") or []))


def held(r, k):
    """Effects read from the free boundary's solves are held between them, as the run holds them."""
    return k in FREE or (k == "stored energy" and has_whmd(r))



def bolo(w):
    """radiation.combine over a run's recorded terms, step by step."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import radiation
    n = len(w["t"])
    keys = ("rad_core", "rad_sol_c", "rad_h_main", "rad_h_div")
    return np.array([radiation.combine({k: (w[k][i] if k in w else 0.0) for k in keys}) for i in range(n)])

def twin_series(r):
    w = r["rows"]
    t = w["t"]
    log = r.get("free", {}).get("log") or []
    tf = np.array([x["t"] for x in log]) if log else np.array([])
    fl = lambda k: np.array([x.get(k, np.nan) if x.get(k) is not None else np.nan for x in log], float)
    get = lambda k: w[k] if k in w else np.full_like(t, np.nan)
    return {
        "plasma current": (t, np.abs(get("Ip_now"))),
        "line density": (t, get("n_line")),
        "core temperature": (t, get("T_e0")),
        "stored energy": (tf, fl("whmd")) if has_whmd(r) else (t, get("W_thermal") + np.nan_to_num(get("beam_W_fast"))),      # the twin's own whmd reading where the run has it (H84), else thermal plus fast as before
        "pressure energy": (t, get("W_thermal") + np.nan_to_num(get("beam_W_fast"))),
        "thermal energy": (t, get("W_thermal")),
        "q95": (t, get("q95")),
        "elongation": (tf, fl("kappa")),
        "triangularity": (tf, fl("delta")),
        "minor radius": (tf, fl("a")),
        "axis radius": (tf, fl("R_axis")),
        "outer edge": (tf, fl("R_out")),
        "vertical position": (tf, fl("Z_axis")),
        "radiated power": (t, bolo(w)) if "rad_core" in w else (t, np.abs(get("P_rad"))),       # the bolometers' reading from the radiation budget where the run has its terms (radiation.combine, H86), else the core's line radiation
        "neutrons": (t, get("beam_neutrons")),
    }


# how each effect's error is measured: relative (share), absolute (in its unit), or cm
KIND = {"plasma current": "rel", "line density": "rel", "core temperature": "rel", "stored energy": "rel", "pressure energy": "rel", "thermal energy": "rel", "q95": "rel", "elongation": "abs", "triangularity": "abs",
        "minor radius": "rel", "axis radius": "cm", "outer edge": "cm", "vertical position": "cm", "radiated power": "rel", "neutrons": "rel"}
SURROGATE = {"plasma current", "line density", "core temperature", "stored energy", "q95", "elongation", "triangularity", "minor radius", "axis radius", "vertical position", "radiated power", "neutrons"}
FREE = {"elongation", "triangularity", "minor radius", "axis radius", "outer edge", "vertical position"}       # from the free boundary's solves: held between them, as in the run
NOTE = {"plasma current": "given: the machine's programme", "line density": "steered by the density programme", "stored energy": "whmd: a boundary B_p^2 integral, no pressure; the twin's own reading where logged",
        "pressure energy": "EFIT's W_MHD, from beta_pol", "thermal energy": "kinetic, from Thomson and CX", "radiated power": "bolometers; the twin's is core only"}
# "stored energy" is the store's whmd, the chip project's target: EFIT++'s diamagnetic energy computed with the measured diamagnetic flux at zero, so a boundary integral of the
# poloidal field (3/(8 mu0) times the integral of B_p^2 [(R - 1 m) n_R + Z n_Z] dS) that carries no pressure (docs/lit/36). EFIT's pressure energy is "pressure energy" (H83).


def perimeter(r, z):
    """The length of a boundary given as points (NaN padding dropped)."""
    ok = np.isfinite(r) & np.isfinite(z) & (r > 0)
    if ok.sum() < 10:
        return np.nan
    rc, zc = np.r_[r[ok], r[ok][0]], np.r_[z[ok], z[ok][0]]
    return float(np.sum(np.hypot(np.diff(rc), np.diff(zc))))


def at(series, tt, until=None):
    """The series at times tt: linear between samples, or, with until (the free boundary's log), the last accepted solve held to that time, as the twin itself holds it."""
    t, v = series
    ok = np.isfinite(v)
    if until is not None:
        if ok.sum() < 1:
            return np.full_like(tt, np.nan)
        t, v = t[ok], v[ok]
        j = np.searchsorted(t, tt, side="right") - 1
        out = np.where(j >= 0, v[np.clip(j, 0, len(v) - 1)], np.nan)
        out[tt > until + 1e-9] = np.nan
        return out
    if len(t) < 2 or ok.sum() < 2:
        return np.full_like(tt, np.nan)
    return np.interp(tt, t[ok], v[ok], left=np.nan, right=np.nan)


def err(kind, twin, real):
    if kind == "rel":
        return np.abs(twin - real) / np.maximum(np.abs(real), 1e-12)
    if kind == "cm":
        return 100 * np.abs(twin - real)
    return np.abs(twin - real)


def events_real(cat, shot):
    x = cat[str(shot)]
    modes = [m for m in (x.get("modes") or []) if m.get("n") == 1 and m.get("t_on") is not None]
    return dict(elms=np.array((x.get("elms") or {}).get("t", []), float), crashes=np.array((x.get("sawteeth") or {}).get("crash_t", []), float),
                modes=[(m["t_on"], m.get("t_end") or m["t_on"]) for m in modes], locks=[m.get("t_lock") for m in modes if m.get("locked")],
                hmode=[(h["t0"], h["t1"]) for h in (x.get("hmode_level_candidates") or [])], end=(x["phases"].get("quench") or {}).get("kind"))


MODE_READING = "island_or_kink"     # how the twin's n = 1 mode time is read (H86); "island" gives the reading before H86

def events_twin(r):
    w = r["rows"]
    t = w["t"]
    crashes = t[np.nan_to_num(w["sawtooth"]) > 0] if "sawtooth" in w else np.array([])
    wid = np.nan_to_num(w["w_2_1"]) if "w_2_1" in w else np.zeros_like(t)
    H = np.nan_to_num(w["H"]) if "H" in w else np.zeros_like(t)
    # an n = 1 mode on: the twin's 2/1 island, or its n = 1 internal mode read from q_min (coupled.KINK_DQ, H86): MAST's catalog counts every n = 1 mode, and on
    # these shots 66% of its mode time has EFIT's q_min under 1.1 against 19% of the rest. Runs before H86 lack the reading; it is rebuilt from their q_min and beam power
    if "kink_n1" in w:
        kink = np.nan_to_num(w["kink_n1"]) > 0.5
    elif "q_min" in w:
        kink = (np.nan_to_num(w["q_min"], nan=9.0) <= 1.12) & (np.nan_to_num(w["beam_P_abs"]) > 1e5 if "beam_P_abs" in w else False)
    else:
        kink = np.zeros_like(t, bool)
    on = (wid > 0.005) | (kink if MODE_READING == "island_or_kink" else False)
    return dict(elms=np.array(r.get("elm_times") or [], float), crashes=crashes, mode_on=(t, on), H=(t, H > 0.5), disrupted=bool(r.get("disrupted")))


def share_on(intervals, tt):
    on = np.zeros_like(tt, bool)
    for a, b in intervals:
        on |= (tt >= a) & (tt <= b)
    return on


def main():
    runs = load_runs()
    cat = json.load(open(os.path.join(ROOT, "runs", "twin", "mast_catalog", "catalog.json")))["shots"]
    print(f"{len(runs)} runs on {len({r['shot'] for r in runs})} shots", flush=True)
    names = list(KIND)
    E = {k: [] for k in names}
    Hd = {k: [] for k in names}
    S = {k: [] for k in names}
    alive = dict(real=[], twin=[])
    ev = dict(elms_twin=[], elms_real=[], crash_twin=[], crash_real=[], mode_twin=[], mode_real=[], H_twin=[], H_real=[])
    ends = []
    per_run = []
    reals = {}
    for r in runs:
        shot, t0, t1 = r["shot"], r["t_cut"], r["t_end"]
        if shot not in reals:
            reals[shot] = real_series(shot)
        rs, ts = reals[shot], twin_series(r)
        tt = t0 + GRID
        t_last = float(r["rows"]["t"][-1])
        live = (tt <= t1 + 1e-9) & (tt <= t_last + 1e-9)          # both still going: the real shot measured, the twin not stopped
        row = dict(shot=shot, seed=r["seed"], t_cut=t0, t_end=t1, t_last=t_last)
        for k in names:
            rv, tv = at(rs[k], tt), at(ts[k], tt, until=t_last if held(r, k) else None)
            hv = np.full_like(tt, at(rs[k], np.array([t0]))[0])
            e, h = err(KIND[k], tv, rv), err(KIND[k], hv, rv)
            s = (tv - rv) / np.maximum(np.abs(rv), 1e-12) if KIND[k] == "rel" else (100 if KIND[k] == "cm" else 1) * (tv - rv)
            e[~live] = np.nan
            h[~np.isfinite(e)] = np.nan          # holding is scored on the same times as the twin
            s[~np.isfinite(e)] = np.nan
            E[k].append(e)
            Hd[k].append(h)
            S[k].append(s)
            row[k] = float(np.nanmedian(e)) if np.isfinite(e).any() else None
            row[k + " held"] = float(np.nanmedian(h)) if np.isfinite(h).any() else None
            row[k + " signed"] = float(np.nanmedian(s)) if np.isfinite(s).any() else None
        er, et = events_real(cat, shot), events_twin(r)
        dt = GRID[1] - GRID[0]
        binned = lambda times: np.array([np.sum((times >= x) & (times < x + dt)) if l else np.nan for x, l in zip(tt, live)], float)      # events in each 5 ms bin while both run
        ev["elms_twin"].append(binned(et["elms"]))
        ev["elms_real"].append(binned(er["elms"]))
        ev["crash_twin"].append(binned(et["crashes"]))
        ev["crash_real"].append(binned(er["crashes"]))
        mt = np.interp(tt, et["mode_on"][0], et["mode_on"][1].astype(float), left=np.nan, right=np.nan)
        mr = share_on(er["modes"], tt).astype(float)
        Ht = np.interp(tt, et["H"][0], et["H"][1].astype(float), left=np.nan, right=np.nan)
        Hr = share_on(er["hmode"], tt).astype(float)
        for key, v in (("mode_twin", mt), ("mode_real", mr), ("H_twin", Ht), ("H_real", Hr)):
            v = v.copy()
            v[~live] = np.nan
            ev[key].append(v)
        t_both = min(t1, t_last)
        inside = lambda times: int(np.sum((times >= t0) & (times <= t_both)))
        ends.append(dict(shot=shot, seed=r["seed"], real_end=er["end"], twin_disrupted=et["disrupted"], twin_stop=r.get("sim_error"), t_stop=t_last, t_end=t1, early_ms=round(1e3 * (t1 - t_last), 1)))
        row.update(elms=(inside(et["elms"]), inside(er["elms"])), crashes=(inside(et["crashes"]), inside(er["crashes"])),
                   island_share=(float(np.nanmean(mt[live])) if live.any() else None, float(np.nanmean(mr[live])) if live.any() else None),
                   H_share=(float(np.nanmean(Ht[live])) if live.any() else None, float(np.nanmean(Hr[live])) if live.any() else None), real_end=er["end"], twin_disrupted=et["disrupted"])
        alive["real"].append((tt <= t1 + 1e-9).astype(float))
        alive["twin"].append(((tt <= t_last + 1e-9) & (tt <= t1 + 1e-9)).astype(float))
        per_run.append(row)
    stack = lambda L: np.vstack(L) if L else np.zeros((0, len(GRID)))
    summ = dict(grid_s=GRID.tolist(), runs=len(runs), shots=sorted({r["shot"] for r in runs}), effects={}, events={}, ends=ends, per_run=per_run)
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in names:
            A, B, C = stack(E[k]), stack(Hd[k]), stack(S[k])
            n = np.sum(np.isfinite(A), axis=0)
            three = lambda X, f: np.where(n >= 3, f(X), np.nan).tolist() if len(X) else []
            pts = np.isfinite(A) & np.isfinite(B)
            summ["effects"][k] = dict(kind=KIND[k], surrogate=k in SURROGATE, note=NOTE.get(k, ""), n=n.tolist(), median=three(A, lambda X: np.nanmedian(X, axis=0)),
                                      q25=three(A, lambda X: np.nanpercentile(X, 25, axis=0)), q75=three(A, lambda X: np.nanpercentile(X, 75, axis=0)),
                                      held=three(B, lambda X: np.nanmedian(X, axis=0)), signed=three(C, lambda X: np.nanmedian(X, axis=0)),
                                      twin_better=float(np.mean(A[pts] < B[pts])) if pts.any() else None,          # the share of run-times where the twin is nearer than holding
                                      within_bar=float(np.mean(A[np.isfinite(A)] <= 0.10)) if KIND[k] == "rel" and np.isfinite(A).any() else None,
                                      held_within_bar=float(np.mean(B[pts] <= 0.10)) if KIND[k] == "rel" and pts.any() else None)
        win = np.ones(5)          # 25 ms, centred
        smooth = lambda v: np.convolve(np.nan_to_num(v), win, mode="same")
        for key in ev:
            A = stack(ev[key])
            m = np.isfinite(A)
            if key.startswith(("elms", "crash")):          # a rate, per second, over the runs going in that window
                val = smooth(np.nansum(A, axis=0)) / np.maximum(smooth(m.sum(axis=0)) * (GRID[1] - GRID[0]), 1e-12)
            else:                                           # a share of time, over the runs going in that window
                val = smooth(np.nansum(A, axis=0)) / np.maximum(smooth(m.sum(axis=0)), 1e-12)
            summ["events"][key] = dict(value=np.where(m.sum(axis=0) >= 3, val, np.nan).tolist(), n=m.sum(axis=0).tolist())
        summ["alive"] = {a: stack(b).sum(axis=0).tolist() for a, b in alive.items()}
    os.makedirs(os.path.join(ROOT, "runs", "twin", "whole82"), exist_ok=True)
    summ["name"], summ["tags"] = NAME, list(TAGS)
    summ["profiles"] = profiles(runs)
    json.dump(summ, open(out_summary(), "w"), indent=0, default=float)
    table(summ)
    figures(summ, runs, reals, cat)


LEADS_S = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]      # mast.FIXED["leads_s"]: the backtest's profile score kept with each run


def saved_profiles(runs):
    """Per run and channel, the error at each lead from the run's saved profiles (natural.py save_profiles): relative RMS over 0.1 < rho < 0.9 against the
    measured fits averaged with their neighbours (mast.scored_shot), as shot.score does, beside holding the start (the trailing fits at the cut); times whose nearest
    measured slice has Thomson stopping short of COVER are left out."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mast
    import shot as shot_lib
    res = {}
    for r in runs:
        f = os.path.join(NAT, r["tag"], f"{r['shot']}_{r['seed']}_prof.npz")
        if not os.path.exists(f):
            return None
        pr = dict(np.load(f))
        path = os.path.join(NAT, "shots", f"{r['shot']}.npz")
        truth = shot_lib.Shot.load(path)
        g = dict(np.load(path, allow_pickle=False))
        scored = mast.scored_shot(truth)
        k = int(r["k"])
        mast.EDGE_COVER = COVER if r.get("overrides", {}).get("edge_start") else None
        known = mast.known_shot(truth, k, True, cover=cover_of(g) if mast.EDGE_COVER is not None else None)
        cov = cover_of(g)
        t0, t_last = r["t_cut"], float(pr["t"][-1])
        band = (pr["rho"] > 0.1) & (pr["rho"] < 0.9)
        rows = {}
        for c in ("T_e", "T_i", "n_e"):
            if getattr(truth, c) is None or c not in pr:
                continue
            e_tw, e_hd = [], []
            for lead in LEADS_S:
                t = t0 + lead
                j = int(np.argmin(np.abs(truth.t - t)))
                if t > r["t_end"] + 1e-9 or t > t_last + 1e-9 or cov[j] < COVER:
                    e_tw.append(np.nan)
                    e_hd.append(np.nan)
                    continue
                i = int(np.argmin(np.abs(pr["t"] - t)))
                true = np.interp(pr["rho"][band], truth.rho, scored.at(c, t))
                sim = pr[c][i][band]
                held = np.interp(pr["rho"][band], truth.rho, known.at(c, t0))
                e_tw.append(float(np.sqrt(np.mean((sim - true) ** 2)) / np.mean(true)))
                e_hd.append(float(np.sqrt(np.mean((held - true) ** 2)) / np.mean(true)))
            rows[c] = (e_tw, e_hd)
        res[(r["shot"], r["seed"], r["tag"])] = rows
    return res


def profiles(runs):
    """T_e, T_i, n_e profiles against Thomson and CX: relative RMS over the minor radius at each lead, beside holding the first frame's profiles: from saved profiles where
    every run has them, else the run's own score (mast.summarize)."""
    out = {}
    saved = saved_profiles(runs)
    if saved is not None:
        for c in ("T_e", "T_i", "n_e"):
            tw = np.array([v[c][0] for v in saved.values() if c in v], float).reshape(-1, len(LEADS_S))
            hd = np.array([v[c][1] for v in saved.values() if c in v], float).reshape(-1, len(LEADS_S))
            with np.errstate(all="ignore"):
                import warnings
                warnings.simplefilter("ignore", RuntimeWarning)
                n = np.sum(np.isfinite(tw), axis=0)
                out[c] = dict(n=n.tolist(), median=np.where(n >= 3, np.nanmedian(tw, axis=0), np.nan).tolist(), q25=np.where(n >= 3, np.nanpercentile(tw, 25, axis=0), np.nan).tolist(),
                              q75=np.where(n >= 3, np.nanpercentile(tw, 75, axis=0), np.nan).tolist(), held=np.where(n >= 3, np.nanmedian(hd, axis=0), np.nan).tolist(), source="saved")
        return draw_profiles(out, runs)
    for c in ("T_e", "T_i", "n_e"):
        both = [r for r in runs if r.get("profile") and r.get("held") and c in r["profile"]["err_at_s"] and c in r["held"]["err_at_s"]]      # T_i only where CX measured it
        tw = np.array([[np.nan if v is None else v for v in r["profile"]["err_at_s"][c]] for r in both], float).reshape(-1, len(LEADS_S))
        hd = np.array([[np.nan if v is None else v for v in r["held"]["err_at_s"][c]] for r in both], float).reshape(-1, len(LEADS_S))
        with np.errstate(all="ignore"):
            import warnings
            warnings.simplefilter("ignore", RuntimeWarning)
            n = np.sum(np.isfinite(tw), axis=0)
            out[c] = dict(n=n.tolist(), median=np.where(n >= 3, np.nanmedian(tw, axis=0), np.nan).tolist(), q25=np.where(n >= 3, np.nanpercentile(tw, 25, axis=0), np.nan).tolist(),
                          q75=np.where(n >= 3, np.nanpercentile(tw, 75, axis=0), np.nan).tolist(), held=np.where(n >= 3, np.nanmedian(hd, axis=0), np.nan).tolist(), source="run score")
    return draw_profiles(out, runs)


def draw_profiles(out, runs):
    os.makedirs(FIG, exist_ok=True)
    print("profiles, relative RMS over the minor radius at leads " + " ".join(f"{1e3 * v:.0f}" for v in LEADS_S) + " ms (twin | holding)")
    for c, e in out.items():
        f = lambda v: f"{100 * v:4.0f}%" if np.isfinite(v) else "   - "
        print(f"  {c:4s} " + " ".join(f(v) for v in e["median"]) + "  |  " + " ".join(f(v) for v in e["held"]) + "   runs " + " ".join(str(v) for v in e["n"]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    x = 1e3 * np.array(LEADS_S)
    for ax, (c, e) in zip(axes, out.items()):
        med, q25, q75, held = (100 * np.array(e[k], float) for k in ("median", "q25", "q75", "held"))
        ax.fill_between(x, q25, q75, color=TWIN, alpha=0.15, lw=0)
        ax.plot(x, med, color=TWIN, lw=2, marker="o", ms=4, label="the twin, hands-off (median, middle half shaded)")
        ax.plot(x, held, color=HOLD, lw=1.5, ls="--", marker="o", ms=4, label="holding the first frame")
        ax.axhline(10, color=BAR, lw=1, ls=":", label="the 10% bar")
        ax.set_title({"T_e": "electron temperature profile (Thomson)", "T_i": "ion temperature profile (CX)", "n_e": "electron density profile (Thomson)"}[c], fontsize=10, loc="left")
        ax.set_xlabel("ms since the start")
        ax.set_ylabel("relative RMS error, %")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"MAST, the whole shot from its first frame: profiles against time ({NAME}, {len(runs)} runs)", fontsize=11)
    fig.savefig(os.path.join(FIG, prefix() + "profiles.png"), dpi=110)
    plt.close(fig)
    return out


def at_ms(summ, k, ms, field="median"):
    g = np.array(summ["grid_s"])
    j = int(np.argmin(np.abs(g - ms / 1e3)))
    v = summ["effects"][k][field][j]
    return v


def table(summ):
    print("effect: median error at 10, 50, 100, 150, 200 ms (twin | holding the first frame, same run-times); signed at 100 ms; share of run-times the twin is nearer; within 10% (twin, holding); runs at 100 ms")
    for k, e in summ["effects"].items():
        f = (lambda v: f"{100 * v:5.1f}%") if e["kind"] == "rel" else ((lambda v: f"{v:5.2f}") if e["kind"] == "abs" else (lambda v: f"{v:5.1f}cm"))
        pick = [10, 50, 100, 150, 200]
        tw = [at_ms(summ, k, m) for m in pick]
        hd = [at_ms(summ, k, m, "held") for m in pick]
        n100 = summ["effects"][k]["n"][int(np.argmin(np.abs(np.array(summ["grid_s"]) - 0.1)))]
        fmt = lambda v: f(v) if v is not None and np.isfinite(v) else "   - "
        pc = lambda v: f"{100 * v:3.0f}%" if v is not None else "  - "
        sg = at_ms(summ, k, 100, "signed")
        print(f"  {k:18s} " + " ".join(fmt(v) for v in tw) + "  |  " + " ".join(fmt(v) for v in hd) + f"  signed {fmt(sg) if sg is None or not np.isfinite(sg) or e['kind'] != 'rel' else f'{100 * sg:+5.1f}%'}"
              + f"  nearer {pc(e['twin_better'])}  in bar {pc(e['within_bar'])} {pc(e['held_within_bar'])}   n {n100}")
    print("ends: shot seed, the real end, the twin's (disrupted or not, ms before the real end it stopped)")
    for x in summ["ends"]:
        print(f"  {x['shot']} {x['seed']}  real {x['real_end']!s:13s} twin {'disrupted' if x['twin_disrupted'] else 'ran on   '} stopped {x['early_ms']:6.1f} ms before the real end")
    print("per shot (seeds pooled): median error over the run, twin | holding; events over the time both run: twin seeds, real")
    keys = ("line density", "core temperature", "thermal energy", "stored energy", "q95", "elongation", "neutrons")
    print("  shot   " + "  ".join(f"{k[:11]:>15s}" for k in keys) + "   ELMs (twin; real)   crashes   island share   H share")
    for shot in summ["shots"]:
        rows = [x for x in summ["per_run"] if x["shot"] == shot]
        cell = []
        for k in keys:
            a = [x[k] for x in rows if x[k] is not None]
            b = [x[k + " held"] for x in rows if x[k + " held"] is not None]
            if not a:
                cell.append(f"{'-':>15s}")
                continue
            s = (lambda v: f"{100 * v:.0f}%") if KIND[k] == "rel" else (lambda v: f"{v:.2f}")
            cell.append(f"{s(np.mean(a)) + ' | ' + (s(np.mean(b)) if b else '-'):>15s}")
        ev = lambda name: f"{[x[name][0] for x in rows]}; {rows[0][name][1]}"
        sh = lambda name: f"{np.mean([x[name][0] for x in rows if x[name][0] is not None]):.2f}; {rows[0][name][1]:.2f}" if rows[0][name][1] is not None else "-"
        print(f"  {shot}  " + "  ".join(cell) + f"   {ev('elms'):18s} {ev('crashes'):10s} {sh('island_share'):12s} {sh('H_share')}")


def figures(summ, runs, reals, cat):
    os.makedirs(FIG, exist_ok=True)
    g = 1e3 * np.array(summ["grid_s"])
    names = list(summ["effects"])
    fig, axes = plt.subplots(4, 5, figsize=(18.5, 13), constrained_layout=True)
    for ax in list(axes.flat)[len(names) + 2:]:
        ax.axis("off")
    for ax, k in zip(axes.flat, names):
        e = summ["effects"][k]
        sc = 100 if e["kind"] == "rel" else 1
        med, q25, q75, held = (np.array(e[x], float) * sc for x in ("median", "q25", "q75", "held"))
        ok = np.isfinite(med)
        ax.fill_between(g, np.where(ok, q25, np.nan), np.where(ok, q75, np.nan), color=TWIN, alpha=0.15, lw=0)
        ax.plot(g, med, color=TWIN, lw=2)
        ax.plot(g, np.where(ok, held, np.nan), color=HOLD, lw=1.5, ls="--")
        if e["kind"] == "rel":
            ax.axhline(10, color=BAR, lw=1, ls=":")
        unit = "%" if e["kind"] == "rel" else ("cm" if e["kind"] == "cm" else "")
        ax.set_title(k + (" *" if e["surrogate"] else "") + (f"\n({e['note']})" if e.get("note") else ""), fontsize=9.5, loc="left")
        ax.set_ylabel(f"error, {unit}" if unit else "error, absolute")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.set_xlim(0, g[ok].max() if ok.any() else 1)
    ax = axes.flat[len(names)]
    ax.step(g, summ["alive"]["real"], where="post", color=REALC, lw=1.5)
    ax.step(g, summ["alive"]["twin"], where="post", color=TWIN, lw=2)
    ax.set_title("runs still going (black: the real shot measured;\nblue: the twin also running, not disrupted)", fontsize=9.5, loc="left")
    ax.set_ylabel("runs")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax = axes.flat[len(names) + 1]
    ax.axis("off")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Line2D([], [], color=TWIN, lw=2, label="the twin, hands-off: median over runs"), Patch(color=TWIN, alpha=0.15, label="the middle half of runs"),
                       Line2D([], [], color=HOLD, lw=1.5, ls="--", label="holding the first frame (same run-times)"), Line2D([], [], color=BAR, lw=1, ls=":", label="the 10% bar")],
              loc="upper left", fontsize=9, frameon=False)
    ax.text(0.0, 0.32, "* an effect the chip project's surrogates forecast\n(surrogates/effects_v1)\n\nA curve stops where fewer than 3 runs reach that time.\nRuns the twin ends early (its own disruption) leave\nboth curves from then on: see the panel to the left.", transform=ax.transAxes, fontsize=8.5, va="top")
    for ax in list(axes.flat)[len(names) - 5:len(names) + 1]:
        ax.set_xlabel("ms since the run's start (its first measured frame)")
    fig.suptitle(f"MAST, the whole shot from its first frame, hands-off: error against time ({summ['runs']} runs on {len(summ['shots'])} fresh shots, 2 seeds each)", fontsize=12)
    fig.savefig(os.path.join(FIG, prefix() + "errors.png"), dpi=110)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.8), constrained_layout=True)
    for ax, (title, a, b, unit) in zip(axes[0], (("ELMs per second", "elms_twin", "elms_real", "per s"), ("sawtooth crashes per second", "crash_twin", "crash_real", "per s"),
                                                 ("share of runs with an n = 1 island on", "mode_twin", "mode_real", ""), ("share of runs in H-mode", "H_twin", "H_real", ""))):
        ax.plot(g, summ["events"][a]["value"], color=TWIN, lw=2, label="the twin")
        ax.plot(g, summ["events"][b]["value"], color=REALC, lw=1.5, label="measured (the catalog)")
        ax.set_title(title + "\n(25 ms window, over the runs going)", fontsize=9.5, loc="left")
        ax.set_xlabel("ms since the start")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
    axes[0][0].legend(fontsize=8)
    shots = summ["shots"]
    x = np.arange(len(shots))
    for ax, (title, key) in zip(axes[1], (("ELMs per shot, while both run", "elms"), ("sawtooth crashes per shot, while both run", "crashes"), ("share of the time with an island on", "island_share"), ("share of the time in H-mode", "H_share"))):
        pick = lambda sh, i: [r[key][i] for r in summ["per_run"] if r["shot"] == sh and r[key][i] is not None]
        tw = [np.mean(pick(sh, 0)) if pick(sh, 0) else np.nan for sh in shots]
        re = [np.mean(pick(sh, 1)) if pick(sh, 1) else np.nan for sh in shots]
        ax.bar(x - 0.2, tw, 0.38, color=TWIN, label="the twin (mean of its seeds)")
        ax.bar(x + 0.2, re, 0.38, color=REALC, label="measured")
        ax.set_xticks(x)
        ax.set_xticklabels([str(sh) for sh in shots], rotation=60, fontsize=8)
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.grid(alpha=0.25, axis="y")
    axes[1][0].legend(fontsize=8)
    fig.suptitle("MAST, the whole shot from its first frame: events, the twin against the catalog", fontsize=12)
    fig.savefig(os.path.join(FIG, prefix() + "events.png"), dpi=110)
    plt.close(fig)
    shown = {}
    for r in runs:
        if r["seed"] != 0 or r["shot"] in shown:
            continue
        shown[r["shot"]] = True
        rs, ts = reals[r["shot"]], twin_series(r)
        er, et = events_real(cat, r["shot"]), events_twin(r)
        panels = ["plasma current", "line density", "core temperature", "stored energy", "q95", "elongation", "vertical position", "radiated power", "neutrons"]
        fig, axes = plt.subplots(2, 6, figsize=(20, 6.5), constrained_layout=True)
        t0, t1 = r["t_cut"], r["t_end"]
        for ax, k in zip(axes.flat, panels):
            sc = {"plasma current": 1e-3, "line density": 1e-19, "stored energy": 1e-3, "radiated power": 1e-6, "neutrons": 1e-12, "vertical position": 100}.get(k, 1.0)
            unit = {"plasma current": "kA", "line density": "1e19 m^-3", "core temperature": "keV", "stored energy": "kJ", "radiated power": "MW", "neutrons": "1e12 per s", "vertical position": "cm"}.get(k, "")
            tr, vr = rs[k]
            m = (tr >= t0 - 0.005) & (tr <= t1 + 0.005)
            ax.plot(1e3 * (tr[m] - t0), vr[m] * sc, color=REALC, lw=1.3, label="measured")
            tt, vt = ts[k]
            mm = np.isfinite(vt)
            ax.plot(1e3 * (tt[mm] - t0), vt[mm] * sc, color=TWIN, lw=1.6, label="the twin")
            ax.set_title(f"{k}" + (f", {unit}" if unit else ""), fontsize=9)
            ax.grid(alpha=0.25)
        ax = axes.flat[len(panels)]
        tw, wid = r["rows"]["t"], np.nan_to_num(r["rows"]["w_2_1"]) if "w_2_1" in r["rows"] else np.zeros_like(r["rows"]["t"])
        ax.plot(1e3 * (tw - t0), 100 * wid, color=TWIN, lw=1.6, label="the twin's 2/1 island, cm")
        for a, b in er["modes"]:
            ax.axvspan(1e3 * (a - t0), 1e3 * (b - t0), color=REALC, alpha=0.15)
        ax.set_title("island width, cm (grey: measured n = 1 modes)", fontsize=9)
        ax = axes.flat[len(panels) + 1]
        for x in er["elms"][(er["elms"] >= t0) & (er["elms"] <= t1)]:
            ax.axvline(1e3 * (x - t0), ymin=0.55, ymax=0.95, color=REALC, lw=0.7)
        for x in et["elms"]:
            ax.axvline(1e3 * (x - t0), ymin=0.05, ymax=0.45, color=TWIN, lw=0.7)
        ax.set_title("ELMs: measured above, the twin below", fontsize=9)
        ax.set_yticks([])
        ax = axes.flat[len(panels) + 2]
        for x in er["crashes"][(er["crashes"] >= t0) & (er["crashes"] <= t1)]:
            ax.axvline(1e3 * (x - t0), ymin=0.55, ymax=0.95, color=REALC, lw=0.7)
        for x in et["crashes"]:
            ax.axvline(1e3 * (x - t0), ymin=0.05, ymax=0.45, color=TWIN, lw=0.7)
        ax.set_title("sawtooth crashes: measured above, the twin below", fontsize=9)
        ax.set_yticks([])
        for ax in list(axes.flat)[len(panels) + 3:]:
            ax.axis("off")
        for ax in list(axes.flat)[:len(panels) + 3]:
            ax.set_xlim(-5, 1e3 * (t1 - t0) + 5)
        for ax in axes[-1]:
            ax.set_xlabel("ms since the start")
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f"MAST {r['shot']}, hands-off from {t0:.3f} s to {t1:.3f} s (seed 0); the real shot ends {er['end']}, the twin {'disrupts' if et['disrupted'] else 'does not disrupt'}", fontsize=11)
        fig.savefig(os.path.join(FIG, prefix() + f"traces_{r['shot']}.png"), dpi=100)
        plt.close(fig)


def run_errors(tags):
    """Per run, (shot, seed) -> (t_cut, {effect: error against GRID, NaN where the run or the measurement has stopped}), as main() scores them."""
    global TAGS
    keep, TAGS = TAGS, tuple(tags)
    runs = load_runs()
    TAGS = keep
    reals, out = {}, {}
    for r in runs:
        shot, t0, t1 = r["shot"], r["t_cut"], r["t_end"]
        if shot not in reals:
            reals[shot] = real_series(shot)
        rs, ts = reals[shot], twin_series(r)
        tt = t0 + GRID
        t_last = float(r["rows"]["t"][-1])
        live = (tt <= t1 + 1e-9) & (tt <= t_last + 1e-9)
        errs = {}
        for k in KIND:
            e = err(KIND[k], at(ts[k], tt, until=t_last if held(r, k) else None), at(rs[k], tt))
            e[~live] = np.nan
            errs[k] = e
        out[(shot, r["seed"])] = (round(t0, 4), errs)
    return out


def common_support(names):
    """Each later version against the first on common ground only: the same shot, seed and start, at the times both are still running and measured. Medians and
    win shares over those points, so that a run one version ends early, or a start that moved, does not change what is compared."""
    S = {n: json.load(open(out_summary(n))) for n in names}
    R = {n: run_errors(S[n]["tags"]) for n in names}
    base = names[0]
    for n in names[1:]:
        pairs = [k for k in R[base] if k in R[n] and R[base][k][0] == R[n][k][0]]
        print(f"common ground, {n} against {base}: {len(pairs)} runs with the same start (of {len(R[n])}); median error over the points both reach, share of points {n} is nearer, inside 10% ({base}; {n})")
        for k in KIND:
            a = np.concatenate([R[base][p][1][k] for p in pairs]) if pairs else np.array([])
            b = np.concatenate([R[n][p][1][k] for p in pairs]) if pairs else np.array([])
            m = np.isfinite(a) & np.isfinite(b)
            if not m.any():
                continue
            sc = 100 if KIND[k] == "rel" else 1
            unit = "%" if KIND[k] == "rel" else ("cm" if KIND[k] == "cm" else "")
            bar = f"   inside 10%: {100 * np.mean(a[m] <= 0.1):3.0f}%; {100 * np.mean(b[m] <= 0.1):3.0f}%" if KIND[k] == "rel" else ""
            print(f"  {k:18s} {sc * np.median(a[m]):6.2f}{unit} -> {sc * np.median(b[m]):6.2f}{unit}   nearer {100 * np.mean(b[m] < a[m]):3.0f}% of {m.sum():4d} points{bar}")


def compare(names):
    """Named summaries side by side: the effects at 50, 100, 150, 200 ms, the share inside the bar, events and the twin's own disruptions."""
    S = {n: json.load(open(out_summary(n))) for n in names}
    os.makedirs(FIG, exist_ok=True)
    w = 9
    print("median error at 50 / 100 / 150 / 200 ms, then inside 10% and nearer than holding, per version: " + ", ".join(names))
    for k in S[names[0]]["effects"]:
        kind = S[names[0]]["effects"][k]["kind"]
        f = (lambda v: f"{100 * v:4.0f}%") if kind == "rel" else ((lambda v: f"{v:5.2f}") if kind == "abs" else (lambda v: f"{v:4.1f}cm"))
        cells = []
        for n in names:
            e = S[n]["effects"].get(k)
            if e is None:
                cells.append(" " * w)
                continue
            vals = [at_ms(S[n], k, m) for m in (50, 100, 150, 200)]
            ib = e.get("within_bar")
            cells.append(" ".join(f(v) if v is not None and np.isfinite(v) else "   - " for v in vals) + (f" in {100 * ib:3.0f}%" if ib is not None else "        ") + (f" near {100 * e['twin_better']:3.0f}%" if e.get("twin_better") is not None else ""))
        print(f"  {k:18s} " + "  ||  ".join(cells))
    print("profiles at 20 / 50 / 100 / 200 ms:")
    for c in ("T_e", "T_i", "n_e"):
        cells = []
        for n in names:
            pr = S[n].get("profiles", {}).get(c)
            if not pr:
                cells.append("-")
                continue
            idx = [LEADS_S.index(x) for x in (0.02, 0.05, 0.1, 0.2)]
            cells.append(" ".join(f"{100 * pr['median'][i]:4.0f}%" if pr["median"][i] is not None and np.isfinite(pr["median"][i]) else "   - " for i in idx) + f" ({pr.get('source', '')})")
        print(f"  {c:4s} " + "  ||  ".join(cells))
    print("events over the time both run (twin; measured), and the twin's own disruptions:")
    for n in names:
        pr = S[n]["per_run"]
        e_t, e_r = sum(x["elms"][0] for x in pr), sum(x["elms"][1] for x in pr)
        c_t, c_r = sum(x["crashes"][0] for x in pr), sum(x["crashes"][1] for x in pr)
        isl = [x["island_share"] for x in pr if x["island_share"][0] is not None]
        hm = [x["H_share"] for x in pr if x["H_share"][0] is not None]
        dis = [x for x in S[n]["ends"] if x["twin_disrupted"]]
        early = [x for x in dis if x["early_ms"] > 50]
        print(f"  {n:6s} runs {len(pr):3d}  ELMs {e_t:4d}; {e_r:3d}   crashes {c_t:3d}; {c_r:3d}   island time {np.mean([a for a, b in isl]):.2f}; {np.mean([b for a, b in isl]):.2f}"
              f"   H time {np.mean([a for a, b in hm]):.2f}; {np.mean([b for a, b in hm]):.2f}   disrupted {len(dis)} ({len(early)} more than 50 ms early)")
    if len(names) >= 2:                     # paired, run by run: each later version against the first, on the same shot and seed
        base = {(x["shot"], x["seed"]): x for x in S[names[0]]["per_run"]}
        print(f"paired against {names[0]}: runs where the version's median error over the run is lower (better) / higher, per effect")
        for n in names[1:]:
            cells = []
            for k in S[names[0]]["effects"]:
                b = w_ = 0
                for x in S[n]["per_run"]:
                    y = base.get((x["shot"], x["seed"]))
                    if y is None or x.get(k) is None or y.get(k) is None:
                        continue
                    if x[k] < y[k] - 1e-12:
                        b += 1
                    elif x[k] > y[k] + 1e-12:
                        w_ += 1
                cells.append(f"{k[:14]} {b}/{w_}")
            print(f"  {n:6s} " + ", ".join(cells))
    keys = [k for k in S[names[0]]["effects"]]
    cols = 5
    rows_ = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(3.7 * cols, 3.0 * rows_), constrained_layout=True)
    palette = ["#8c8c8c", "#1f6fb2", "#d9822b", "#2e8b57", "#7a4fa3"]
    for ax, k in zip(axes.flat, keys):
        for n, col in zip(names, palette):
            e = S[n]["effects"].get(k)
            if e is None:
                continue
            g = 1e3 * np.array(S[n]["grid_s"])
            sc = 100 if e["kind"] == "rel" else 1
            ax.plot(g, np.array(e["median"], float) * sc, color=col, lw=1.8, label=n)
        if S[names[0]]["effects"][k]["kind"] == "rel":
            ax.axhline(10, color=BAR, lw=1, ls=":")
        ax.set_title(k, fontsize=9.5, loc="left")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Median error against ms since the start, per version: " + ", ".join(names), fontsize=11)
    fig.savefig(os.path.join(FIG, "h83_compare_" + "_".join(names) + ".png"), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    if "--compare" in sys.argv:
        compare(sys.argv[sys.argv.index("--compare") + 1].split(","))
    elif "--common" in sys.argv:
        common_support(sys.argv[sys.argv.index("--common") + 1].split(","))
    else:
        configure(sys.argv)
        main()
