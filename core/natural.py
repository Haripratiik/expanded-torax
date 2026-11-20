"""Natural recreation of real MAST shots, instabilities included (H43, docs/digital-twin.md). A shot runs by itself and is compared with what really happened; ELMs, sawteeth and islands start on their own.

Going in, at the start only: the profiles (mean of the last slices), EFIT's equilibrium (held), transport anchored once on that state (mast_anchor.anchor), whether the plasma is in H-mode and how fast it is ELMing, the toroidal rotation.
Going in, against time: the machine's programme: injected beam power, the programmed plasma current (scaled to the measured value at the start), the programmed density target. Nothing else measured after the start enters, and no command is issued.

Laws (fit_laws, written to runs/twin/mast_natural2/laws.json). Coefficients come from the instability catalog without the 19 held-out shots (docs/mast-instabilities.md) and from the 19 fit shots:
  pedestal   the top at rho 0.91 follows the programme: T_e, T_i, n_e against their starting values as a power law in Ip, beam power and the density target, with a jump at L-H. Robust fit on the fit shots
  hmode      L-H at a hazard rate from a logistic fit of state at H-mode starts against L-mode times, in beam phases; back at a constant rate. Or given: the measured windows, labelled as such
  elms       only in H-mode. The interval from a law in power, current and density, measured to fall with power, a per-shot offset (from the trailing ELMs if there are any, otherwise drawn from the scatter left), log-normal scatter within a run. The drop from a share of the exhaust power, in the measured ratio of T to n
  sawtooth   TORAX's own trigger and redistribution, with the critical shear set on the fit shots by the measured period. With gate, the trigger is on or off in stretches, for the measured share of low-q0 beam time against beam power per density (0.54 under 0.3 MW per 1e19 m-3, 0.11 to 0.17 above 0.5): the state does not say which, and the data look like fast ions stabilizing them (inferred)
  modes      a 2/1 island at a hazard rate from a logistic fit of state at slow n = 1 onsets against random times. The island equation alone gives no growth on MAST (experiments/torax/check_mre_mast.py: bootstrap term 0.1 to 0.8 against -2 from Delta' and about -2 from curvature), and real modes grow, so a tearing drive is put in that a 1.5D code cannot compute: rbar Delta' = rdelta0 (1 - w / w_sat), with w_sat drawn from the measured widths and rdelta0 set on the fit shots by the measured rise; the drive stops after a measured lifetime unless the mode has locked. Rotation: the measured plasma rotation at q = 2 at the start, held. The lock at 1 kHz as in the catalog's rule, the quench at its measured rate. Whether a mode locks, and when, and what it costs in energy, then come from the model

  free       FreeGSNKE's free boundary every 10 ms, in place of EFIT's surfaces held (class FreeBoundary)
Overrides that replace a fit with an open tool or a general model: "pedestal": "eped" (the temperature follows EPED-NN's pressure from the start), "elm_law": "energy" (the general cycle in ntm.ELMs). "beams": "frozen" keeps the beam code of the frozen test.

A run gets a seed: the laws are stochastic where the data say the state alone does not decide (cross-validated areas under the curve of about 0.65), so instabilities are scored as statistics over seeds, and T, n as profiles.
Usage (TORAX's venv; scalars.npz and shot files from mast_scalars.py and mast_convert_natural.py in WSL):
  python core/natural.py laws
  python core/natural.py run --split fit --models beams pedestal hmode elms sawtooth modes --seeds 3 --tag first
  python core/natural.py score --split fit --tag first
Outputs in runs/twin/mast_natural2/.
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mast  # noqa: E402
import machines

OUT = os.path.join(mast.ROOT, "runs", "twin", "mast_natural2")
CATALOG = os.path.join(mast.ROOT, "runs", "twin", "mast_catalog")
EXTRAS = os.path.join(mast.ROOT, "runs", "twin", "mast_natural", "extras")
SHOTS = os.path.relpath(os.path.join(OUT, "shots"), mast.ROOT)
QE = 1.602176634e-19


def catalog():
    return json.load(open(os.path.join(CATALOG, "catalog.json")))["shots"]


def split(name):
    return [s["shot"] for s in json.load(open(os.path.join(CATALOG, "natural_test_shots.json")))["shots"] if s["split"] == name]


def logistic(X, y, it=100, l2=1e-3):
    b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p) + 1e-9
        b = b + np.linalg.solve((X * W[:, None]).T @ X + l2 * np.eye(X.shape[1]), X.T @ (y - p) - l2 * b)
    return b


def auc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())


def cv_auc(X, y, groups, folds=5):
    fold = np.array([int(g) * 2654435761 % 2 ** 32 % folds for g in groups])
    pred = np.zeros(len(y))
    for f in range(folds):
        pred[fold == f] = X[fold == f] @ logistic(X[fold != f], y[fold != f])
    return auc(pred[y == 1], pred[y == 0])


def huber(X, y, it=30, c=1.345):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(it):
        r = y - X @ b
        s = max(1.4826 * np.median(np.abs(r)), 1e-3)
        w = np.minimum(1.0, c * s / np.maximum(np.abs(r), 1e-9))
        b = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    return b


H_FEATURES = ("P", "nel", "Ip", "tn", "q95")
MODE_FEATURES = ("nel", "beta", "Ip", "band")


def h_row(P, nel, Ip, tn, q95):
    return [1.0, np.log(max(P, 1e4) / 2.5e6), np.log(max(nel, 1e17) / 2.5e19), np.log(max(Ip, 1e4) / 7e5), np.log(max(tn, 1e-3) / 0.1), np.log(max(q95, 1.0) / 6.0)]


def mode_row(nel, beta, Ip, q_min):
    return [1.0, np.log(max(nel, 1e17) / 2.5e19), beta - 2.0, np.log(max(Ip, 1e4) / 7e5), 1.0 if 1.2 <= q_min < 1.7 else 0.0]


def fit_laws():
    warnings.filterwarnings("ignore")
    cat = catalog()
    held = set(split("held_out"))
    raw = np.load(os.path.join(OUT, "scalars.npz"))
    Z = {k: raw[k] for k in raw.files}
    laws = dict(held_out=sorted(held))
    rng = np.random.default_rng(0)

    def at(sh, t, names):
        if f"{sh}_t" not in Z:
            return None
        tt = Z[f"{sh}_t"]
        i = int(np.searchsorted(tt, t) - 1)
        if i < 4 or i >= len(tt) or any(f"{sh}_{k}" not in Z for k in names):
            return None
        return {k: float(np.nanmedian(Z[f"{sh}_{k}"][i - 1:i + 1])) for k in names}

    # ELM frequency
    rows, sigma = [], []
    for k, x in cat.items():
        if int(k) in held:
            continue
        times = np.array(x.get("elms", {}).get("t") or [])
        for p in x.get("elm_phases") or []:
            if p["n"] >= 5:
                d = np.diff(times[(times >= p["t0"]) & (times <= p["t1"])])
                if len(d) >= 4:
                    sigma.append(float(np.std(np.log(d))))
            if p["n"] >= 5 and "level" in p.get("basis", "") and (p["p_nbi_MW"] or 0) > 0.5 and 0.3 < (p["nel_1e19"] or 0) < 30 and (p["ip_kA"] or 0) > 100:
                rows.append((int(k), p["f_median_Hz"], p["p_nbi_MW"], p["ip_kA"], p["nel_1e19"]))
    r = np.array(rows, float)
    X = np.c_[np.ones(len(r)), np.log(r[:, 2] / 2), np.log(r[:, 3] / 700), np.log(r[:, 4] / 3)]
    y = np.log(r[:, 1])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    laws["elm"] = dict(f0_Hz=float(np.exp(b[0])), P=float(b[1]), Ip=float(b[2]), nel=float(b[3]), scatter_shot=float((y - X @ b).std()), sigma_within=float(np.median(sigma)), n_runs=len(r), n_shots=len(set(r[:, 0])), drop_T_over_n=11.9 / 15.0, drop_sum_isolated=(11.9 + 15.0) / 100, f_factor=0.5, share=0.13)      # share: of the exhaust power, so that large isolated ELMs on the fit shots drop by the measured 11.9% + 15.0% (4 events, 0.105 to 0.167)

    # L-H and H-L
    cases, controls, lasts = [], [], []
    names = ("P_nbi", "V_loop", "Ip", "nel", "q95", "li", "beta_n", "q_min")
    for k, x in cat.items():
        sh = int(k)
        if sh in held:
            continue
        ws = x["phases"]["hmode"]["windows"]
        for w in [w for w in x["phases"]["nbi"].get("windows") or [] if w["p_MW"] >= 1 and w["t1"] - w["t0"] >= 0.02]:
            for h in ws:
                if w["t0"] + 0.01 <= h["t0"] <= w["t1"]:
                    s = at(sh, h["t0"] - 0.005, names)
                    if s:
                        cases.append(h_row(s["P_nbi"] + abs(s["V_loop"] * s["Ip"]), s["nel"], s["Ip"], h["t0"] - 0.005 - w["t0"], s["q95"]) + [sh])
                    lasts.append(min(h["t1"], w["t1"]) - h["t0"])
            for _ in range(max(1, int((w["t1"] - w["t0"]) / 0.03))):
                t = rng.uniform(w["t0"] + 0.01, w["t1"])
                if any(h["t0"] - 0.01 <= t <= h["t1"] for h in ws):
                    continue
                s = at(sh, t - 0.005, names)
                if s:
                    controls.append(h_row(s["P_nbi"] + abs(s["V_loop"] * s["Ip"]), s["nel"], s["Ip"], t - 0.005 - w["t0"], s["q95"]) + [sh])
    a = np.array(cases + controls, float)
    good = np.isfinite(a).all(axis=1)
    y = np.r_[np.ones(len(cases)), np.zeros(len(controls))][good]
    a = a[good]
    laws["hmode"] = dict(features=H_FEATURES, coef=logistic(a[:, :-1], y).tolist(), control_rate=1 / 0.03, cv_auc=cv_auc(a[:, :-1], y, a[:, -1]), n_starts=int(y.sum()), n_controls=int((1 - y).sum()), mean_duration_s=float(np.mean(lasts)), median_duration_s=float(np.median(lasts)), rate_factor=1.8)        # rate_factor: set on the fit shots, whose H-mode time is 1.52 s against 0.84 s from the fitted hazard: the test shots are chosen to hold ELMy shots, so they spend more of their time in H-mode than the catalog's 20%

    # mode onsets, where a q = 2 surface exists
    cases, controls = [], []
    names = ("nel", "beta_n", "Ip", "q_min")
    T = 0.0
    for k, x in cat.items():
        sh = int(k)
        if sh in held or not x.get("modes_available"):
            continue
        ws = [w for w in x["phases"]["nbi"].get("windows") or [] if w["p_MW"] >= 1 and w["t1"] - w["t0"] >= 0.01]
        for w in ws:
            for _ in range(max(1, int((w["t1"] - w["t0"]) / 0.03))):
                t = rng.uniform(w["t0"], w["t1"])
                s = at(sh, t, names)
                if s and s["q_min"] < 2:
                    controls.append(mode_row(s["nel"], s["beta_n"], s["Ip"], s["q_min"]) + [sh])
        for m in x.get("modes") or []:
            if m["n"] == 1 and m["family"] == "slow" and m.get("r1") and any(w["t0"] <= m["t_on"] <= w["t1"] for w in ws):
                s = at(sh, m["t_on"] - 0.005, names)
                if s and s["q_min"] < 2:
                    cases.append(mode_row(s["nel"], s["beta_n"], s["Ip"], s["q_min"]) + [sh])
    a = np.array(cases + controls, float)
    good = np.isfinite(a).all(axis=1)
    y = np.r_[np.ones(len(cases)), np.zeros(len(controls))][good]
    a = a[good]
    widths, lives, rises, f_ons = [], [], [], []
    for k, x in cat.items():
        if int(k) in held:
            continue
        for m in x.get("modes") or []:
            if m["n"] == 1 and m["family"] == "slow" and m.get("r1"):
                w = (m.get("cost") or {}).get("w_over_a")
                if w:
                    widths.append(float(w))
                if not m.get("locked"):
                    lives.append(m["dur_ms"] / 1e3)
                if m.get("t_sat"):
                    rises.append(1e3 * (m["t_sat"] - m["t_on"]))
                f_ons.append(1e3 * m["f_on_kHz"])
    laws["mode"] = dict(features=MODE_FEATURES, coef=logistic(a[:, :-1], y).tolist(), control_rate=1 / 0.03, cv_auc=cv_auc(a[:, :-1], y, a[:, -1]), n_onsets=int(y.sum()), n_controls=int((1 - y).sum()), seed_factor=3.0, w_seed_cm=[0.5, 1.5], f_lock=1000.0, quench_within_20ms=0.46, tau_M0=0.05, tau_A0=7e-7, tau_w=0.005, w_vac=0.0125, C_M=0.0, C_1=0.03,     # rotation: spin_scan.py, 48 settings against the lock share per width and the onset to lock in the catalog without the held-out shots, then C_1 by the runs on the fit shots (cost): at 0.025 locked 0, 21, 44, 75%, at 0.04 locked 0, 28, 72, 100%, against 25, 15, 59, 85%
                        rdelta0=5.5, w_sat_over_a=sorted(widths), life_s=sorted(lives), rise_ms_median=float(np.median(rises)), f_on_Hz=sorted(f_ons))

    # pedestal and rotation, fit shots
    rows, rot = [], []
    for sh in split("fit"):
        e = dict(np.load(os.path.join(EXTRAS, f"{sh}.npz")))
        g = dict(np.load(os.path.join(mast.ROOT, SHOTS, f"{sh}.npz"), allow_pickle=False))
        t, rho = g["t"], g["rho"]
        i, k0 = int(np.argmin(np.abs(rho - mast.FIXED["rho_ped"]))), mast.FIXED["trailing_slices"] - 1
        mean5 = lambda v: np.array([np.mean(v[max(k - 2, 0):k + 3]) for k in range(len(t))])
        start = lambda v: np.mean(v[max(k0 - 4, 0):k0 + 1])
        Te, ne, Ti = g["T_e"][:, i], g["n_e"][:, i], g["T_i"][:, i] if "T_i" in g else None
        N = np.array([np.trapezoid(g["n_e"][j], g["V_rho"][j]) for j in range(len(t))])
        Ip = np.interp(t, e["t_prog"], np.nan_to_num(e["prog_i_plasma"]))
        P = np.maximum(np.nan_to_num(g["P_heat"]), 0.05)
        pr = np.interp(t, e["t_prog"], np.nan_to_num(e["prog_n_e_line"]))
        windows = cat[str(sh)]["phases"]["hmode"]["windows"]
        H = np.array([any(w["t0"] <= x <= w["t1"] for w in windows) for x in t], float)
        for k in range(k0 + 1, len(t)):
            if abs(np.log(g["Ip"][k] / g["Ip"][k0]) - np.log(max(Ip[k], 1e4) / max(Ip[k0], 1e4))) > 0.15:     # the plasma has left the programme: its end
                continue
            rows.append((sh, np.log(mean5(Te)[k] / start(Te)), np.log(mean5(ne)[k] / start(ne)), np.log(mean5(Ti)[k] / start(Ti)) if Ti is not None else np.nan, np.log(max(Ip[k], 1e4) / max(Ip[k0], 1e4)), np.log(P[k] / P[k0]), np.log(pr[k] / pr[k0]) if pr[k0] > 0 and pr[k] > 0 else np.nan, H[k] - H[k0], np.log(mean5(N)[k] / start(N))))
        if "v_i" in e:
            q, rp = g["q_psi"], g["rho_psi"]
            for k in range(k0, len(t), 4):
                below = np.where(q[k] <= 2)[0]
                if len(below) == 0:
                    continue
                r2 = rp[k][below[-1]]
                j = int(np.argmin(np.abs(e["t_cx"] - t[k])))
                c = g["cx_rho"][k]
                near = np.isfinite(c) & (np.abs(c - r2) < 0.1) & (e["R_cx"] > g["R_mag"][k])
                if near.sum() >= 2:
                    v = np.abs(np.nanmedian(e["v_i"][near, j]))
                    if np.isfinite(v) and v > 1e3:
                        rot.append((sh, v / (2 * np.pi * np.median(e["R_cx"][near])), P[k], g["nel"][k]))
    R = np.array(rows, float)
    laws["pedestal"] = {}
    for name, col in (("T_e", 1), ("n_e", 2), ("T_i", 3), ("N", 8)):
        res = {}
        for label, cols in (("programme", (4, 5, 6, 7)), ("no_programme", (4, 5, 7))):
            X, y = R[:, cols], R[:, col]
            good = np.isfinite(y) & np.isfinite(X).all(axis=1)
            b = huber(X[good], y[good])
            res[label] = dict(zip(np.array(("Ip", "P", "programme", "H") if label == "programme" else ("Ip", "P", "H")).tolist(), b.round(4).tolist()))
            res[label + "_rms"] = [float(np.sqrt(np.mean(y[good] ** 2))), float(np.sqrt(np.mean((y[good] - X[good] @ b) ** 2)))]
        laws["pedestal"][name] = res
    laws["pedestal"]["tau_s"] = 0.01
    Q = np.array(rot, float)                           # within a shot the rotation follows neither power nor density (exponents -0.17, 0.26; scatter unchanged), so it stays at its value at the start
    laws["rotation"] = dict(f0_Hz=float(np.median(Q[:, 1])), n=len(Q), mode_over_plasma=1.6)      # R1 slow n = 1 modes on the fit shots: frequency over the plasma's at q = 2, median 1.61, IQR 1.04 to 2.67 (12). Fast n = 1 modes sit at 8 times it: core modes, which this model does not hold
    # sawteeth in beam phases: where q0 is low, the share of time in a sequence falls with beam power per density (fast ions stabilize them, inferred). Dwell: how long a sequence lasts
    Pn_edges = [0.0, 0.3, 0.5, 0.7, 0.9, 1.2, 1.6, 99.0]
    in_seq, out_seq, lengths = np.zeros(len(Pn_edges) - 1), np.zeros(len(Pn_edges) - 1), []
    for k, x in cat.items():
        sh = int(k)
        if sh in held:
            continue
        sq = (x.get("sawteeth") or {}).get("sequences") or []
        for w in [w for w in x["phases"]["nbi"].get("windows") or [] if w["p_MW"] >= 1 and w["t1"] - w["t0"] >= 0.02]:
            for q in sq:
                a_, b_ = max(q["t0"], w["t0"]), min(q["t1"], w["t1"])
                if b_ > a_:
                    lengths.append(b_ - a_)
            for t in np.arange(w["t0"], w["t1"], 0.01):
                s = at(sh, t, ("q0", "P_nbi", "nel"))
                if s and s["q0"] < 1.05:
                    i = int(np.clip(np.searchsorted(Pn_edges, s["P_nbi"] / 1e6 / max(s["nel"] / 1e19, 0.01)) - 1, 0, len(in_seq) - 1))
                    if any(q["t0"] <= t <= q["t1"] for q in sq):
                        in_seq[i] += 1
                    else:
                        out_seq[i] += 1
    laws["sawtooth"] = dict(Pn_edges=Pn_edges, share=(in_seq / np.maximum(in_seq + out_seq, 1)).round(3).tolist(), samples=(in_seq + out_seq).tolist(), dwell_s=float(np.median(lengths)), s_critical=0.4, mixing=1.3)
    json.dump(laws, open(os.path.join(OUT, "laws.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in laws.items() if k != "held_out"}, indent=1))


def ntm_rho():
    import ntm
    return ntm.TURB_RHO


class Natural:
    """What the plasma does by itself, each step. Attached to a harness; reads only the plasma and the programme."""

    def __init__(self, h, g, extras, laws, shot, k, seed, models, given_h=None, h0=None, elm_rate0=None, overrides=None):
        self.h, self.r, self.laws, self.models = h, h.reactor, laws, set(models)
        self.rng = np.random.default_rng(100003 * seed + shot)
        t = g["t"]
        self.t_cut = float(t[k])
        Ip_prog = np.nan_to_num(extras["prog_i_plasma"])
        self.Ip = lambda x: float(np.interp(x, extras["t_prog"], Ip_prog))
        self.Ip_scale = float(np.mean(g["Ip"][max(k - 1, 0):k + 1])) / max(self.Ip(self.t_cut), 1e4)
        self.P = lambda x: max(float(np.interp(x, t, np.nan_to_num(g["P_heat"]))), 0.0)
        pr = np.nan_to_num(extras["prog_n_e_line"]) if "prog_n_e_line" in extras else None
        self.prog = None if pr is None or np.interp(self.t_cut, extras["t_prog"], pr) <= 0 else (lambda x: max(float(np.interp(x, extras["t_prog"], pr)), 1e-3))
        self.Ip0, self.P0 = self.Ip(self.t_cut), max(self.P(self.t_cut), 0.05)
        self.prog0 = self.prog(self.t_cut) if self.prog else None
        on = np.nan_to_num(g["P_heat"]) >= 1.0
        self.t_beam = float(t[np.argmax(on)]) if on.any() else float(t[0])
        self.given_h = given_h
        self.H = bool(h0)
        self.H0 = self.H
        self.fT, self.fn, self.fTi = 1.0, 1.0, 1.0
        self.h_log, self.seeds = [], []
        self.t_last = self.t_cut
        e = laws["elm"]
        self.elm_offset = None
        self.elm_rate0 = elm_rate0
        self.mode_law = dict(laws["mode"], **(overrides or {}))
        self.overrides = overrides or {}
        self.f0 = None
        self.t_decay = None
        self.N0 = None
        self.saw_on = None
        self.s_critical = None
        self.nel_scale = None
        self.closure = None
        self.nel_start = float(np.mean(g["nel"][max(k - 4, 0):k + 1]))
        profile_mean = float(np.mean(g["n_e"][max(k - 4, 0):k + 1]))
        if not np.isfinite(self.nel_start) or not 1 / 3 < self.nel_start / profile_mean < 3:      # some shots have no interferometer, others read 100 times too high (15870, 21828, 27714, 28225). The fitted profile's mean sits within about 5% of the line density on the rest (median 1.02, IQR 0.99 to 1.05, 75 shots)
            self.nel_start = profile_mean
        self.n_line_now = self.nel_start

    def elm_law(self, n_line, Ip, P):
        e = self.laws["elm"]
        return e["f_factor"] * e["f0_Hz"] * (max(P, 0.05) / 2) ** e["P"] * (max(Ip, 1e4) / 7e5) ** e["Ip"] * (max(n_line, 1e17) / 3e19) ** e["nel"]     # f_factor: set on the fit shots, 589 Hz against 292 Hz without it: the run's density is low, and the law goes as density to the -0.9

    def elm(self, r, t):
        """The law's answer for the next ELM: see ntm.ELMs."""
        if "elms" not in self.models or not self.H:
            return None
        e = self.laws["elm"]
        rec = r.records[-1] if r.records else {}
        if self.overrides.get("elm_law") == "energy":          # no MAST number: the harness's general cycle, a crash of set size paid for by a share of the exhaust power (ntm.ELMs, Herrmann 2002)
            import ntm
            g = ntm.ELMs()
            cp, geo = r.state.core_profiles, r.state.geometry
            top = int(np.argmin(np.abs(np.asarray(geo.rho_norm) - mast.FIXED["rho_ped"])))
            W_ped = 3e3 * QE * float(cp.n_e.value[top]) * float(cp.T_e.value[top]) * float(geo.volume_face[-1])
            interval = (g.drop_T + g.drop_n) * W_ped / (g.power_share * max(rec.get("P_SOL", 1e6), 1e5)) * max(1.0 + g.jitter * float(self.rng.normal()), 0.2)
            return max(interval, 1e-3), g.drop_T, g.drop_n
        n_line, Ip, P = self.n_line_now, self.Ip(t) * self.Ip_scale, self.P(t)
        f = self.elm_law(n_line, Ip, P)
        if self.elm_offset is None:
            self.elm_offset = np.log(self.elm_rate0 / f) if self.elm_rate0 else float(self.rng.normal(0.0, e["scatter_shot"]))
        interval = float(np.exp(self.elm_offset + float(self.rng.normal(0.0, e["sigma_within"])))) / f
        cp, geo = r.state.core_profiles, r.state.geometry
        top = int(np.argmin(np.abs(np.asarray(geo.rho_norm) - mast.FIXED["rho_ped"])))
        W_ped = 3e3 * QE * float(cp.n_e.value[top]) * float(cp.T_e.value[top]) * float(geo.volume_face[-1])
        total = min(self.overrides.get("share", e["share"]) * max(rec.get("P_SOL", 1e6), 1e5) * interval / max(W_ped, 1.0), 0.6)
        ratio = e["drop_T_over_n"]
        return interval, total * ratio / (1 + ratio), total / (1 + ratio)

    def __call__(self, r):
        t = r.t
        dt = t - self.t_last
        self.t_last = t
        rec = r.records[-1] if r.records else {}
        cp = r.state.core_profiles
        q = np.asarray(cp.q_face)
        q_min = float(q.min())
        Ip_now = self.Ip(t) * self.Ip_scale
        P_now = self.P(t)
        n_line, beta, q95 = rec.get("n_line", float(r.ppo.n_e_line_avg)), rec.get("beta_N", float(r.ppo.beta_N)), rec.get("q95", float(r.ppo.q95))
        if self.nel_scale is None:                     # the laws were fit on the interferometer's line density and EFIT's beta, which includes fast ions
            self.nel_scale = self.nel_start / max(n_line, 1e17)
        n_line = n_line * self.nel_scale
        self.n_line_now = n_line
        W_fast = (self.h.plant.beam.last or {}).get("W_fast", 0.0) if self.h.plant.beam is not None else 0.0
        beta = beta * (1 + W_fast / max(rec.get("W_thermal", float(r.ppo.W_thermal_total)), 1.0))
        # H-mode
        if self.given_h is not None:
            H = any(a <= t <= b for a, b in self.given_h)
        elif "hmode" in self.models:
            H = self.H
            if dt > 0 and P_now >= 1.0:
                if not H:
                    x = np.array(h_row(P_now * 1e6 + abs(rec.get("P_ohmic", 0.0)), n_line, Ip_now, t - self.t_beam, q95))
                    H = self.rng.random() < 1 - np.exp(-self.laws["hmode"]["rate_factor"] * np.exp(x @ np.array(self.laws["hmode"]["coef"])) * self.laws["hmode"]["control_rate"] * dt)
                else:
                    H = self.rng.random() >= 1 - np.exp(-dt / self.laws["hmode"]["mean_duration_s"])
            elif dt > 0 and H:
                H = False
        else:
            H = self.H
        if H != self.H:
            self.h_log.append((t, bool(H)))
            self.H = H
        # the pedestal follows the programme
        if "pedestal" in self.models:
            d = dict(Ip=np.log(max(self.Ip(t), 1e4) / max(self.Ip0, 1e4)), P=np.log(max(P_now, 0.05) / self.P0), H=float(self.H) - float(self.H0))
            if self.prog:
                d["programme"] = np.log(self.prog(t) / self.prog0)
            kind = "programme" if self.prog else "no_programme"
            target = {c: float(np.exp(np.clip(sum(self.laws["pedestal"][c][kind][a] * d[a] for a in self.laws["pedestal"][c][kind]), -1.5, 1.5))) for c in ("T_e", "n_e", "T_i", "N")}
            a = 1 - np.exp(-dt / self.laws["pedestal"]["tau_s"]) if dt > 0 else 0.0
            self.fT += a * (target["T_e"] - self.fT)
            self.fTi += a * (target["T_i"] - self.fTi)
            if self.overrides.get("density") == "N":           # the plasma's electrons follow the programme, and the pedestal density is what moves them. Tried on the fit shots: n_e no better than the law on the pedestal top, T_e worse (46 against 36% at 100 ms)
                geo = r.state.geometry
                N_now = float(np.sum(np.asarray(cp.n_e.value) * np.asarray(geo.vpr))) * float(np.asarray(geo.drho_norm).ravel()[0])
                if self.N0 is None:
                    self.N0 = N_now
                if dt > 0:
                    self.fn = float(np.clip(self.fn * np.exp(dt / self.overrides.get("tau_N", 0.01) * np.log(target["N"] * self.N0 / N_now)), 0.2, 12.0))
            else:
                self.fn += a * (target["n_e"] - self.fn)
            if self.overrides.get("density") == "N_source" and self.h.plant is not None:      # the same count, made up by a source inside the plasma in place of the pedestal: pushing the pedestal gets the count right and the profile wrong, core low and edge high
                import actuators
                geo = r.state.geometry
                rho_c, dv_c = np.asarray(geo.rho_norm), np.asarray(geo.vpr) * float(np.asarray(geo.drho_norm).ravel()[0])
                N_now = float(np.sum(np.asarray(cp.n_e.value) * dv_c))
                if self.N0 is None:
                    self.N0 = N_now
                rate = N_now * np.log(max(target["N"], 1e-3) * self.N0 / max(N_now, 1.0)) / self.overrides.get("tau_N", 0.03)
                rho0, width = self.overrides.get("fuel_rho", 0.5), self.overrides.get("fuel_width", 0.3)
                if rate >= 0 or not self.overrides.get("signed"):
                    self.h.plant.parts["fuel"] = max(rate, 0.0) * actuators.gaussian(ntm_rho(), rho0, width) / max(float((actuators.gaussian(rho_c, rho0, width) * dv_c).sum()), 1e-30)
                else:       # more particles than the law holds: they leave in proportion to the density, which keeps the profile's shape and cannot go negative. Stands in for transport: the anchor balances a steady profile, which fixes the pinch over D and not D, so the plasma's own particle time comes out far too long
                    self.h.plant.parts["fuel"] = rate * np.interp(ntm_rho(), rho_c, np.asarray(cp.n_e.value)) / max(N_now, 1.0)
            import ntm
            if self.overrides.get("pedestal") == "eped":       # the pressure follows EPED-NN's trend from where this shot starts, in place of the fitted law. Its level is 4 times MAST's (0.26, IQR 0.18 to 0.33), as MAST lies outside its training range and its ELMs are not the kind it predicts, but its trend ranks MAST's pedestals at Spearman 0.80
                trend = self.eped_trend(r, cp)
                self.fT += a * (trend / max(self.fn, 0.05) * np.exp(self.laws["pedestal"]["T_e"][kind]["H"] * d["H"]) - self.fT)
                self.fTi += a * (trend / max(self.fn, 0.05) * np.exp(self.laws["pedestal"]["T_i"][kind]["H"] * d["H"]) - self.fTi)
            r.ped_extra = np.array([self.fT, self.fn])
            ntm.PEDESTAL_TI[0] = self.fTi / self.fT
            if r.elms is None or r.elm_period is None:     # with ELMs on, the reactor sets the pedestal after each step, ripple included, from ped_extra
                ntm.PEDESTAL[:] = r.ped_scale * r.ped_extra
        # sawteeth: on or stabilized, with the measured share of time against beam power per density
        if "sawtooth" in self.models and "gate" in self.models and dt > 0:
            law = self.laws["sawtooth"]
            Pn = P_now / max(n_line / 1e19, 0.01)
            p = float(np.clip(law["share"][int(np.clip(np.searchsorted(law["Pn_edges"], Pn) - 1, 0, len(law["share"]) - 1))], 0.01, 0.99)) if P_now >= 0.5 else 0.5
            if self.saw_on is None:
                self.saw_on = bool(self.rng.random() < p)
                changed = True
            else:
                rate = (1 if self.saw_on else p / (1 - p)) / law["dwell_s"]
                changed = self.rng.random() < 1 - np.exp(-rate * dt)
                if changed:
                    self.saw_on = not self.saw_on
            if changed:
                r.set_param("mhd.sawtooth.trigger_model.s_critical", self.s_critical if self.saw_on else 1e3)
        # transport changes with the state beyond what the base model says: see closure_data
        if self.closure is not None:
            dx = np.array([np.log(max(n_line, 1e17) / self.nel_start), np.log(max(self.Ip(t), 1e4) / max(self.Ip0, 1e4)), np.log(max(P_now, 0.05) / self.P0), float(self.H) - float(self.H0)])
            self.h.table0 = self.table_start * np.exp(np.clip(self.closure @ dx, -3.0, 3.0))
        # rotation and seeds
        if "modes" in self.models:
            md = self.h.modes[0]
            if self.f0 is None:
                self.f0 = self.start_rotation(r)
            md.rotation.f0 = self.f0
            if r.widths[0] < 0.003 and q_min < 2 and P_now >= 1.0 and dt > 0:
                m = self.mode_law
                rate = m["seed_factor"] * m["control_rate"] * np.exp(np.array(mode_row(n_line, beta, Ip_now, q_min)) @ np.array(m["coef"]))
                if self.rng.random() < 1 - np.exp(-rate * dt):
                    w = float(np.exp(self.rng.uniform(*np.log(m["w_seed_cm"]))))
                    a = float(r.state.geometry.a_minor)
                    self.h.coeffs.rdelta0, self.h.coeffs.w_sat = m["rdelta0"], a * float(self.rng.choice(m["w_sat_over_a"]))
                    if not self.rot_start:                 # no q = 2 surface at the start, so no rotation there: a measured onset frequency, at random
                        self.f0 = max(float(self.rng.choice(m["f_on_Hz"])), 1.5 * m["f_lock"])
                        md.rotation.f0 = self.f0
                        r.spins[0] = (2 * np.pi * self.f0, r.spins[0][1])
                    self.t_decay = t + float(self.rng.choice(m["life_s"]))
                    self.h.apply(dict(cmd="island", m=2, n=1, w_cm=w))
                    self.seeds.append(dict(t=t, w_cm=w, w_sat_cm=100 * self.h.coeffs.w_sat, t_decay=self.t_decay))
            if r.widths[0] >= 0.003 and self.t_decay is not None and t >= self.t_decay and r.spins[0][0] > 0.7 * 2 * np.pi * self.f0:      # the drive goes, unless the mode is already braking towards a lock: large modes nearly always lock on MAST (85% above w/a 0.2)
                self.h.coeffs.rdelta0, self.h.coeffs.w_sat = None, None
        rf = np.asarray(r.state.geometry.rho_face_norm)
        below = np.where(q <= 1.0)[0]
        r.carry.update(rho_q1=float(rf[below[-1]]) if len(below) else 0.0, s_q1=float(np.asarray(cp.s_face)[below[-1]]) if len(below) else 0.0)
        r.carry.update(q0=float(q[0]), q_min=q_min, H=float(self.H), fT=self.fT, fn=self.fn, f0=self.f0 or 0.0, Ip_now=Ip_now)

    def eped_trend(self, r, cp):
        """EPED-NN's pedestal pressure now over at the start of the run."""
        import epednn
        if getattr(self, "eped_model", None) is None:
            self.eped_model = epednn.Model("EPED1NNensemble")
        geo, ppo = r.state.geometry, r.ppo
        top = int(np.argmin(np.abs(np.asarray(geo.rho_norm) - mast.FIXED["rho_ped"])))
        W_th = max(float(ppo.W_thermal_total), 1.0)
        W_fast = (self.h.plant.beam.last or {}).get("W_fast", 0.0) if self.h.plant.beam is not None else 0.0
        delta = epednn.effective_delta(float(np.asarray(getattr(geo, "delta_lower_face", geo.delta_face))[-1]), float(np.asarray(getattr(geo, "delta_upper_face", geo.delta_face))[-1]))
        p = self.eped_model(a=float(geo.a_minor), betan=float(ppo.beta_N) * (1.0 + W_fast / W_th), bt=abs(float(geo.B_0)), delta=delta, ip=max(abs(float(np.asarray(cp.Ip_profile_face)[-1])) / 1e6, 0.05), kappa=float(np.asarray(geo.elongation_face)[-1]), m=2.0, neped=max(float(np.asarray(cp.n_e.value)[top]) / 1e19, 0.1), r=float(geo.R_major), zeffped=1.5, nesep_ratio=0.25, tesep=75.0)["p"]
        if getattr(self, "eped_p0", None) is None:
            self.eped_p0 = p
        return float(np.clip(p / self.eped_p0, 0.1, 10.0))

    def start_rotation(self, r):
        """Plasma rotation at q = 2 at the start, from the charge-exchange chords if they reach it, otherwise the law. Hz."""
        law = self.laws["rotation"]
        self.f_rot_source = "charge exchange" if self.rot_start else "median"
        return max(law["mode_over_plasma"] * (self.rot_start or law["f0_Hz"]), 1.5 * self.mode_law["f_lock"])


R_OUT_MAST = machines.MACHINES["MAST"]["free"]["R_out_flat"]                      # m, MAST's outer edge once the current is up: median over the 19 fit shots, IQR 1.40 to 1.48. Moving to it as the programmed current rises: within 3 to 5 cm of EFIT on those shots, against 8 to 12 for holding the start value
LAST_SURFACE = 0.99                    # of normalized flux, as for EFIT's surfaces at the start (mast.PSI_LAST)


class FreeBoundary:
    SNAPSHOT_SHARED = ("fb", "g", "coil", "coil_t", "Ip_prog", "res0", "vessel0", "anchor", "shape0")       # the shot's data and the start: fixed for the run
    """FreeGSNKE in the run (H46): every interval, TORAX's plasma and the measured coil currents go to the free-boundary solver and its surfaces come back. Vessel currents follow their circuit equation, driven by the coils and the plasma, from before the shot's start up to the end. At the start, what EFIT's flux map wants beyond that, with TORAX's plasma on it, is kept as a fixed set of vessel currents, so the run starts on EFIT's shape and carries differences.
    MAST's radial position is not left to the measured coils: 1 mT of vertical field is about 5 cm there, the coil model is off by several mT against EFIT, and the machine itself runs on position feedback. So radial="programme" moves the outer edge from its start value to MAST's usual one as the programmed current rises, what an operator would ask for ("hold" keeps the start value, "efit" follows EFIT's edge, not hands-off, to see what the rest of the shape does given the edge, and "none" leaves it free).
    Nothing else measured after the start; the coils are the machine's programme. The shape it predicts is scored against EFIT's."""

    def __init__(self, h, g, shot, k, interval=0.01, radial="programme", fast_share=1.0, Ip_prog=None, order=4, profile="torax", last_surface=LAST_SURFACE, smooth=1, phibdot=True):
        import freeboundary
        import mast_machine
        self.fb, self.h, self.r = freeboundary, h, h.reactor
        self.fast_share, self.order, self.profile, self.last_surface, self.smooth, self.phibdot = fast_share, order, profile, last_surface, smooth, phibdot
        self.Ip_prog = None if Ip_prog is None else np.asarray(Ip_prog, float)
        self.interval, self.k, self.radial, self.g = interval, k, radial, g
        c = np.load(os.path.join(OUT, "coils", f"{shot}.npz"))
        self.coil_t, self.coil = c["t"], {n: c[n] for n in ("P1", "P2", "P3", "P4", "P5")}
        self.c = freeboundary.Client(log=os.path.join(OUT, f"server_{os.getpid()}.log"))
        folder = os.path.join(mast.ROOT, "runs", "twin", "freeboundary", "mast_machine2")
        R, Z = g["efit_R"].astype(float), g["efit_z"].astype(float)
        self.c("build", machine="MAST", Rmin=float(R[0]), Rmax=float(R[-1]), Zmin=float(Z[0]), Zmax=float(Z[-1]), nx=len(R), ny=len(Z), paths=dict(active_coils_path=os.path.join(folder, "active.pickle"), passive_coils_path=os.path.join(folder, "passive.pickle"), limiter_path=os.path.join(folder, "limiter.pickle"), wall_path=os.path.join(folder, "wall.pickle")))
        t = float(g["t"][k])
        self.c("currents", values=self.coils_at(t))
        plasma = freeboundary.plasma_from_torax(self.r, order=self.order, p_fast=self.fast())
        psi = g["efit_psi"][:, :, k].astype(float)
        lc = np.c_[g["lcfs_r"][:, k], g["lcfs_z"][:, k]]
        lc = lc[np.isfinite(lc).all(axis=1)]
        self.R_out0 = float(lc[:, 0].max())
        self.vessel0 = self.c("vessel_start", R=R, Z=Z, t=c["t"], coils=self.coil, t_eq=c["t_eq"], j_tor=c["j_tor"], t_end=t)
        axis = (float(g["R_mag"][k]), float(g["Z_mag"][k]))
        if self.profile == "anchored":
            # EFIT's p' and FF' at the start, and from then on p' against FF' as the poloidal beta moves in TORAX: level with TORAX's own on transport, closer on elongation, further on the inner edge, on the 19 fit shots
            j = np.nan_to_num(c["j_tor"][:, :, int(np.argmin(np.abs(c["t_eq"] - t)))].astype(float))
            grid, pp, ffp, _ = mast_machine.fit_profiles(R, Z, psi, j, lc, axis)
            sign = float(np.sign(j.sum()))
            self.shape0 = dict(psi_n=grid, pprime=sign * pp, ffprime=sign * ffp)
            self.beta0 = self.beta_measure(plasma["Ip"])
            plasma = dict(plasma, **self.shape0)
        else:
            j = mast_machine.current_on_map(R, Z, psi, lc, axis, plasma["psi_n"], plasma["pprime"], plasma["ffprime"], plasma["Ip"])
        self.c("profiles", Ip=plasma["Ip"], fvac=plasma["fvac"], psi_n=plasma["psi_n"], pprime=plasma["pprime"], ffprime=plasma["ffprime"], Raxis=1.0)
        self.anchor = self.c("vessel_fit", psi=psi, j=j, R=R, Z=Z)
        self.c("save")
        self.res0 = self.c("forward", tol=1e-4, max_iter=60, symmetric=True)      # a static solve has no vertical control and drifts off (Z = 0.42 m on 24853), while MAST has it and its plasmas here sit within 3 cm of the midplane
        # the start's solve must be EFIT's plasma: on 18886 the static solve found another branch, a 4 cm plasma against the outer wall at R 1.71 m, TORAX took its
        # surfaces (the current density 400 times up, q 0.03, 19 GW of Ohmic heat in one step) and every later solve was refused against it (H86)
        efit0 = dict(a=0.5 * float(lc[:, 0].max() - lc[:, 0].min()), R_out=float(lc[:, 0].max()), R_in=float(lc[:, 0].min()))
        def sane(res):
            if res is None or res.get("lcfs") is None or res.get("axis") is None or res.get("q") is None:
                return False
            s_ = freeboundary.shape_of(res)
            return abs(s_["a"] - efit0["a"]) < 0.3 * efit0["a"] and abs(s_["R_out"] - efit0["R_out"]) < 0.1 and abs(s_["Z_axis"]) < 0.5
        self.anchor_note = "forward"
        if not sane(self.res0):
            self.c("back")
            try:                                               # the run's own solve, aimed at EFIT's outer and inner edge
                trial = self.c("inverse", R=[self.R_out0, efit0["R_in"]], Z=[0.0, 0.0], tol=1e-3, max_iter=40, l2_reg=1e-10, frozen=("P1", "P2", "P3", "P4", "P5"), symmetric=True)
            except RuntimeError:
                trial = None
            if sane(trial):
                self.res0, self.anchor_note = trial, "inverse"
            else:
                self.c("back")
                self.res0, self.anchor_note = None, "refused"
        self.log, self.rejected, self.wall = [], 0, 0.0
        self.t_next = t
        self.t_now = t
        self.off = self.res0 is None
        if self.off:                                           # no free boundary on this shot: TORAX keeps EFIT's surfaces from the start, and the results say so
            print(f"free boundary off: no start solve near EFIT's plasma (a {efit0['a']:.3f} m, outer edge {efit0['R_out']:.3f} m)", flush=True)
            self.s0 = dict(R0=0.5 * (efit0["R_out"] + efit0["R_in"]), a=efit0["a"], kappa=float(g["kappa"][k]), delta=0.5 * float(g["delta_u"][k] + g["delta_l"][k]), delta_lower=float(g["delta_l"][k]),
                           R_axis=float(g["R_mag"][k]), Z_axis=float(g["Z_mag"][k]), R_out=efit0["R_out"], R_in=efit0["R_in"])
            self.res, self.R_in, self.geo_now = None, efit0["R_in"], None
            return
        self.s0 = freeboundary.shape_of(self.res0)
        self.res = self.res0
        self.R_in = self.s0["R_in"]
        self.geo_now = freeboundary.geo_of(self.res0, self.r, last_surface=self.last_surface)
        self.r.geo_overrides = freeboundary.provider_between(self.t_now, self.geo_now, self.t_now + 10.0, self.geo_now)

    def fast(self):
        beam = self.h.plant.beam
        return None if beam is None or beam.W is None or not self.fast_share else self.fast_share * beam.pressure()

    def beta_measure(self, Ip):
        """Stored energy, thermal and fast, over the current squared: what moves p' against FF'."""
        beam = self.h.plant.beam
        W_fast = (beam.last or {}).get("W_fast", 0.0) if beam is not None else 0.0
        return (float(self.r.ppo.W_thermal_total) + W_fast) / max(Ip, 1e4) ** 2

    def coils_at(self, t):
        return {n: float(np.interp(t, self.coil_t, v)) for n, v in self.coil.items()}

    def target(self, t):
        if self.radial == "efit":
            return float(np.interp(t, self.g["t"], np.nanmax(self.g["lcfs_r"], axis=0)))
        if self.radial == "programme" and self.Ip_prog is not None:
            now, start, top = float(np.interp(t, self.g["t"], self.Ip_prog)), float(np.interp(self.g["t"][self.k], self.g["t"], self.Ip_prog)), float(np.max(self.Ip_prog))
            return self.R_out0 + (R_OUT_MAST - self.R_out0) * float(np.clip((now - start) / max(top - start, 1e3), 0.0, 1.0))
        return self.R_out0

    def __call__(self, r):
        t = r.t
        if getattr(self, "off", False) or t < self.t_next - 1e-9:
            return
        self.t_next = t + self.interval
        t0 = time.perf_counter()
        fb = self.fb
        plasma = fb.plasma_from_torax(r, order=self.order, p_fast=self.fast())
        if self.profile == "anchored":
            plasma = dict(plasma, psi_n=self.shape0["psi_n"], pprime=self.shape0["pprime"] * self.beta_measure(plasma["Ip"]) / max(self.beta0, 1e-30), ffprime=self.shape0["ffprime"])
        coils = self.coils_at(t)
        try:                                                   # a server lost or stopped (freeboundary.Client): the surfaces stay as they are for the rest of the run
            self.c("save")
            self.c("currents", values=coils)
            if t > self.t_now:
                self.c("vessel_advance", dt=t - self.t_now, coils=coils)
            self.c("profiles", Ip=plasma["Ip"], fvac=plasma["fvac"], psi_n=plasma["psi_n"], pprime=plasma["pprime"], ffprime=plasma["ffprime"], Raxis=1.0)
        except RuntimeError as err:
            print("server failed", repr(err)[:120], flush=True)
            self.rejected += 1
            self.wall += time.perf_counter() - t0
            return
        try:
            if self.radial == "none":
                trial = self.c("forward", tol=1e-3, max_iter=30, symmetric=True)
            else:                                              # the far coil is found inside the solve, for the outer edge and the last inner edge on one flux surface: around the solve, stepping the coil and solving again, the plasma leaves for other branches and does not come back
                trial = self.c("inverse", R=[self.target(t), self.R_in], Z=[0.0, 0.0], tol=1e-3, max_iter=40, l2_reg=1e-10, frozen=("P1", "P2", "P3", "P4", "P5"), symmetric=True)
                trial["feedback"] = trial["currents_active"]["BV"]
        except RuntimeError as err:                            # no flux surface it could work with (first met on 25007): the surfaces stay as they are until a later solve goes through
            print("solve failed", repr(err)[:120], flush=True)
            if not getattr(self.c, "dead", False):
                self.c("back")
            self.rejected += 1
            self.t_now = t
            self.wall += time.perf_counter() - t0
            return
        self.t_now = t
        ok = trial["lcfs"] is not None and trial["q"] is not None and trial.get("axis") is not None
        if ok:
            s_ = fb.shape_of(trial)
            ok = s_["a"] > 0.7 * self.s0["a"] and abs(s_["Z_axis"]) < 0.5 and (self.radial == "none" or abs(s_["R_out"] - self.target(t)) < 0.1) and (not self.log or abs(s_["kappa"] - self.log[-1]["kappa"]) < 0.3)      # static solves have other branches: a small plasma off to the inside, or one against the outer limiter. The limit on elongation came after the fresh test, for 18696, where it jumped between 2.4 and 3.5
        if not ok and os.environ.get("FREE_DUMP"):
            print("refused at", round(t, 4), "target", round(self.target(t), 3), "feedback", trial.get("feedback"), {k: round(v, 3) for k, v in fb.shape_of(trial).items()} if trial["lcfs"] is not None else None, flush=True)
        if not ok and os.environ.get("FREE_DUMP") and not getattr(self, "dumped", False):
            self.dumped = True
            import pickle
            pickle.dump(dict(t=t, plasma=plasma, coils=coils, target=self.target(t), state=self.c("dump_state"), trial_shape=fb.shape_of(trial) if trial["lcfs"] is not None else None), open(os.environ["FREE_DUMP"], "wb"))
        if ok:
            try:
                geo_new = fb.geo_of(trial, r, last_surface=self.last_surface)
                provider = fb.provider_between(t, self.geo_now, t + self.smooth * self.interval, geo_new, phibdot=self.phibdot)      # smooth > 1: the surfaces move only part of the way each interval, so the jitter of one solve does not drive the moving-flux term in TORAX's density equation
                r.geo_overrides = provider
                self.geo_now = provider(t + self.interval) if self.smooth > 1 else geo_new
                self.R_in = s_["R_in"]
                try:
                    whmd = fb.synthetic_whmd(trial)          # the store's whmd as it would read this equilibrium (H84, docs/lit/36)
                except Exception:
                    whmd = None
                self.log.append(dict(t=t, Ip=plasma["Ip"], feedback=trial.get("feedback", 0.0), whmd=whmd, **s_))
                self.res = trial
            except Exception as err:
                ok = False
                print("geometry failed", repr(err)[:160], flush=True)
        if not ok:
            if not getattr(self.c, "dead", False):
                self.c("back")
            self.rejected += 1
        self.wall += time.perf_counter() - t0

    def summary(self):
        """What the harness and the front end read, as for freeboundary.Loop; and EFIT's boundary now, to set beside it."""
        k = int(np.argmin(np.abs(self.g["t"] - self.r.t)))
        lc = np.c_[self.g["lcfs_r"][:, k], self.g["lcfs_z"][:, k]]
        lc = lc[np.isfinite(lc).all(axis=1)]
        if getattr(self, "off", False):
            return dict(mode="off: no start solve near EFIT's plasma, EFIT's surfaces held", solves=0, rejected=self.rejected, wall_s=self.wall, shape=self.s0, coils_A=self.coils_at(self.r.t), coils_moved_A={}, lcfs=np.round(lc[::4], 3).tolist(), measured_lcfs=np.round(lc[::2], 3).tolist(), stability=None)
        return dict(mode="the shot's coils, the outer edge from the programme", solves=len(self.log), rejected=self.rejected, wall_s=self.wall, shape=self.fb.shape_of(self.res), coils_A=self.coils_at(self.r.t), coils_moved_A={}, lcfs=np.round(self.res["lcfs"][::4], 3).tolist(), measured_lcfs=np.round(lc[::2], 3).tolist(), stability=None)

    def close(self):
        if self in self.r.before_step:
            self.r.before_step.remove(self)
        self.c.close()


def rotation_at_start(g, e, k):
    """v_i / 2 pi R on the outboard side at q = 2, over the last slices. None if the chords miss it."""
    if "v_i" not in e:
        return None
    vals = []
    for j in range(max(k - 4, 0), k + 1):
        below = np.where(g["q_psi"][j] <= 2)[0]
        if len(below) == 0:                            # early in a shot q is above 2 everywhere. Tried: the chords at rho 0.55 instead, and 6 of the 19 fit shots then read under 1 kHz, so their modes count as locked from the first moment
            continue
        r2 = g["rho_psi"][j][below[-1]]
        c = g["cx_rho"][j]
        near = np.isfinite(c) & (np.abs(c - r2) < 0.1) & (e["R_cx"] > g["R_mag"][j])
        i = int(np.argmin(np.abs(e["t_cx"] - g["t"][j])))
        if near.sum() >= 2:
            v = np.abs(np.nanmedian(e["v_i"][near, i]))
            if np.isfinite(v) and v > 1e3:
                vals.append(v / (2 * np.pi * np.median(e["R_cx"][near])))
    return float(np.median(vals)) if vals else None


def build_run(shot, models, seed, laws, overrides=None, given_h=False, s_critical=0.25, mixing=1.3, k=None, keep=True):
    """A hands-off run ready to step: the harness with the laws and, if asked, FreeGSNKE on it."""
    import mast_anchor
    import ntm
    import twin
    path = os.path.join(mast.ROOT, SHOTS, f"{shot}.npz")
    truth, g, info = mast.load_shot(path)
    e = dict(np.load(os.path.join(EXTRAS, f"{shot}.npz")))
    mast.START_LINE = (overrides or {}).get("start") == "line"
    k = mast.FIXED["trailing_slices"] - 1 if k is None else k
    t_cut = float(truth.t[k])
    x = catalog_shot(shot)
    hwin = [(w["t0"], w["t1"]) for w in x["phases"]["hmode"]["windows"]]
    h0 = any(a <= t_cut <= b for a, b in hwin)
    elm_t = np.array(x.get("elms", {}).get("t") or [])
    before = elm_t[(elm_t >= t_cut - 0.03) & (elm_t <= t_cut)]
    rate0 = 1 / float(np.median(np.diff(before))) if len(before) >= 4 else None
    holder = {}
    # the programmed current, scaled to the measured one at the start, instead of the measured current
    Ip_prog = np.interp(truth.t, e["t_prog"], np.nan_to_num(e["prog_i_plasma"]))
    scale = float(np.mean(truth.Ip[max(k - 1, 0):k + 1])) / max(float(np.interp(t_cut, e["t_prog"], np.nan_to_num(e["prog_i_plasma"]))), 1e4)
    s = dict(name=f"MAST {shot}", mast=dict(shot=shot, model=(overrides or {}).get("base", "bgb"), past=0.0, natural=True, cut=t_cut, shots=SHOTS, Ip=(Ip_prog * scale).tolist(), anchor_rates=(overrides or {}).get("anchor_rates"), anchor_bounds=(overrides or {}).get("anchor_bounds", {}), **({"table": np.ones_like(ntm.TURBULENCE).tolist()} if (overrides or {}).get("base", "bgb") in ("tglfnn_step", "tglfnn_multi") else {})), dt=1e-3, edge=False, rotation=False, impurities={"C": 0.0167, "Ne": 0.0, "Ar": 0.0}, physics={"a_bs": (overrides or {}).get("a_bs", 1.5), "a_cd": 0.5},
             gyrotrons={"units": 4, "unit_MW": 0.25, "mirror_rate": 2.0}, sensors={"sigma_w_cm": 0.0, "sigma_rho_cm": 0.0, "latency_steps": 0, "seed": 1}, decision={"kind": "none"}, events=[], modes=[], beams=((overrides or {}).get("beams", True) if "beams" in models else False))
    if "elms" in models:
        s["elms"] = dict(law=lambda r, t: holder["n"].elm(r, t), seed=shot % 1000 + seed)
    else:
        s["elms"] = False
    if "sawtooth" in models:
        s["sawtooth"] = dict(trigger_model=dict(model_name="simple", minimum_radius=0.15, s_critical=s_critical), redistribution_model=dict(model_name="simple", flattening_factor=1.001, mixing_radius_multiplier=mixing), crash_step_duration=1e-4)
    if "modes" in models:
        m = dict(laws["mode"], **(overrides or {}))
        s["rotation"] = dict(f0=10000.0, tau_M0=m["tau_M0"], tau_A0=m["tau_A0"], tau_w=m["tau_w"], w_vac=m["w_vac"], dt=1e-5, quench_rate=-np.log(1 - m["quench_within_20ms"]) / 0.02, f_lock=m["f_lock"], **{a: m[a] for a in ("C_M", "C_1") if a in m})
        s["events"] = [dict(type="island", t=1e3, m=2, n=1, w_cm=1.0)]             # never reached: names the mode
    t0 = time.perf_counter()
    h = twin.Harness(s, keep_history=keep)
    r = h.reactor
    table = h.table0.copy()
    nat = Natural(h, g, e, laws, shot, k, seed, models, given_h=hwin if given_h else None, h0=h0, elm_rate0=rate0, overrides=overrides)
    nat.rot_start = rotation_at_start(g, e, k)
    nat.s_critical = s_critical
    if (overrides or {}).get("closure"):
        fold = (overrides or {}).get("closure_fold")          # leave-half-out on the fit shots, for choosing whether to use it
        fit = split("fit")
        nat.closure = fit_closure2(None if fold is None else [a for i, a in enumerate(fit) if i % 2 != fold])[0]
        nat.table_start = h.table0.copy()
    holder["n"] = nat
    r.before_step.append(nat)
    r.retries = 3
    free = None
    if "free" in models:
        import dataclasses as dc
        import jax.numpy as jnp
        free = FreeBoundary(h, g, shot, k, interval=(overrides or {}).get("free_interval", 0.01), radial=(overrides or {}).get("radial", "programme"), fast_share=(overrides or {}).get("fast_share", 1.0), Ip_prog=Ip_prog * scale, order=(overrides or {}).get("free_order", 4), profile=(overrides or {}).get("free_profile", "torax"), last_surface=(overrides or {}).get("last_surface", LAST_SURFACE), smooth=(overrides or {}).get("free_smooth", 1), phibdot=(overrides or {}).get("free_phibdot", True))
        r.before_step.append(free)
        if (overrides or {}).get("free_anchor"):               # balance the transport again on these surfaces. Tried on 6 fit shots, to see if the plasma was relaxing from EFIT's surfaces to these: no change, n_e 15.6, 40.3 against 15.7, 40.5 at 20, 100 ms
            geo = free.geo_now
            r.state = dc.replace(r.state, geometry=dc.replace(geo, Phi_b_dot=jnp.asarray(0.0, dtype=jnp.float64)))
            rates = None
            if s["mast"].get("anchor_rates"):
                rates = mast_anchor.measured_rates(h.measured, r, channels=s["mast"]["anchor_rates"])
            mast_anchor.anchor(r, rates=rates, geo_overrides=r.geo_overrides, **((overrides or {}).get("anchor_bounds") or {}))
            h.table0 = ntm.TURBULENCE.copy()
            if (overrides or {}).get("closure"):
                nat.table_start = h.table0.copy()
    return dict(h=h, nat=nat, free=free, truth=truth, info=info, k=k, t_cut=t_cut, h0=h0, rate0=rate0, t0=t0)


def run_shot(shot, models, seed, laws, tag, overrides=None, given_h=False, s_critical=0.25, mixing=1.3, k=None, keep=True):
    import shot as shot_lib
    import mast_anchor
    if (overrides or {}).get("general"):
        import coupled
        b = coupled.build_shot(shot, models, seed, overrides, s_critical, mixing, k, keep)
    else:
        b = build_run(shot, models, seed, laws, overrides, given_h, s_critical, mixing, k, keep)
    h, nat, free, truth, info, k, t_cut, h0, rate0, t0 = (b[a] for a in ("h", "nat", "free", "truth", "info", "k", "t_cut", "h0", "rate0", "t0"))
    r = h.reactor
    rows = []
    keys = ("t", "W_thermal", "P_SOL", "P_rad", "P_LH", "elm", "sawtooth", "disrupted", "w_2_1", "f_2_1", "rho_2_1", "locked_2_1", "beam_W_fast", "beam_neutrons", "beam_through", "beam_P_abs", "T_e0", "T_i0", "n_e0", "n_line", "beta_N", "q95", "q0", "q_min", "H", "fT", "fn", "f0", "Ip_now", "li3", "rho_q1", "s_q1", "P_net", "P_LH_twin", "tau_E", "beta_kin", "beta_over_limit", "x_crit", "collapse", "g_law", "T_top", "n_top", "line_target",
            # readings added in H86: the n = 1 internal mode, the bolometers' terms and total (radiation.py), the scrape-off layer (sol.py), KN1D's neutrals, the core's Ohmic power and Z_eff
            "kink_n1", "rad_core", "rad_sol_c", "rad_h_main", "rad_h_div", "rad_bolo", "rad_G_target", "rad_T_t", "rad_e_ly", "sol_n_sep", "sol_T_sep", "sol_T_t", "sol_G_out", "sol_G_target",
            "kn1d_source", "kn1d_gauge", "kn1d_inside", "kn1d_ly", "P_ohmic", "P_brems", "Z_eff_avg")
    while r.step() is not None:
        rows.append({a: r.records[-1].get(a) for a in keys if a in r.records[-1]})
        if not keep:
            del r.records[:-2]
    res = dict(shot=shot, seed=seed, tag=tag, models=list(models), k=int(k), t_cut=t_cut, t_end=float(truth.t[-1]), sim_error=str(r.sim_error), disrupted=bool(r.disrupted), steps=len(rows), wall_s=round(time.perf_counter() - t0, 1), h0=bool(h0), elm_rate0=rate0, rot_start=nat.rot_start, f_rot_source=getattr(nat, "f_rot_source", None), table_median=[float(np.median(v)) for v in mast_anchor.ntm_table()],
               elm_times=[round(float(v), 5) for v in r.elm_times], h_log=[(round(a, 4), b) for a, b in nat.h_log], seeds=nat.seeds, overrides={a: b for a, b in (overrides or {}).items()}, s_critical=s_critical, mixing=mixing, given_h=bool(given_h), n_retried=r.n_retried)
    if hasattr(nat, "collapses"):
        res["collapses"] = nat.collapses
    if free is not None:
        free.close()
        res["free"] = dict(log=free.log, rejected=free.rejected, wall_s=round(free.wall, 1), anchor=free.anchor, s0=free.s0, start=getattr(free, "anchor_note", "forward"))
    if keep and rows:
        tree, _ = r.finish()
        sc = shot_lib.score(mast.scored_shot(truth), tree, t_cut)
        res["profile"] = mast.summarize(sc, t_cut, info["tau_E_s"])
        res["held"] = mast_anchor.held(truth, k, info["tau_E_s"])
        if (overrides or {}).get("save_profiles"):      # the profiles themselves, every 2 ms: level and shape can then be scored apart (profile_split.py)
            p = tree["profiles"].to_dataset()
            every = max(int(round(0.002 / max(float(np.median(np.diff(p["time"].values))), 1e-6))), 1)
            np.savez_compressed(os.path.join(OUT, tag, f"{shot}_{seed}_prof.npz"), t=p["time"].values[::every], rho=p["rho_norm"].values, T_e=p["T_e"].values[::every], T_i=p["T_i"].values[::every], n_e=p["n_e"].values[::every])
    return res, rows


VERSIONS = {          # the settings of the clean tests in docs/digital-twin.md
    "2.2": dict(models=("beams", "pedestal", "hmode", "elms", "sawtooth", "gate", "modes", "free"), overrides={"start": "line", "closure": True, "pedestal": "eped", "elm_law": "energy", "density": "N_source"}, s_critical=0.55, mixing=1.1),
    "1": dict(models=("beams", "pedestal", "hmode", "elms", "sawtooth", "gate", "modes"), overrides={"start": "line", "closure": True, "beams": "frozen"}, s_critical=0.55, mixing=1.1),
    "4": dict(models=("beams", "hmode", "elms", "sawtooth", "modes", "collapse", "free"), overrides={"general": True, "density_feedback": True, "follow": {"tau": 0.005, "shape": "wall", "length": 0.1}}, s_critical=0.25, mixing=1.3),     # version 3 with the line density held on its programme by a source (H54, H56). Tested on a fixed boundary only, and not yet on shots kept clean for it
    "5": dict(models=("beams", "hmode", "elms", "sawtooth", "modes", "collapse", "free"), overrides={"general": True, "density_feedback": True, "follow": {"tau": 0.005, "shape": "wall", "length": 0.1}, "physics": {"a_bs": 1.5, "a_cd": 0.5, "a_ggj": 1.0 / 6, "k_pol": 2.5}, "C_M": 3.0}, s_critical=0.25, mixing=1.3),     # version 4 with the island of the check against shots: curvature at 1.0 D_R (La Haye 2012), a threshold of 2.5 ion banana widths, the rotation's viscous term at 3 (docs/digital-twin.md, H59 to H61)
    "6": dict(models=("beams", "hmode", "elms", "sawtooth", "modes", "collapse", "free"), overrides={"general": True, "density_feedback": True, "follow": {"tau": 0.005, "shape": "wall", "length": 0.1}, "physics": {"a_bs": 1.5, "a_cd": 0.5, "a_ggj": 1.0 / 6, "k_pol": 2.5, "w_cap": True}, "C_M": 3.0}, s_critical=0.25, mixing=1.3),     # version 5 with the island kept inside the plasma (H69)
    "3": dict(models=("beams", "hmode", "elms", "sawtooth", "modes", "collapse", "free"), overrides={"general": True}, s_critical=0.25, mixing=1.3),      # the general coupled plasma: no law fitted to any machine. TORAX's own test settings for the crash
    "7": dict(models=("beams", "hmode", "elms", "sawtooth", "modes", "collapse", "free"), overrides={"general": True, "edge_start": True, "lmode_edge": True, "beam_history": True, "beam_current": False},
              s_critical=0.25, mixing=1.3),      # (version 8 below adds the neutron rate of the fast ions present) version 3 with what H83 fixed: the start from slices whose Thomson reaches the edge, no transport barrier in L-mode, the fast ions filled from the measured beam history, and the beam current out of the current equation until fast-ion redistribution is in (docs/digital-twin.md, H83)
    "8": dict(models=("beams", "hmode", "elms", "sawtooth", "modes", "collapse", "free"), overrides={"general": True, "edge_start": True, "lmode_edge": True, "beam_history": True, "beam_current": False, "neutron_model": "population"},
              s_critical=0.25, mixing=1.3),      # version 7 with neutrons from the fast ions present (H84); every free-boundary solve also logs the synthetic whmd, the chip project's stored-energy reading
    "9": dict(models=("beams", "hmode", "elms", "sawtooth", "modes", "collapse", "free"), overrides={"general": True, "edge_start": True, "lmode_edge": True, "beam_history": True, "beam_current": False, "neutron_model": "population",
                                                                                                     "noncoronal": True, "limit": "piccione", "porcelli": True},
              s_critical=0.25, mixing=1.3),      # version 8 with carbon out of coronal balance, Piccione's no-wall limit below R/a 2 and Porcelli's thermal sawtooth trigger (H84, tags v9sa to v9dc): continuous effects as version 8; crashes 297 to 62 against 105 measured (the chatter gone, frequent real sawteeth now undercounted), island time lower. The one to use for hands-off MAST runs after H84 (hands_off() still defaults to 3, which the checks use)
    "10": dict(models=("beams", "hmode", "elms", "sawtooth", "modes", "collapse", "free"), overrides={"general": True, "edge_start": True, "lmode_edge": True, "beam_history": True, "beam_current": False, "neutron_model": "population",
                                                                                                      "noncoronal": True, "limit": "piccione", "porcelli": True,
                                                                                                      "beam_options": {"stopping": "suzuki", "cx": "halo", "coulomb": "nrl", "orbit_average": True, "orbit_width": "pphi", "species": [0.89, 0.08, 0.03]}},
               s_critical=0.25, mixing=1.3),     # version 9 with the beams rebuilt from published physics (H84, tags v10sa to v10dc): on measured profiles the beam model's fast-ion energy 2.04 to 1.69 times EFIT's and neutrons from a fitted energy to physics; in the coupled twin line density, thermal energy and ELMs better, neutrons, shape and stored energy worse, and 11% more absorbed power in the ramp digs a hollow current on 5 of 25 shots (3 self-disruptions against 1). Not recommended until the current profile in the ramp is fixed
    "11": dict(models=("beams", "hmode", "elms", "sawtooth", "modes", "collapse", "free"), overrides={"general": True, "edge_start": True, "edge_cap": True, "lmode_edge": True, "beam_history": True, "beam_current": False, "neutron_model": "population",
                                                                                                      "noncoronal": True, "limit": "piccione", "porcelli": True, "bolometer": True},
               s_critical=0.25, mixing=1.3),     # version 9 with the readings of H86 (tags v11a to v11e): the bolometers' total from a radiation budget (radiation.py; radiated power 95 to 88% low, a median 0.12 of prad_pol), the n = 1 internal mode (kink_n1), the free boundary's start checked against EFIT (12 of 25 starts had been on another branch), and edge_cap on the start. The plasma is version 9's: every other effect level on the same starts. The one to use after H86
    "12": dict(models=("beams", "hmode", "elms", "sawtooth", "modes", "collapse", "free"), overrides={"general": True, "edge_start": True, "edge_cap": True, "lmode_edge": True, "beam_history": True, "beam_current": False, "neutron_model": "population",
                                                                                                      "noncoronal": True, "limit": "piccione", "porcelli": True, "bolometer": True,
                                                                                                      "density_feedback": True, "follow": {"tau": 0.005, "shape": "profile", "integral": 0.02, "no_sink": True}, "particle_law": "published"},
               s_critical=0.25, mixing=1.3),     # version 11 with the line density held on its programme (MAST's programmed density, else the start's Greenwald fraction) by an add-only source shaped like the density profile with integral action, and the published particle transport kept at TORAX's D_e_min (H87, tags v19a to v19e): line density 22.5 to 13.0%, core T_e 14.5 to 12.9%, radiated power 88 to 76% low, every run to its end (18464 still disrupts on its own 41 ms early); the shape a little worse (axis 2.0 to 2.3 cm, elongation 0.074 to 0.086). The one to use after H87
}


RECOMMENDED = "12"     # the version hands_off, the front end, the checks and the howto use by default (H86); change it here only after readiness.py says the new one is better



def development_shots():
    """The shots the twin may run and score when the workspace keeps a list, runs/twin/dev_shots.json ({"development": [shot, ...]}), so that
    shots held out for a later test are never run by mistake. None when there is no list: then any shot with files may run."""
    path = os.path.join(mast.ROOT, "runs", "twin", "dev_shots.json")
    return set(json.load(open(path))["development"]) if os.path.exists(path) else None


def check_development(shots, allow=False):
    """Refuse any shot off the development list when there is one (README.md, "MAST shots"). allow is for reproducing runs made before a list existed."""
    if allow:
        return
    dev = development_shots()
    if dev is None:
        return
    bad = [int(s) for s in shots if int(s) not in dev]
    if bad:
        raise SystemExit(f"refused, not development shots (runs/twin/dev_shots.json): {bad}")

def hands_off(shot, seed=0, version=None):
    """The run `natural.py run` makes, handed over unstepped, for the front end or a script: returns the harness, with FreeGSNKE on it as h.free. Versions 3 and up have no law fitted to MAST; 2.2 and 1 do.
    version None (or "recommended") is RECOMMENDED."""
    version = RECOMMENDED if version in (None, "recommended") else str(version)
    check_development([shot])
    v = VERSIONS[version]
    if v["overrides"].get("general"):
        import coupled
        b = coupled.build_shot(shot, v["models"], seed, v["overrides"], v["s_critical"], v["mixing"], None, False)
    else:
        b = build_run(shot, v["models"], seed, json.load(open(os.path.join(OUT, "laws.json"))), v["overrides"], False, v["s_critical"], v["mixing"], None, False)
    h = b["h"]
    h.free, h.natural = b["free"], b["nat"]
    return h


def free_boundary_now(h, interval=0.01):
    """The harness's free_boundary command on a MAST shot: FreeGSNKE from this moment, on the shot's coil traces, EFIT's map nearest to now and its programmed current."""
    shot, g = h.measured["shot"], h.measured["g"]
    if not os.path.exists(os.path.join(OUT, "coils", f"{shot}.npz")):
        raise ValueError(f"no coil traces for shot {shot}: mast_machine.py makes them")
    if "efit_psi" not in g:
        _, g, _ = mast.load_shot(os.path.join(mast.ROOT, SHOTS, f"{shot}.npz"))
    t, r = np.asarray(g["t"]), h.reactor
    k = int(np.argmin(np.abs(t - r.t)))
    e = np.load(os.path.join(EXTRAS, f"{shot}.npz"))
    Ip_prog = np.interp(t, e["t_prog"], np.nan_to_num(e["prog_i_plasma"]))
    free = FreeBoundary(h, g, shot, k, interval=interval, Ip_prog=Ip_prog * float(np.mean(g["Ip"][max(k - 1, 0):k + 1])) / max(float(Ip_prog[k]), 1e4))
    r.before_step.append(free)
    return free


_CAT = None


def catalog_shot(shot):
    global _CAT
    if _CAT is None:
        _CAT = catalog()
    return _CAT[str(shot)]


def run(args):
    laws = json.load(open(os.path.join(OUT, "laws.json")))
    folder = os.path.join(OUT, args.tag)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "results.jsonl")
    done = {(json.loads(line)["shot"], json.loads(line)["seed"]) for line in open(path)} if os.path.exists(path) else set()
    log = open(path, "a")
    shots = args.shots or split(args.split)
    check_development(shots, allow=getattr(args, "allow_non_dev", False))
    overrides = json.loads(args.overrides) if args.overrides else None
    for shot in shots:
        for seed in range(args.seeds):
            if (shot, seed) in done:
                continue
            try:
                res, rows = run_shot(shot, tuple(args.models), seed, laws, args.tag, overrides, args.given_h, args.s_critical, args.mixing)
            except Exception as err:
                import traceback
                traceback.print_exc()
                print(shot, seed, "failed", repr(err)[:300], flush=True)
                continue
            log.write(json.dumps(res) + "\n")
            log.flush()
            if rows:
                np.savez_compressed(os.path.join(folder, f"{shot}_{seed}.npz"), **{a: np.array([x.get(a, np.nan) if x.get(a) is not None else np.nan for x in rows], float) for a in set().union(*rows)})
            crashes = sum(1 for x in rows if x.get("sawtooth"))
            print(shot, seed, f"{res['steps']} steps {res['wall_s']} s {res['sim_error']}; elms {len(res['elm_times'])}, crashes {crashes}, seeds {len(res['seeds'])}, H changes {len(res['h_log'])}, disrupted {res['disrupted']}", flush=True)


def shapes(args):
    """Shape against EFIT, against holding EFIT's shape at the start, by lead, for runs with the free boundary. Reads the fit from the shot files."""
    folder = os.path.join(OUT, args.tag)
    rows = [json.loads(line) for line in open(os.path.join(folder, "results.jsonl"))]
    leads = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.3])
    table = {k: {L: [] for L in leads} for k in ("R_out", "R_in", "kappa", "Z_axis", "R_axis")}
    held = {k: {L: [] for L in leads} for k in table}
    for r in rows:
        if "free" not in r or not r["free"]["log"]:
            continue
        g = dict(np.load(os.path.join(mast.ROOT, SHOTS, f"{r['shot']}.npz"), allow_pickle=False))
        t = g["t"]
        efit = dict(R_out=np.nanmax(g["lcfs_r"], axis=0), R_in=np.nanmin(g["lcfs_r"], axis=0), kappa=g["kappa"], Z_axis=g["Z_mag"], R_axis=g["R_mag"])
        lt = np.array([x["t"] for x in r["free"]["log"]])
        t0 = r["t_cut"]
        for L in leads:
            tt = t0 + L
            if tt > min(lt[-1], t[-1]):
                continue
            for k in table:
                v = np.interp(tt, lt, [x[k] for x in r["free"]["log"]])
                real = np.interp(tt, t, efit[k])
                table[k][L].append(abs(v - real))
                held[k][L].append(abs(np.interp(t0, t, efit[k]) - real))
    print(f"{args.tag}: {len(rows)} runs; rejected solves {sum(r['free']['rejected'] for r in rows if 'free' in r)}; wall in the solver per run, median {np.median([r['free']['wall_s'] for r in rows if 'free' in r]):.0f} s")
    for k in table:
        scale = 100 if k != "kappa" else 1
        unit = "cm" if k != "kappa" else ""
        print(f"  {k:7s} median |error| at {[int(1e3 * x) for x in leads]} ms ({unit}): run " + ", ".join(f"{scale * np.median(table[k][L]):.1f}" if table[k][L] else "-" for L in leads) + "; held " + ", ".join(f"{scale * np.median(held[k][L]):.1f}" if held[k][L] else "-" for L in leads) + f"; n {[len(table[k][L]) for L in leads]}")


def cost(args):
    """The island's own effect: each run against the same shot without islands, with no other chance in it (no H-mode changes, no ELMs, no sawteeth), defined as the catalog's excess: mean energy over 10 ms from saturation over the mean of 15 ms before onset, minus the same ratio in the reference. Also lock against width, rise, onset to lock."""
    laws = json.load(open(os.path.join(OUT, "laws.json")))
    folder = os.path.join(OUT, args.tag)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "cost.jsonl")
    done = {(json.loads(line)["shot"], json.loads(line)["seed"]) for line in open(path)} if os.path.exists(path) else set()
    log = open(path, "a")
    over = json.loads(args.overrides) if args.overrides else {}
    for shot in args.shots or split(args.split):
        _, ref = run_shot(shot, ("beams", "pedestal"), 0, laws, args.tag, over, keep=False)
        tr, Wr = np.array([x["t"] for x in ref]), np.array([x["W_thermal"] for x in ref])
        a = float(np.load(os.path.join(mast.ROOT, SHOTS, f"{shot}.npz"), allow_pickle=False)["a"][4])
        for seed in range(args.seeds):
            if (shot, seed) in done:
                continue
            try:
                _, rows = run_shot(shot, ("beams", "pedestal", "modes"), seed, laws, args.tag, dict(over, seed_factor=args.seed_factor), keep=False)
            except Exception as err:
                print(shot, seed, "failed", repr(err)[:200], flush=True)
                continue
            t, W = np.array([x["t"] for x in rows]), np.array([x["W_thermal"] for x in rows])
            w = np.array([x.get("w_2_1") or 0.0 for x in rows])
            f = np.array([x.get("f_2_1") if x.get("f_2_1") is not None else 1e9 for x in rows])
            out = []
            for isl in onsets(w, f, t, 0.02):
                i0 = int(np.searchsorted(t, isl["t_on"]))
                seg = w[i0:]
                sat = i0 + int(np.argmax(seg >= 0.9 * isl["w_max"]))
                ts = t[sat]
                mean = lambda tt, v, x0, x1: float(np.mean(v[(tt >= x0) & (tt < x1)])) if ((tt >= x0) & (tt < x1)).any() else np.nan
                ratio = mean(t, W, ts, ts + 0.01) / mean(t, W, isl["t_on"] - 0.015, isl["t_on"])
                ratio_ref = mean(tr, Wr, ts, ts + 0.01) / mean(tr, Wr, isl["t_on"] - 0.015, isl["t_on"])
                out.append(dict(w_over_a=isl["w_max"] / a, excess=ratio - ratio_ref, locked=isl["t_lock"] is not None, rise_ms=1e3 * (ts - isl["t_on"]), lock_ms=None if isl["t_lock"] is None else 1e3 * (isl["t_lock"] - isl["t_on"]), f_on=isl["f_on"], t_on=isl["t_on"], dur_ms=1e3 * (isl["t_end"] - isl["t_on"])))
            log.write(json.dumps(dict(shot=shot, seed=seed, islands=out, disrupted=bool(rows and rows[-1].get("disrupted")), t_stop=float(t[-1]))) + chr(10))
            log.flush()
            print(shot, seed, len(out), "islands", [(round(x["w_over_a"], 3), round(100 * x["excess"], 1), x["locked"], round(x["rise_ms"]), x["lock_ms"]) for x in out][:4], flush=True)
    cost_table(path)


def cost_table(path):
    I = [x for line in open(path) for x in json.loads(line)["islands"]]
    print(f"{len(I)} islands. Catalog (w/a inferred): W_th excess -9.9% at 0.15 to 0.2, -29.4% at 0.2 to 0.3; locked 25%, 15%, 59%, 85% in the bins 0 to 0.1, 0.1 to 0.15, 0.15 to 0.2, 0.2 to 0.3; rise median 16 ms; onset to lock median 39 ms (R1), 54 ms (clean)")
    for lo, hi in ((0, 0.1), (0.1, 0.15), (0.15, 0.2), (0.2, 0.3), (0.3, 1)):
        g = [x for x in I if lo <= x["w_over_a"] < hi]
        if g:
            print(f"  w/a {lo} to {hi}: n {len(g)}, W_th excess median {100 * np.nanmedian([x['excess'] for x in g]):.1f}% (IQR {100 * np.nanpercentile([x['excess'] for x in g], 25):.1f}, {100 * np.nanpercentile([x['excess'] for x in g], 75):.1f}), locked {np.mean([x['locked'] for x in g]):.2f}")
    if I:
        lk = [x["lock_ms"] for x in I if x["lock_ms"] is not None]
        print(f"  rise median {np.median([x['rise_ms'] for x in I]):.0f} ms; onset to lock median {np.median(lk) if lk else np.nan:.0f} ms, IQR {np.percentile(lk, 25) if lk else np.nan:.0f} to {np.percentile(lk, 75) if lk else np.nan:.0f}; locked {np.mean([x['locked'] for x in I]):.2f}; f at onset median {np.median([x['f_on'] for x in I]) / 1e3:.1f} kHz")


def steps(args):
    """The events at sharp steps in beam power (mast_anchor.transients), with the frozen settings and no chance in them: beams, pedestal, closure. Against the held profile; the earlier table is in docs/digital-twin.md."""
    global SHOTS
    import mast_anchor
    laws = json.load(open(os.path.join(OUT, "laws.json")))
    over = json.loads(args.overrides) if args.overrides else {}
    SHOTS = os.path.relpath(os.path.join(mast_anchor.OUT, "shots"), mast.ROOT)
    events = [e for e in json.load(open(os.path.join(mast_anchor.OUT, "transients.json")))["events"] if e["shot"] != 27714]
    folder = os.path.join(OUT, args.tag)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "steps.jsonl")
    rows = [json.loads(line) for line in open(path)] if os.path.exists(path) else []
    done = {r["shot"] for r in rows}
    log = open(path, "a")
    for e in events:
        f = os.path.join(mast.ROOT, SHOTS, f"{e['shot']}.npz")
        if e["shot"] in done or not os.path.exists(f) or not os.path.exists(os.path.join(EXTRAS, f"{e['shot']}.npz")):
            continue
        truth, _, _ = mast.load_shot(f)
        k = max(int(np.searchsorted(truth.t, e["t_step"] - 0.002)) - 1, mast.FIXED["trailing_slices"] - 1)
        try:
            res, _ = run_shot(e["shot"], ("beams", "pedestal"), 0, laws, args.tag, over, k=k)
        except Exception as err:
            print(e["shot"], "failed", repr(err)[:300], flush=True)
            continue
        res["step_W"] = e["step_W"]
        rows.append(res)
        log.write(json.dumps(res) + chr(10))
        log.flush()
        print(e["shot"], f"{e['step_W'] / 1e6:+.2f} MW", res["sim_error"], flush=True)
    leads = mast.FIXED["leads_s"]
    for sign, name in ((1, "power up"), (-1, "power down")):
        g = [r for r in rows if np.sign(r["step_W"]) == sign]
        print(f"{name}: {len(g)} events; median error (%) at 20, 30, 50, 75 ms, and events won against the held profile")
        for c in ("T_e", "T_i", "n_e"):
            for label, pick in (("held", lambda r: r["held"]["err_at_s"]), ("run", lambda r: r["profile"]["err_at_s"])):
                line = []
                for L in (0.02, 0.03, 0.05, 0.075):
                    i = leads.index(L)
                    v = [pick(r)[c][i] for r in g if c in pick(r) and pick(r)[c][i] is not None]
                    wins = ""
                    if label == "run":
                        p = [(r["profile"]["err_at_s"][c][i], r["held"]["err_at_s"][c][i]) for r in g if c in r["profile"]["err_at_s"] and r["profile"]["err_at_s"][c][i] is not None and r["held"]["err_at_s"][c][i] is not None]
                        wins = f" ({sum(a < b for a, b in p)}/{len(p)})"
                    line.append(f"{100 * np.median(v):5.1f}{wins}" if v else "  -  ")
                print(f"  {c} {label}: " + "  ".join(line))


def closure_data(args):
    """Anchored multipliers at several times in each fit shot, for how transport changes with the state beyond what the base model says. Every 40 ms, from the measured state then."""
    import ntm
    import twin
    path = os.path.join(OUT, "closure.jsonl")
    done = {(json.loads(line)["shot"], json.loads(line)["k"]) for line in open(path)} if os.path.exists(path) else set()
    log = open(path, "a")
    for shot in args.shots or split("fit"):
        truth, g, _ = mast.load_shot(os.path.join(mast.ROOT, SHOTS, f"{shot}.npz"))
        windows = catalog_shot(shot)["phases"]["hmode"]["windows"]
        t = truth.t
        for k in range(mast.FIXED["trailing_slices"] - 1, len(t) - 2, 8):
            if (shot, k) in done:
                continue
            s = dict(name=f"MAST {shot}", mast=dict(shot=shot, model="bgb", past=0.0, natural=True, cut=float(t[k]), shots=SHOTS), dt=1e-3, edge=False, rotation=False, elms=False, impurities={"C": 0.0167, "Ne": 0.0, "Ar": 0.0}, physics={"a_bs": 1.5, "a_cd": 0.5}, gyrotrons={"units": 4, "unit_MW": 0.25, "mirror_rate": 2.0}, sensors={"sigma_w_cm": 0.0, "sigma_rho_cm": 0.0, "latency_steps": 0, "seed": 1}, decision={"kind": "none"}, events=[], modes=[], beams=True)
            try:
                h = twin.Harness(s, keep_history=False)
            except Exception as err:
                print(shot, k, "failed", repr(err)[:200], flush=True)
                continue
            mean = lambda v: float(np.mean(v[max(k - 4, 0):k + 1]))
            log.write(json.dumps(dict(shot=shot, k=k, t=float(t[k]), nel=mean(g["nel"]), Ip=mean(g["Ip"]), P=max(mean(g["P_heat"]), 0.05), H=float(any(w["t0"] <= t[k] <= w["t1"] for w in windows)), table=np.asarray(h.table0).round(4).tolist())) + chr(10))
            log.flush()
            print(shot, k, flush=True)


def fit_closure2(shots=None):
    """Within a shot, the change of log multiplier against the change of log density, log current, log power and H-mode, per channel in six radial zones. Returns coefficients on the turbulence grid."""
    import ntm
    rows = [json.loads(line) for line in open(os.path.join(OUT, "closure.jsonl"))]
    if shots is not None:
        rows = [r for r in rows if r["shot"] in shots]
    X, Y = [], []
    for s in sorted({r["shot"] for r in rows}):
        g = sorted([r for r in rows if r["shot"] == s], key=lambda r: r["k"])
        a = g[0]
        for b in g[1:]:
            X.append([np.log(b["nel"] / a["nel"]), np.log(b["Ip"] / a["Ip"]), np.log(b["P"] / a["P"]), b["H"] - a["H"]])
            Y.append(np.log(np.array(b["table"]) / np.array(a["table"])))
    X, Y = np.array(X), np.array(Y)
    grid = ntm.TURB_RHO
    edges = np.array([0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 0.92])
    centres = 0.5 * (edges[1:] + edges[:-1])
    coef = np.zeros((3, len(grid), X.shape[1]))
    left = []
    for c in range(3):
        zone = np.zeros((len(centres), X.shape[1]))
        for i in range(len(centres)):
            y = Y[:, c, (grid >= edges[i]) & (grid < edges[i + 1])].mean(axis=1)
            good = np.isfinite(y) & np.isfinite(X).all(axis=1)
            zone[i] = huber(X[good], y[good])
            left.append((c, i, float(np.std(y[good])), float(np.std(y[good] - X[good] @ zone[i]))))
        for j in range(X.shape[1]):
            coef[c, :, j] = np.interp(grid, centres, zone[:, j]) * (grid < 0.92)
    return coef, left, len(X)


def real_events(shot, t0, t1):
    x = catalog_shot(shot)
    elm = np.array(x.get("elms", {}).get("t") or [])
    sw = x.get("sawteeth") or {}
    crash = np.array(sw.get("crash_t") or [])
    seq = sw.get("sequences") or []
    modes = [m for m in x.get("modes") or [] if m["n"] == 1 and m["family"] == "slow" and m.get("r1") and t0 <= m["t_on"] <= t1]
    locks = [m for m in x.get("modes") or [] if m.get("locked") and t0 <= (m.get("t_lock") or -1) <= t1]
    hwin = [(max(w["t0"], t0), min(w["t1"], t1)) for w in x["phases"]["hmode"]["windows"] if w["t1"] > t0 and w["t0"] < t1]
    q = x["phases"].get("quench") or {}
    flags = np.array(sw.get("in_sequence") or [], bool)
    return dict(crash_seq=int(((crash >= t0) & (crash <= t1) & flags).sum()) if len(flags) == len(crash) else 0, elm=elm[(elm >= t0) & (elm <= t1)], crash=crash[(crash >= t0) & (crash <= t1)], sequences=[z for z in seq if z["t1"] > t0 and z["t0"] < t1], modes=modes, locks=locks, H_s=sum(b - a for a, b in hwin), hwin=hwin, quench=q.get("t"), quench_kind=q.get("kind"), n1_any=len([m for m in x.get("modes") or [] if m["n"] == 1 and t0 <= m["t_on"] <= t1]))


def sequences(times):
    """Runs of 3 or more events 3 to 60 ms apart: the catalog's rule for a sawtooth sequence. Returns the runs."""
    runs, cur = [], []
    for t in times:
        if cur and not 0.003 <= t - cur[-1] <= 0.06:
            if len(cur) >= 3:
                runs.append(cur)
            cur = []
        cur.append(t)
    if len(cur) >= 3:
        runs.append(cur)
    return runs


def onsets(w, f, t, wmin=0.02, last=0.015):
    """Sim islands that count as modes: above wmin (m) for at least last (s). Returns (t_on, t_lock or None, w_max) per island."""
    out = []
    i = 0
    n = len(t)
    while i < n:
        if not np.isfinite(w[i]) or w[i] <= 0.003:
            i += 1
            continue
        j = i
        while j + 1 < n and np.isfinite(w[j + 1]) and w[j + 1] > 0.003:
            j += 1
        seg_w, seg_t, seg_f = w[i:j + 1], t[i:j + 1], f[i:j + 1]
        above = seg_w >= wmin
        if above.sum() * np.median(np.diff(t)) >= last:
            lock = seg_t[np.argmax(np.abs(seg_f) < 1000.0)] if (np.abs(seg_f) < 1000.0).any() else None
            out.append(dict(t_on=float(seg_t[0]), t_lock=None if lock is None else float(lock), w_max=float(seg_w.max()), t_end=float(seg_t[-1]), f_on=float(seg_f[:5].mean())))
        i = j + 1
    return out


def score(args):
    warnings.filterwarnings("ignore")
    folder = os.path.join(OUT, args.tag)
    rows = [json.loads(line) for line in open(os.path.join(folder, "results.jsonl"))]
    shots = sorted({r["shot"] for r in rows if r["shot"] in set(args.shots or split(args.split))})
    leads = mast.FIXED["leads_s"]
    pick = (0.01, 0.02, 0.05, 0.1, 0.2)
    idx = [leads.index(a) for a in pick]
    print(f"{args.tag}: {len(shots)} shots, seeds {sorted({r['seed'] for r in rows})}")
    stops = [r for r in rows if r["sim_error"] != "SimError.NO_ERROR"]
    if stops:
        print("runs stopped by an error:", [(r["shot"], r["seed"], r["steps"], r["sim_error"]) for r in stops][:12])
    for c in ("T_e", "T_i", "n_e"):
        def mean(get):
            per = []
            for s in shots:
                g = [get(r) for r in rows if r["shot"] == s and "profile" in r]
                per.append([np.nanmean([np.nan if c not in d["err_at_s"] or d["err_at_s"][c][i] is None else d["err_at_s"][c][i] for d in g] or [np.nan]) for i in idx])
            return 100 * np.nanmean(per, axis=0), np.array(per)
        a, pa = mean(lambda r: r["held"])
        b, pb = mean(lambda r: r["profile"])
        wins = [f"{np.nansum(pb[:, j] < pa[:, j])}/{np.sum(np.isfinite(pb[:, j]) & np.isfinite(pa[:, j]))}" for j in range(len(idx))]
        print(f"  {c} at {[int(1e3 * x) for x in pick]} ms: held {a.round(1).tolist()}, run {b.round(1).tolist()}, shots won {wins}")
    table = []
    for s in shots:
        g = [r for r in rows if r["shot"] == s]
        t0, t1 = g[0]["t_cut"], g[0]["t_end"]
        real = real_events(s, t0, t1)
        sim = []
        for r in g:
            d = np.load(os.path.join(folder, f"{s}_{r['seed']}.npz"))
            t = d["t"]
            crashes = t[np.nan_to_num(d["sawtooth"]) > 0] if "sawtooth" in d else np.array([])
            runs = sequences(list(crashes))
            H_s = float(np.sum(np.nan_to_num(d["H"]) * np.gradient(t))) if "H" in d and len(t) > 1 else 0.0
            elm = np.array(r["elm_times"])
            is_ = onsets(np.nan_to_num(d["w_2_1"]) if "w_2_1" in d else 0 * t, np.nan_to_num(d["f_2_1"], nan=1e9) if "f_2_1" in d else 0 * t, t, 0.02)
            te0 = d["T_e0"]
            W_ = d["W_thermal"]
            crash_i = np.where(np.nan_to_num(d["sawtooth"]) > 0)[0] if "sawtooth" in d else np.array([], int)
            crash_i = crash_i[crash_i > 0]
            e = dict(np.load(os.path.join(EXTRAS, f"{s}.npz")))
            g_ = dict(np.load(os.path.join(mast.ROOT, SHOTS, f"{s}.npz"), allow_pickle=False))
            neut = np.nan
            if "beam_neutrons" in d and "neutron_rates_total" in e:
                real_n = np.interp(t, e["t_sum"], np.nan_to_num(e["neutron_rates_total"]))
                neut = float(np.nanmean(d["beam_neutrons"]) / max(np.mean(real_n), 1.0))
            W_ratio = float(np.nanmean(d["W_thermal"] + np.nan_to_num(d["beam_W_fast"] if "beam_W_fast" in d else 0 * t)) / np.mean(np.interp(t, g_["t"], g_["W_dia"])))
            sim.append(dict(neutron_ratio=neut, W_ratio=W_ratio, drop_T=[float(te0[i] / te0[i - 1] - 1) for i in crash_i], drop_W=[float(W_[i] / W_[i - 1] - 1) for i in crash_i], elm=len(elm), f=1 / float(np.median(np.diff(elm))) if len(elm) >= 5 else np.nan, H_s=H_s, crashes=len(crashes), period=1e3 * float(np.median(np.concatenate([np.diff(x) for x in runs]))) if runs else np.nan, n_runs=len(runs), modes=len(is_), seeds=len(r["seeds"]), locks=sum(x["t_lock"] is not None for x in is_), lock_ms=[1e3 * (x["t_lock"] - x["t_on"]) for x in is_ if x["t_lock"]], w_max=[100 * x["w_max"] for x in is_], disrupted=r["disrupted"], t_stop=float(t[-1]), q0=float(d["q0"][-1]) if "q0" in d else np.nan, f_on=[x["f_on"] for x in is_]))
        real_f = 1 / float(np.median(np.diff(real["elm"]))) if len(real["elm"]) >= 5 else np.nan
        periods = [z["period_ms"] for z in real["sequences"]]
        table.append(dict(shot=s, T=t1 - t0, t_end=t1, real=real, real_f=real_f, real_period=float(np.median(periods)) if periods else np.nan, sim=sim))
    print(f"\n  {'shot':6s} {'run s':>6s} | {'ELMs real':>9s} {'sim':>6s} | {'H s real':>8s} {'sim':>5s} | {'f real':>6s} {'sim':>5s} | {'crash real':>10s} {'sim':>5s} | {'period real':>11s} {'sim':>5s} | {'n=1 real':>8s} {'sim':>5s} {'seeds':>5s} | {'lock real':>9s} {'sim':>5s} | stop")
    m = lambda x: float(np.nanmean(x)) if np.isfinite(x).any() else np.nan
    for x in table:
        S = x["sim"]
        arr = lambda k: np.array([y[k] for y in S], float)
        print(f"  {x['shot']:6d} {x['T']:6.3f} | {len(x['real']['elm']):9d} {m(arr('elm')):6.0f} | {x['real']['H_s']:8.3f} {m(arr('H_s')):5.3f} | {x['real_f']:6.0f} {m(arr('f')):5.0f} | {len(x['real']['crash']):10d} {m(arr('crashes')):5.0f} | {x['real_period']:11.1f} {m(arr('period')):5.1f} | {len(x['real']['modes']):8d} {m(arr('modes')):5.1f} {m(arr('seeds')):5.1f} | {len(x['real']['locks']):9d} {m(arr('locks')):5.1f} | {np.mean([y['t_stop'] < x['t_end'] - 0.002 for y in S]):.1f}")
    # aggregate
    spearman = lambda a, b: float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]) if len(a) > 2 else np.nan
    A = lambda k: np.array([float(np.nanmean(np.array([y[k] for y in x["sim"]], float))) if np.isfinite(np.array([y[k] for y in x["sim"]], float)).any() else np.nan for x in table])       # means over seeds: a chance event can be absent from most seeds
    n_real = np.array([len(x["real"]["elm"]) for x in table], float)
    print(f"\n  ELMs: total real {n_real.sum():.0f}, sim {A('elm').sum():.0f}; Spearman across shots {spearman(n_real, A('elm')):.2f}")
    ok = np.isfinite(np.array([x["real_f"] for x in table])) & np.isfinite(A("f"))
    if ok.sum() > 2:
        rf, sf = np.array([x["real_f"] for x in table])[ok], A("f")[ok]
        print(f"  ELM frequency, shots with 5 or more in both ({ok.sum()}): median real {np.median(rf):.0f} Hz, sim {np.median(sf):.0f} Hz; median |log ratio| {np.median(np.abs(np.log(sf / rf))):.2f}, Spearman {spearman(rf, sf):.2f}")
    print(f"  H-mode time: real {sum(x['real']['H_s'] for x in table):.2f} s, sim {A('H_s').sum():.2f} s; Spearman {spearman(np.array([x['real']['H_s'] for x in table]), A('H_s')):.2f}")
    c_real = np.array([len(x["real"]["crash"]) for x in table], float)
    print(f"  Crashes: real {c_real.sum():.0f} ({sum(x['real']['crash_seq'] for x in table)} in sequences), sim {A('crashes').sum():.0f}; Spearman {spearman(c_real, A('crashes')):.2f}. Sequences: real {sum(len(x['real']['sequences']) for x in table)}, sim {A('n_runs').sum():.0f}; period real {np.nanmedian([x['real_period'] for x in table]):.1f} ms, sim {np.nanmedian(A('period')):.1f} ms")
    drops = [v for x in table for y in x["sim"] for v in y["drop_T"]]
    if drops:
        print(f"  At crashes: core T_e median {100 * np.median(drops):.1f}% (measured -10.1%), W_th {100 * np.median([v for x in table for y in x['sim'] for v in y['drop_W']]):.2f}% (W_dia measured -0.5%)")
    T = sum(x["T"] for x in table)
    print(f"  Neutrons, run mean over measured mean: median over shots {np.nanmedian(A('neutron_ratio')):.2f}, IQR {np.nanpercentile(A('neutron_ratio'), 25):.2f} to {np.nanpercentile(A('neutron_ratio'), 75):.2f}. Thermal + fast energy over EFIT's W_dia: median {np.nanmedian(A('W_ratio')):.2f}, IQR {np.nanpercentile(A('W_ratio'), 25):.2f} to {np.nanpercentile(A('W_ratio'), 75):.2f}")
    ended = [x for x in table if x["real"]["quench"] is not None and x["real"]["quench"] <= x["t_end"] + 0.05 and x["real"]["quench_kind"] == "swing_left"]
    print(f"  Fast quench with solenoid swing left within 50 ms of the end of the data: {len(ended)} of {len(table)} real shots. Runs that end by a locked quench: {np.mean([y['disrupted'] for x in table for y in x['sim']]):.2f} of runs, median {np.nanmedian([x['t_end'] - y['t_stop'] for x in table for y in x['sim'] if y['disrupted']] or [np.nan]):.3f} s before the end of the data")
    all_sim = [y for x in table for y in x["sim"]]
    print(f"  Modes: real onsets {sum(len(x['real']['modes']) for x in table)} in {T:.1f} s ({sum(len(x['real']['modes']) for x in table) / T:.2f} /s), sim per seed {np.mean([np.sum([y['modes'] for y in x['sim']]) / len(x['sim']) for x in table]) * len(table):.1f} (seeds {np.mean([np.sum([y['seeds'] for y in x['sim']]) / len(x['sim']) for x in table]) * len(table):.1f})")
    lk = [v for y in all_sim for v in y["lock_ms"]]
    real_lk = [z.get("onset_to_lock_ms") for x in table for z in x["real"]["modes"] if z.get("locked") and z.get("onset_to_lock_ms") is not None]
    print(f"  Lock share of modes: real {np.mean([bool(z.get('locked')) for x in table for z in x['real']['modes']] or [np.nan]):.2f}, sim {sum(y['locks'] for y in all_sim) / max(sum(y['modes'] for y in all_sim), 1):.2f}; onset to lock median real {np.median(real_lk) if real_lk else np.nan:.0f} ms ({len(real_lk)}), sim {np.median(lk) if lk else np.nan:.0f} ms ({len(lk)}); largest island median {np.median([v for y in all_sim for v in y['w_max']] or [np.nan]):.1f} cm; disrupted {np.mean([y['disrupted'] for y in all_sim]):.2f}; f at onset median {np.median([v for y in all_sim for v in y['f_on']] or [np.nan]) / 1e3:.1f} kHz")
    json.dump([dict(shot=x["shot"], T=x["T"], elm_real=len(x["real"]["elm"]), H_real=x["real"]["H_s"], f_real=x["real_f"], crashes_real=len(x["real"]["crash"]), period_real=x["real_period"], modes_real=len(x["real"]["modes"]), locks_real=len(x["real"]["locks"]), sim=[{a: b for a, b in y.items()} for y in x["sim"]]) for x in table], open(os.path.join(folder, f"score_{args.split}.json"), "w"), indent=0, default=float)


def figure(args):
    """Recreated shots: one seed against what happened. One column per shot; core T_e, line density, thermal energy, and events on one strip."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = os.path.join(OUT, args.tag)
    rows = [json.loads(line) for line in open(os.path.join(folder, "results.jsonl"))]
    shots = args.shots or split(args.split)
    INK, MUTED, SIM, RULE, PAPER = "#1c1a16", "#7c7562", "#2a78d6", "#d6ccb5", "#fbf8f1"
    n = len(shots)
    fig, axes = plt.subplots(4, n, figsize=(max(3.0 * n, 7), 7.2), sharex="col", gridspec_kw=dict(height_ratios=[1, 1, 1, 1.15], hspace=0.08, wspace=0.28), squeeze=False, facecolor=PAPER)
    for j, shot in enumerate(shots):
        r = next(x for x in rows if x["shot"] == shot and x["seed"] == args.seed)
        d = np.load(os.path.join(folder, f"{shot}_{args.seed}.npz"))
        g = dict(np.load(os.path.join(mast.ROOT, SHOTS, f"{shot}.npz"), allow_pickle=False))
        t, t0, t1 = d["t"], r["t_cut"], r["t_end"]
        m = (g["t"] >= t0 - 0.02) & (g["t"] <= t1)
        real = real_events(shot, t0, t1)
        panels = ((g["T_e"][m, 0], d["T_e0"], "core $T_e$ (keV)"), (g["nel"][m] / 1e19, d["n_line"] / 1e19, "line density ($10^{19}$ m$^{-3}$)"), (g["W_th"][m] / 1e3, d["W_thermal"] / 1e3, "thermal energy (kJ)"))
        for i, (a, b, label) in enumerate(panels):
            ax = axes[i][j]
            ax.plot(g["t"][m], a, color=INK, lw=1.3, label="measured")
            ax.plot(t, b, color=SIM, lw=1.5, label="simulated")
            ax.axvline(t0, color=RULE, lw=1)
            if j == 0:
                ax.set_ylabel(label, fontsize=8, color=MUTED)
        ax = axes[3][j]
        strips = (("ELMs", real["elm"], np.array(r["elm_times"])), ("crashes", real["crash"], t[np.nan_to_num(d["sawtooth"]) > 0] if "sawtooth" in d else np.array([])))
        y = 0
        names = []
        for name, a, b in strips:
            ax.vlines(a, y + 0.55, y + 0.95, color=INK, lw=0.6)
            ax.vlines(b, y + 0.05, y + 0.45, color=SIM, lw=0.6)
            names.append((y + 0.5, name))
            y += 1.2
        for x in real["modes"]:
            ax.plot([x["t_on"], x["t_on"] + x["dur_ms"] / 1e3], [y + 0.75, y + 0.75], color=INK, lw=2.2, solid_capstyle="butt")
            if x.get("t_lock"):
                ax.plot(x["t_lock"], y + 0.75, marker="x", ms=6, color=INK, mew=1.5)
        for isl in onsets(np.nan_to_num(d["w_2_1"]) if "w_2_1" in d else 0 * t, np.nan_to_num(d["f_2_1"], nan=1e9) if "f_2_1" in d else 0 * t, t, 0.02):
            ax.plot([isl["t_on"], isl["t_end"]], [y + 0.25, y + 0.25], color=SIM, lw=2.2, solid_capstyle="butt")
            if isl["t_lock"]:
                ax.plot(isl["t_lock"], y + 0.25, marker="x", ms=6, color=SIM, mew=1.5)
        names.append((y + 0.5, "modes"))
        y += 1.2
        for a, b in real["hwin"]:
            ax.plot([a, b], [y + 0.75, y + 0.75], color=INK, lw=2.2, solid_capstyle="butt")
        H = np.nan_to_num(d["H"]) > 0.5 if "H" in d else np.zeros_like(t, bool)
        ax.fill_between(t, y + 0.17, y + 0.33, where=H, color=SIM, lw=0, step="mid")
        names.append((y + 0.5, "H-mode"))
        y += 1.2
        ax.set_ylim(-0.1, y)
        ax.set_yticks([a for a, _ in names])
        ax.set_yticklabels([b for _, b in names] if j == 0 else [], fontsize=8, color=MUTED)
        ax.set_xlabel("time (s)", fontsize=8, color=MUTED)
        axes[0][j].set_title(str(shot), fontsize=10, color=INK, loc="left", fontweight="bold")
        axes[0][j].set_xlim(t0 - 0.02, t1)
        for i in range(4):
            ax = axes[i][j]
            ax.set_facecolor(PAPER)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(RULE)
            ax.tick_params(colors=MUTED, labelsize=7.5, length=2.5)
            if i < 3:
                ax.grid(axis="y", color=RULE, lw=0.4)
    axes[0][0].legend(frameon=False, fontsize=8, loc="best", labelcolor=[INK, SIM])
    fig.suptitle(f"Hands-off runs against what happened: nothing measured after the vertical line but the machine programme (seed {args.seed}). On each strip, measured above, simulated below; x marks a lock", fontsize=9, color=INK, x=0.01, ha="left", y=0.965)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    path = os.path.join(OUT, "figures", f"{args.tag}_{args.split}_{args.seed}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PAPER)
    print(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["laws", "run", "score", "cost", "figure", "closure_data", "steps", "shapes"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed_factor", type=float, default=30.0)
    ap.add_argument("--split", default="fit")
    ap.add_argument("--shots", type=int, nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=["beams", "pedestal", "hmode", "elms", "sawtooth", "modes"])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--overrides", default=None, help="JSON: law values to replace, for calibration on the fit shots")
    ap.add_argument("--given_h", action="store_true", help="H-mode windows as measured, not from the hazard")
    ap.add_argument("--s_critical", type=float, default=0.25)
    ap.add_argument("--mixing", type=float, default=1.3)
    ap.add_argument("--allow_non_dev", action="store_true", help="run shots off runs/twin/dev_shots.json: only to reproduce runs made before that list existed; never the chip project's held-out shots")
    args = ap.parse_args()
    if args.command == "laws":
        fit_laws()
    elif args.command == "run":
        run(args)
    elif args.command == "cost":
        cost(args)
    elif args.command == "figure":
        figure(args)
    elif args.command == "closure_data":
        closure_data(args)
    elif args.command == "steps":
        steps(args)
    elif args.command == "shapes":
        shapes(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
