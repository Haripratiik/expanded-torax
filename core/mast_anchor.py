"""Anchored transport on MAST (H40, docs/digital-twin.md): start a real shot in balance, then ask the twin about what changes.

The backtest (mast.py) found every transport option leaves 10% within 1 to 4 ms, because no model's electron and ion heat losses match MAST's. Anchoring: at the cut, find radial multipliers on the model's chi_e, chi_i and D (TORAX's turbulence table, ntm.TURBULENCE) such that
the measured profiles are stationary under the measured sources, by TORAX's own step: take a step from the measured state, turn each channel's rate of change into the missing flux through each surface, correct the diffusivity there, repeat from the same state. This is power balance, in TORAX's own
discretization and sources. Nothing from after the cut enters, and nothing is fitted across shots. Afterwards the multipliers stay frozen, so transport changes only as the base model says when the plasma changes: frozen (constant base, fixed diffusivity), bgb (Bohm-gyroBohm) or qlknn.
On a flat top this can at best tie the held profile, which by selection barely changes. So the test that counts is a change: cuts just before a measured step in NBI power, where the held profile must be wrong (transients below). Each is scored against the held profile, at the same leads.
Usage (TORAX's venv; shot files from mast.py convert, transient shots from WSL with the command transients):
  python core/mast_anchor.py flat --models bgb frozen qlknn
  python core/mast_anchor.py transients          (WSL, reads the store read-only)
  python core/mast_anchor.py steps --models bgb frozen qlknn
Outputs in runs/twin/mast_anchor/.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mast  # noqa: E402
import tglfnn_st  # noqa: E402,F401  (registers the spherical-tokamak transport model)

OUT = os.path.join(mast.ROOT, "runs", "twin", "mast_anchor")
KEV = 1e3 * 1.602176634e-19
# MAST's beams, not any other machine's. Energy 65 keV instead of 60: set on 17 MAST shots outside the test set, where a first version gave neutrons at 0.94 (Ip / 600 kA)^-0.51 of measured, scatter 23%. Tangency radius from the machine. sigma: charge exchange and ionization on deuterium at 30 keV per nucleon, from memory; to compare with Suzuki et al. 1998.
# Prompt loss comes from the orbit width (beams.py). On the 19 fit shots that does as well as the law it replaced, 1 - min(1, (Ip / 900 kA)^0.5), which was fitted to that same neutron trend: neutrons 1.04 of measured against 0.98, T_e and T_i at 100 ms 25.0, 40.5% against 24.6, 41.5%.
BEAMS = dict(E_keV=65.0, R_tan=0.7, sigma=2.9e-20, f_lost="orbit")
# The frozen test of 2026-11-04 (docs/digital-twin.md), to reproduce it: scenario beams="frozen". Binning 201 points along the path put a factor of 2 of cell-to-cell noise into the heating, which turned out not to matter, and the pencil beam a singularity at the axis.
BEAMS_FROZEN = dict(BEAMS, f_lost=lambda Ip: 1 - min(1.0, (Ip / 9e5) ** 0.5), n_path=201, spread=False)
BASE = dict(bgb=dict(model_name="bohm-gyrobohm_table"), frozen=dict(model_name="constant_table", chi_i=1.0, chi_e=1.0, D_e=1.0, V_e=0.0), qlknn=dict(model_name="qlknn_island"), tglfnn_step=dict(model_name="tglfnn-ukaea", machine="step"), tglfnn_multi=dict(model_name="tglfnn-ukaea", machine="multimachine"), tglfnn_anchored=dict(model_name="tglfnn-ukaea_table", machine="multimachine"), tglfnn_st=dict(model_name="combined_table", transport_models=[dict(model_name="tglfnn-st", rho_max=0.6), dict(model_name="bohm-gyrobohm", rho_min=0.6)]))      # tglfnn_st: the spherical-tokamak networks inside rho 0.6, Bohm-gyroBohm outside, where MAST leaves their training range (tglfnn_st.py, H70)


def ntm_table():
    import ntm
    return [ntm.TURBULENCE[i][(ntm.TURB_RHO > 0.1) & (ntm.TURB_RHO < 0.88)] for i in range(3)]


def smooth(v, n=2):
    k = np.ones(2 * n + 1) / (2 * n + 1)
    return np.convolve(np.pad(v, n, mode="edge"), k, mode="valid")


def anchor(r, n_iter=60, relax=0.6, lo=0.01, hi=500.0, rates=None, geo_overrides=None, tied=False, add=False):
    """Multipliers on chi_e, chi_i and D such that TORAX's step leaves this state still, or changing at the measured rates (rates: T_e, T_i, n_e per second on TORAX's grid). Writes ntm.TURBULENCE. Returns the rates of change left, relative, per channel.
    tied: a steady density profile fixes only V / D, not D (docs/lit/28, section 5). So D comes from the heat channels (ntm.PARTICLE) and what is solved for is V / D, which then stays."""
    import ntm
    ntm.ANCHOR_ADD[0] = 0.0                              # the balance is found multiplying; with add, the same start is then held by an added term (below)
    s0, p0 = r.state, r.ppo
    geo, cp = s0.geometry, s0.core_profiles
    rho, rf = np.asarray(geo.rho_norm), np.asarray(geo.rho_face_norm)
    drho = float(np.asarray(geo.drho_norm).ravel()[0])
    vpr, g1v = np.asarray(geo.vpr), np.asarray(geo.g1_over_vpr_face)
    T_e, T_i, n_e = np.asarray(cp.T_e.value), np.asarray(cp.T_i.value), np.asarray(cp.n_e.value)
    n_i = np.asarray(cp.n_i.value)
    face = lambda c: np.concatenate([[c[0]], 0.5 * (c[1:] + c[:-1]), [c[-1]]])
    grad = lambda c: np.concatenate([[0.0], np.diff(c) / drho, [(c[-1] - c[-2]) / drho]])
    log = {}
    tables = np.ones_like(ntm.TURBULENCE)
    x = ntm.TURB_RHO
    if tied:
        ntm.PARTICLE[0] = 1.0
    trigger = None
    if r.provider.mhd.sawtooth is not None:            # no crash while balancing: a crash would read as a transport error
        trigger = r.get_param("mhd.sawtooth.trigger_model.s_critical")
        r.set_param("mhd.sawtooth.trigger_model.s_critical", 1e3)
    for it in range(n_iter):
        ntm.TURBULENCE[:] = tables
        s1, _ = r.step_fn(s0, p0, max_dt=np.inf, runtime_params_overrides=r.provider, geo_overrides=geo_overrides)
        dt = float(s1.t) - float(s0.t)
        c1, tr = s1.core_profiles, s1.core_transport
        rows = ((0, T_e, np.asarray(c1.T_e.value), n_e, np.asarray(tr.chi_face_el), 1.5 * KEV), (1, T_i, np.asarray(c1.T_i.value), n_i, np.asarray(tr.chi_face_ion), 1.5 * KEV), (2, n_e, np.asarray(c1.n_e.value), np.ones_like(n_e), np.asarray(tr.d_face_el), 1.0))
        for row, X0, X1, dens, base, coef in rows:
            rate = (X1 - X0) / dt - (0.0 if rates is None else rates[row])
            log[row] = float(np.sqrt(np.mean((rate[(rho > 0.1) & (rho < 0.88)] / X0[(rho > 0.1) & (rho < 0.88)]) ** 2)))
            Q = np.concatenate([[0.0], np.cumsum(coef * dens * rate * vpr * drho)])                 # the flux missing through each surface
            g = grad(X0)
            work = g1v * face(dens) * (KEV if row < 2 else 1.0) * (-g)
            if tied and row == 2:
                inner = (rf > 0.03) & (rf < 0.9)
                dV = np.where(inner, relax * Q / np.maximum(np.asarray(geo.g0_face) * face(n_e), 1e-30), 0.0)
                ntm.PARTICLE[1] = np.clip(np.interp(x, rf, smooth((np.asarray(tr.v_face_el) + dV) / np.maximum(np.asarray(tr.d_face_el), 1e-3))), -60.0, 60.0)
                continue
            good = (work > 1e-3 * np.abs(work).max()) & (rf > 0.03) & (rf < 0.9)
            dchi = np.where(good, Q / np.where(good, work, 1.0), 0.0)
            ratio = np.where(good, np.maximum(base + relax * dchi, 0.02) / np.maximum(base, 1e-3), 1.0)
            ratio = np.where(rf < 0.12, ratio[int(np.argmin(np.abs(rf - 0.12)))], ratio)      # near the axis the gradient says nothing: go on from just outside, or the axis keeps the base model's chi and runs away
            tables[row] = np.clip(tables[row] * np.interp(x, rf, smooth(ratio)), lo, hi)
    ntm.TURBULENCE[:] = tables
    if add:                                              # the base model's own chi at this start: at the start the two forms give the same chi, afterwards the added part stays as it was (H70)
        ntm.TURBULENCE[:2] = 1.0
        s1, _ = r.step_fn(s0, p0, max_dt=np.inf, runtime_params_overrides=r.provider, geo_overrides=geo_overrides)
        ntm.CHI_START[0] = np.interp(x, rf, np.asarray(s1.core_transport.chi_face_el))
        ntm.CHI_START[1] = np.interp(x, rf, np.asarray(s1.core_transport.chi_face_ion))
        ntm.TURBULENCE[:] = tables
        ntm.ANCHOR_ADD[0] = 1.0
    if trigger is not None:
        r.set_param("mhd.sawtooth.trigger_model.s_critical", trigger)
    return log


def measured_rates(m, r, slices=5, channels=("T_e", "T_i", "n_e")):
    """dT_e/dt, dT_i/dt, dn_e/dt at the start, per second on TORAX's grid: a straight line through the last slices at each radius, smoothed in radius; only slices up to the start. Channels not named stay at zero."""
    truth, k = m["truth"], m["k"]
    rho = np.asarray(r.state.geometry.rho_norm)
    idx = np.arange(max(k - slices + 1, 0), k + 1)
    t = truth.t[idx] - truth.t[idx].mean()
    out = []
    for name in ("T_e", "T_i", "n_e"):
        if name not in channels:
            out.append(np.zeros_like(rho))
            continue
        v = getattr(truth, name)
        if v is None:
            v = truth.T_e
        slope = (t[:, None] * (v[idx] - v[idx].mean(axis=0))).sum(axis=0) / (t ** 2).sum()
        now = np.maximum(v[idx].mean(axis=0), 1e-6)
        relative = np.clip(smooth(slope / now, 8), -30.0, 30.0)        # per second, as a fraction: within what a plasma does on these shots
        out.append(np.interp(rho, truth.rho, relative * now))
    return out


def scenario_config(s):
    """A MAST shot as a harness scenario. s["mast"]: shot, cut (s; default the first allowed cut), model (bgb, frozen, qlknn), past (s to run past the end of the data, holding the last programme), shots (folder). Returns TORAX's config and the measured shot, for comparison.
    Adds to the backtest's config what the harness drives: an ECCD source (MAST has none: a what-if), gas, a pedestal that can be scaled, and impurities as ratios."""
    import run_ntm
    m = s["mast"]
    folder = m.get("shots", os.path.join(mast.OUT, "shots"))
    truth, g, info = mast.load_shot(os.path.join(mast.ROOT, folder, f"{m['shot']}.npz"))
    k = int(np.searchsorted(truth.t, m["cut"])) if "cut" in m else mast.cut_indices(truth.t)[0]
    if m.get("Ip") is not None:                        # a programme in place of the measured current
        import dataclasses
        truth = dataclasses.replace(truth, Ip=np.asarray(m["Ip"], float))
    model = m.get("model", "bgb")
    cfg = mast.build_config(truth, g, k, "bgb", frozen=bool(m.get("natural")))     # natural: the edge and pedestal stay at their values at the start, as nothing measured enters afterwards
    cfg["transport"] = {**mast.transport_settings("qlknn"), **BASE[model]} if model == "qlknn" else dict(BASE[model])
    cfg["numerics"]["t_final"] = float(truth.t[-1]) + m.get("past", 0.0)
    if s.get("beams"):                                 # beams.Beams heats and fuels instead
        cfg["sources"]["generic_heat"]["P_total"] = 0.0
        cfg["sources"]["generic_particle"]["S_total"] = 0.0
        if isinstance(s["beams"], dict) and s["beams"].get("drive_current"):
            cfg["sources"]["generic_current"] = dict(model_name="beams")
    if s.get("sawtooth"):
        cfg["mhd"] = dict(sawtooth=s["sawtooth"])
    cfg["pedestal"]["model_name"] = "set_T_ped_n_ped_elm"
    cfg["sources"]["generic_heat"]["model_name"] = "controlled"
    cfg["sources"]["gas_puff"] = dict(model_name="controlled", S_total=0.0, puff_decay_length=0.05)
    cfg["sources"]["ecrh"] = dict(model_name="controlled", gaussian_width=m.get("eccd_width", 0.03), is_explicit=True, current_drive_efficiency={0.0: {0.0: 0.25, 1.0: 0.25}})
    if s.get("impurities"):
        run_ntm.with_impurities(cfg, s["impurities"])
    return cfg, dict(shot=m["shot"], truth=truth, g=g, info=info, k=k)


def run_anchored(truth, g, k, model, tau_E, **kw):
    import ntm
    import shot as shot_lib
    import torax
    t_cut = float(truth.t[k])
    cfg = mast.build_config(truth, g, k, "bgb", transport_extra=None, **kw)
    cfg["transport"] = {**mast.transport_settings("qlknn"), **BASE[model]} if model == "qlknn" else dict(BASE[model])
    t0 = time.perf_counter()
    r = ntm.Reactor(torax.ToraxConfig.from_dict(cfg), [])
    left = anchor(r)
    while r.step() is not None:
        pass
    tree, _ = r.finish()
    sc = shot_lib.score(mast.scored_shot(truth), tree, t_cut)
    res = mast.summarize(sc, t_cut, tau_E)
    res.update(cut=t_cut, wall_s=round(time.perf_counter() - t0, 1), anchor_left=left, steps=len(tree["profiles"].to_dataset()["time"]) - 1, sim_error=str(r.sim_error))
    res["table_median"] = [float(np.median(ntm.TURBULENCE[i][(ntm.TURB_RHO > 0.1) & (ntm.TURB_RHO < 0.88)])) for i in range(3)]
    return res, sc


def held(truth, k, tau_E):
    import shot as shot_lib
    t_cut = float(truth.t[k])
    sim_t = np.arange(t_cut, truth.t[-1] + 1e-9, mast.FIXED["dt"])
    tree = mast.persistence_tree(mast.known_shot(truth, k, False), t_cut, sim_t, truth.rho)
    return mast.summarize(shot_lib.score(mast.scored_shot(truth), tree, t_cut), t_cut, tau_E) | dict(cut=t_cut)


def table(rows, models, leads):
    """Mean over cuts per shot, mean over shots: error at leads (ms) per channel; and the median trust window."""
    for c in ("T_e", "T_i", "n_e"):
        print(f"\n{c}: error (%) at lead, mean over shots")
        print(f"{'':10s}" + "".join(f"{int(L * 1e3):>7d}" for L in leads) + "   median horizon (ms)")
        for m in ["held"] + models:
            line = []
            for L in leads:
                i = mast.FIXED["leads_s"].index(L)
                per = [np.mean([x["res"]["err_at_s"][c][i] for x in rows if x["shot"] == s and x["model"] == m and c in x["res"]["err_at_s"] and x["res"]["err_at_s"][c][i] is not None] or [np.nan]) for s in sorted({x["shot"] for x in rows})]
                line.append(100 * np.nanmean(per))
            hz = [x["res"][f"horizon_{c}_s"] for x in rows if x["model"] == m and x["res"].get(f"horizon_{c}_s") is not None]
            print(f"{m:10s}" + "".join(f"{v:7.1f}" for v in line) + f"   {1e3 * np.median(hz) if hz else float('nan'):.1f}")


def flat(models, shots, out):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "flat.jsonl")
    rows = [json.loads(line) for line in open(path)] if os.path.exists(path) else []
    done = {(x["shot"], x["model"], x["k"]) for x in rows}
    log = open(path, "a")
    for s in shots:
        truth, g, info = mast.load_shot(os.path.join(mast.OUT, "shots", f"{s}.npz"))
        for k in mast.cut_indices(truth.t):
            for m in ["held"] + models:
                if (s, m, k) in done:
                    continue
                try:
                    res = held(truth, k, info["tau_E_s"]) if m == "held" else run_anchored(truth, g, k, m, info["tau_E_s"])[0]
                except Exception as err:
                    print(s, m, k, "failed", repr(err)[:300], flush=True)
                    continue
                rows.append(dict(shot=s, model=m, k=k, res=res))
                log.write(json.dumps(rows[-1]) + "\n")
                log.flush()
                if m != "held":
                    print(s, m, f"cut {res['cut']:.3f}", f"{res['wall_s']} s", "left", {a: round(b, 2) for a, b in res["anchor_left"].items()}, "tables", [round(v, 2) for v in res["table_median"]], "err 20 ms", [round(res["err_at_s"][c][2] or 0, 3) for c in ("T_e", "T_i", "n_e") if c in res["err_at_s"]], flush=True)
    table(rows, models, [0.005, 0.01, 0.02, 0.05, 0.1])


def transients(out):
    """WSL: shots with a sharp step in NBI power on a steady plasma, data on both sides. Selection fixed here before any run."""
    import zarr
    rule = dict(step_min_W=5e5, step_rel=0.3, step_within_s=0.02, before_s=0.04, after_s=0.08, ip_min_A=3e5, ip_tol=0.1, zmag_max_m=0.03, ts_min=100, cx_min=50, session_gap=0, test_shots_excluded=False, per_shot=1, max_events=30)
    shots = [s["shot"] for s in json.load(open(os.path.join(mast.STORE, "manifest.json")))["shots"]]
    found = []
    for shot in shots:
        s = mast.scalars(zarr.open_group(os.path.join(mast.STORE, f"{shot}.zarr"), mode="r"))
        if s is None:
            continue
        t, P = s["t"], np.nan_to_num(s["pnbi"])
        dt = float(np.median(np.diff(t)))
        n_step, n_before, n_after = int(rule["step_within_s"] / dt), int(rule["before_s"] / dt), int(rule["after_s"] / dt)
        best = None
        for i in range(n_before, len(t) - n_after - n_step):
            a, b = i - n_before, i + n_step + n_after
            before, after = np.median(P[a:i]), np.median(P[i + n_step:b])
            step = after - before
            if abs(step) < rule["step_min_W"] or abs(step) < rule["step_rel"] * max(before, after):
                continue
            if np.ptp(P[a:i]) > 0.15 * max(before, 1e5) or np.ptp(P[i + n_step:b]) > 0.15 * max(after, 1e5):
                continue
            ip = np.nan_to_num(s["ip"][a:b])
            if ip.min() < rule["ip_min_A"] or np.ptp(ip) > 2 * rule["ip_tol"] * np.median(ip):
                continue
            if (np.abs(np.nan_to_num(s["zmag"][a:b], nan=9.0)) > rule["zmag_max_m"]).any() or s["nts"][a:b].min() < rule["ts_min"] or s["ncx"][a:b].min() < rule["cx_min"] or not np.isfinite(s["q95"][a:b]).all():
                continue
            if best is None or abs(step) > abs(best["step_W"]):
                best = dict(shot=shot, i=i, t_step=float(t[i]), t0=float(t[a]), t1=float(t[b - 1]), step_W=float(step), before_W=float(before), after_W=float(after), ip_kA=float(np.median(ip) / 1e3), nel_1e19=float(np.median(s["nel"][a:b]) / 1e19))
        if best:
            found.append(best)
    found.sort(key=lambda e: -abs(e["step_W"]))
    os.makedirs(out, exist_ok=True)
    json.dump(dict(rule=rule, n_found=len(found), events=found), open(os.path.join(out, "transients.json"), "w"), indent=1)
    print(len(found), "events")
    for e in found[:40]:
        print(e["shot"], f"t {e['t_step']:.3f} step {e['step_W'] / 1e6:+.2f} MW from {e['before_W'] / 1e6:.2f}; window {e['t0']:.3f}-{e['t1']:.3f}; Ip {e['ip_kA']:.0f} kA, nel {e['nel_1e19']:.2f}")


def convert_transients(out, n):
    """WSL: shot files for the events, through mast.convert."""
    events = json.load(open(os.path.join(out, "transients.json")))["events"][:n]
    os.makedirs(os.path.join(out, "shots"), exist_ok=True)
    infos = {}
    for e in events:
        entry = dict(shot=e["shot"], t0=e["t0"], t1=e["t1"], campaign="", shape="", heating="")
        try:
            infos[str(e["shot"])] = mast.convert(entry, out)
        except Exception as err:
            print(e["shot"], "failed", repr(err)[:300], flush=True)
    json.dump(infos, open(os.path.join(out, "conversion.json"), "w"), indent=1)


def steps(models, out, n):
    """The test that counts: from just before a step in power, against the held profile."""
    events = json.load(open(os.path.join(out, "transients.json")))["events"][:n]
    path = os.path.join(out, "steps.jsonl")
    rows = [json.loads(line) for line in open(path)] if os.path.exists(path) else []
    done = {(x["shot"], x["model"]) for x in rows}
    log = open(path, "a")
    for e in events:
        f = os.path.join(out, "shots", f"{e['shot']}.npz")
        if not os.path.exists(f):
            continue
        truth, g, info = mast.load_shot(f)
        k = int(np.searchsorted(truth.t, e["t_step"] - 0.002)) - 1
        if k < mast.FIXED["trailing_slices"] - 1:
            k = mast.FIXED["trailing_slices"] - 1
        for m in ["held"] + models:
            if (e["shot"], m) in done:
                continue
            try:
                res = held(truth, k, info["tau_E_s"]) if m == "held" else run_anchored(truth, g, k, m, info["tau_E_s"])[0]
            except Exception as err:
                print(e["shot"], m, "failed", repr(err)[:300], flush=True)
                continue
            rows.append(dict(shot=e["shot"], model=m, k=k, step_W=e["step_W"], res=res))
            log.write(json.dumps(rows[-1]) + "\n")
            log.flush()
            print(e["shot"], m, f"step {e['step_W'] / 1e6:+.2f} MW", "err 20, 50 ms", [(c, round(100 * (res["err_at_s"][c][2] or 0), 1), round(100 * (res["err_at_s"][c][4] or 0), 1)) for c in ("T_e", "T_i", "n_e") if c in res["err_at_s"]], flush=True)
    table(rows, models, [0.01, 0.02, 0.05])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["flat", "transients", "convert_transients", "steps"])
    ap.add_argument("--models", nargs="+", default=["bgb", "frozen", "qlknn"])
    ap.add_argument("--shots", type=int, nargs="*", default=[27552, 30275, 25007, 27672, 28041, 22200, 24442])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.command == "flat":
        flat(args.models, args.shots, args.out)
    elif args.command == "transients":
        transients(args.out)
    elif args.command == "convert_transients":
        convert_transients(args.out, args.n)
    else:
        steps(args.models, args.out, args.n)


if __name__ == "__main__":
    main()
