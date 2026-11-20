"""The pedestal from an open-source model (H46): EPED-NN (epednn.py) sets the height of TORAX's pedestal top, and where it sits, from the plasma as it stands. For a machine in its training range; a machine outside it gets the warning with the number.
TORAX takes its pedestal as an input, so this runs before each step, every interval: geometry, beta_N and current from TORAX, density as it is at the top, EPED-NN's pressure and width back. T_i over T_e stays as configured. With ELMs on, the height is the top of the cycle, as EPED's limit is what sets them off, so the mean sits half a drop lower.
Usage:
  python core/pedestal.py iter          TORAX's ITER hybrid, its configured pedestal against EPED-NN's
  python core/pedestal.py mast          EPED-NN against MAST's measured H-mode pedestals
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import epednn  # noqa: E402

QE = 1.602176634e-19
ROOT = os.path.abspath(os.environ.get("XTORAX_HOME") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the workspace: runs/twin and external/ (README, "Where things live")
OUT = os.path.join(ROOT, "runs", "twin", "epednn")


class EPEDPedestal:
    """A before-step hook for a reactor, or for a harness through its scenario."""

    def __init__(self, reactor, name="EPED1NNmodel", interval=0.1, solution="GH_H", mass=None, move_top=True):
        self.r, self.model, self.interval, self.solution = reactor, epednn.Model(name), interval, solution
        self.t_next = reactor.t
        self.mass, self.move_top = mass, move_top
        self.last = {}

    def inputs(self):
        """EPED-NN's inputs from TORAX's state."""
        r = self.r
        st, ppo = r.state, r.ppo
        geo, cp = st.geometry, st.core_profiles
        rho = np.asarray(geo.rho_norm)
        top = r.get_param("pedestal.rho_norm_ped_top")
        lower = np.asarray(getattr(geo, "delta_lower_face", geo.delta_face))[-1]
        upper = np.asarray(getattr(geo, "delta_upper_face", geo.delta_face))[-1]
        n_e = float(np.interp(top, rho, np.asarray(cp.n_e.value)))
        mass = self.mass if self.mass is not None else float(np.ravel(np.asarray(cp.A_i))[0])
        Z_eff = float(np.interp(top, rho, np.asarray(cp.Z_eff)))
        return dict(a=float(geo.a_minor), betan=float(ppo.beta_N), bt=abs(float(geo.B_0)), delta=epednn.effective_delta(float(lower), float(upper)), ip=abs(float(np.asarray(cp.Ip_profile_face)[-1])) / 1e6, kappa=float(np.asarray(geo.elongation_face)[-1]), m=mass, neped=n_e / 1e19, r=float(geo.R_major), zeffped=Z_eff), n_e

    def __call__(self, r=None):
        r = self.r
        if r.t < self.t_next - 1e-9:
            return
        self.t_next = r.t + self.interval
        st = r.state
        geo, cp = st.geometry, st.core_profiles
        rho = np.asarray(geo.rho_norm)
        x, n_e = self.inputs()
        res = self.model(solution=self.solution, nesep_ratio=0.25, tesep=75.0, **x)
        top = r.get_param("pedestal.rho_norm_ped_top")
        n_all = float(np.interp(top, rho, np.asarray(cp.n_i.value))) + float(np.interp(top, rho, np.asarray(cp.n_impurity_thermal.value))) if hasattr(cp, "n_impurity_thermal") else float(np.interp(top, rho, np.asarray(cp.n_i.value)))
        ratio = r.get_param("pedestal.T_i_ped") / max(r.get_param("pedestal.T_e_ped"), 1e-6)
        T_e = res["p"] / (QE * (n_e + ratio * n_all)) / 1e3
        if r.elms is not None:
            T_e = T_e / (1.0 + 0.5 * r.elms.drop_T)
        T_e = float(np.clip(T_e, 0.02, 40.0))
        r.set_param("pedestal.T_e_ped", T_e)
        r.set_param("pedestal.T_i_ped", T_e * ratio)
        new_top = top
        if self.move_top and res["width"] is not None:
            pn_face = np.asarray(r.ppo.psi_norm)
            new_top = float(np.clip(np.interp(1.0 - 2.0 * res["width"], pn_face, np.asarray(geo.rho_face_norm)), 0.7, 0.97))
            if abs(new_top - top) > 0.005:
                r.set_param("pedestal.rho_norm_ped_top", new_top)
        self.last = dict(t=r.t, p_kPa=res["p"] / 1e3, width=res["width"], T_e_ped=T_e, n_e_ped=n_e, rho_top=new_top, extrapolation=res["extrapolation"], worst=res["worst"], **x)


def run_iter(args):
    import check_plant
    import twin
    t0 = time.perf_counter()
    h = twin.Harness(check_plant.scenario(False), keep_history=False)
    r = h.reactor
    h.apply(dict(cmd="step", until=args.settle))
    h.apply(dict(cmd="snapshot", name="start"))
    t_start = r.t
    o = h.observe()
    print(f"TORAX settled in {time.perf_counter() - t0:.0f} s at t {r.t:.1f}: configured pedestal T_e {r.get_param('pedestal.T_e_ped'):.2f} keV, n_e {r.get_param('pedestal.n_e_ped'):.3g}, top at {r.get_param('pedestal.rho_norm_ped_top'):.3f}; fusion {o['P_fusion_MW']:.0f} MW, W {o['W_thermal_MJ']:.0f} MJ, beta_N {o['beta_N']:.2f}", flush=True)
    rows = {}
    for name in ("configured", "EPED1NNmodel", "EPED1NNensemble"):
        h.apply(dict(cmd="restore", name="start"))
        hook = None
        if name != "configured":
            hook = EPEDPedestal(r, name=name, interval=args.interval)
            r.before_step.append(hook)
        rows[name] = []
        while r.t < t_start + args.window - 1e-6:
            target = r.t + 1.0
            while r.t < target - 1e-6 and r.step() is not None:
                pass
            o = h.observe()
            rows[name].append(dict(t=r.t - t_start, P_fus=o["P_fusion_MW"], W=o["W_thermal_MJ"], beta_N=o["beta_N"], T_e_ped=r.get_param("pedestal.T_e_ped"), rho_top=r.get_param("pedestal.rho_norm_ped_top"), **({"p_kPa": hook.last.get("p_kPa"), "extrapolation": hook.last.get("extrapolation"), "worst": hook.last.get("worst"), "width": hook.last.get("width")} if hook else {})))
        if hook:
            r.before_step.remove(hook)
        x = rows[name][-1]
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()}, f"{r.sim_error}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, "iter.json"), "w"), indent=0)


def compare_mast(args):
    """EPED-NN against MAST's H-mode slices in the natural-run shots: the height at rho 0.91 of Thomson's T_e and n_e, with T_i = T_e, n_i = n_e. EPED's pedestal top is not at a fixed rho, and MAST's T_i is often above T_e, so this reads low by some tens of percent."""
    cat = json.load(open(os.path.join(ROOT, "runs", "twin", "mast_catalog", "catalog.json")))["shots"]
    names = json.load(open(os.path.join(ROOT, "runs", "twin", "mast_catalog", "natural_test_shots.json")))["shots"]
    single, ensemble = epednn.Model("EPED1NNmodel"), epednn.Model("EPED1NNensemble")
    rows = []
    for s in names:
        shot = s["shot"]
        path = os.path.join(ROOT, "runs", "twin", "mast_natural2", "shots", f"{shot}.npz")
        if not os.path.exists(path):
            continue
        g = dict(np.load(path, allow_pickle=False))
        win = [(w["t0"], w["t1"]) for w in cat[str(shot)]["phases"]["hmode"]["windows"]]
        i91 = int(np.argmin(np.abs(g["rho"] - 0.91)))
        for k, t in enumerate(g["t"]):
            if not any(a <= t <= b for a, b in win):
                continue
            T, n = float(g["T_e"][k, i91]), float(g["n_e"][k, i91])
            if not (np.isfinite(T) and np.isfinite(n) and T > 0.02 and n > 1e18):
                continue
            T_kev = T if T < 50 else T / 1e3
            beta_n = float(g["W_dia"][k]) * (2.0 / 3.0) / max(np.nanmax(g["V_rho"][k]), 1e-3) / (float(g["B0"][k]) ** 2 / (2 * 4e-7 * np.pi)) * 100 * float(g["a"][k]) * abs(float(g["B0"][k])) / (abs(float(g["Ip"][k])) / 1e6)
            x = dict(a=float(g["a"][k]), betan=beta_n, bt=abs(float(g["B0"][k])), delta=epednn.effective_delta(float(g["delta_l"][k]), float(g["delta_u"][k])), ip=abs(float(g["Ip"][k])) / 1e6, kappa=float(g["kappa"][k]), m=2.0, neped=n / 1e19, r=float(g["R0"][k]), zeffped=1.5, nesep_ratio=0.25, tesep=75.0)
            if not all(np.isfinite(v) for v in x.values()):
                continue
            a, b = single(**x), ensemble(**x)
            rows.append(dict(shot=shot, t=float(t), p_meas=2 * QE * n * T_kev * 1e3, p_single=a["p"], width=a["width"], p_ens=b["p"], sigma=b["sigma"], far=a["extrapolation"], worst=a["worst"], far_ens=b["extrapolation"], worst_ens=b["worst"], **x))
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, "mast.json"), "w"), indent=0)
    y, a, b = np.array([r["p_meas"] for r in rows]), np.array([r["p_single"] for r in rows]), np.array([r["p_ens"] for r in rows])
    from scipy.stats import spearmanr
    print(f"{len(rows)} H-mode slices of {len(set(r['shot'] for r in rows))} shots; measured pressure at rho 0.91: median {np.median(y) / 1e3:.2f} kPa")
    print(f"EPED1NNmodel: median {np.median(a) / 1e3:.2f} kPa; measured over model: median {np.median(y / a):.2f}, IQR {np.percentile(y / a, 25):.2f} to {np.percentile(y / a, 75):.2f}; Spearman {spearmanr(y, a)[0]:.2f}; outside bounds by {np.median([r['far'] for r in rows]):.2f} of a range (median), worst input mostly {max(set(r['worst'] for r in rows), key=[r['worst'] for r in rows].count)}")
    print(f"EPED1NNensemble: median {np.median(b) / 1e3:.2f} kPa; measured over model: median {np.median(y / b):.2f}, IQR {np.percentile(y / b, 25):.2f} to {np.percentile(y / b, 75):.2f}; Spearman {spearmanr(y, b)[0]:.2f}; spread {np.median([r['sigma'] / r['p_ens'] for r in rows]):.2f} of the height; outside bounds by {np.median([r['far_ens'] for r in rows]):.2f}, worst mostly {max(set(r['worst_ens'] for r in rows), key=[r['worst_ens'] for r in rows].count)}")
    print(f"Inputs: median a {np.median([r['a'] for r in rows]):.2f} m, R {np.median([r['r'] for r in rows]):.2f} m, B {np.median([r['bt'] for r in rows]):.2f} T, Ip {np.median([r['ip'] for r in rows]):.2f} MA, beta_N {np.median([r['betan'] for r in rows]):.2f}, n_ped {np.median([r['neped'] for r in rows]):.2f}e19, delta {np.median([r['delta'] for r in rows]):.2f}, kappa {np.median([r['kappa'] for r in rows]):.2f}")
    logs = np.polyfit(np.log(a), np.log(y), 1)
    print(f"log measured = {logs[0]:.2f} log model + {logs[1]:.2f}, single; scale of measured over model alone within a shot: correlation of the two within shots {np.nanmedian([np.corrcoef(y[[i for i, r in enumerate(rows) if r['shot'] == s]], a[[i for i, r in enumerate(rows) if r['shot'] == s]])[0, 1] for s in set(r['shot'] for r in rows) if sum(r['shot'] == s for r in rows) >= 4]):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["iter", "mast"])
    ap.add_argument("--settle", type=float, default=60.0)
    ap.add_argument("--window", type=float, default=15.0)
    ap.add_argument("--interval", type=float, default=0.5)
    args = ap.parse_args()
    if args.command == "iter":
        run_iter(args)
    else:
        compare_mast(args)


if __name__ == "__main__":
    main()
