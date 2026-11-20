"""Checks of the simulated reactor: TORAX's ITER hybrid flat top with the island model (core/ntm.py, H34)
and ECCD commanded by a controller (core/controllers.py, H35).

Base: external/torax/torax/examples/iterhybrid_predictor_corrector.py (10.5 MA, 51 MW, QLKNN transport, bootstrap
current, CHEASE geometry) with 100 radial cells (2 cm), a fixed 0.5 s step and a longer flat top. Its resistivity
multiplier (200) is kept, so the q profile settles within seconds; the island equation uses the physical conductivity.
ECCD: ntm's 'controlled' ECRH source (TORAX's formula, efficiency 0.25, deposition sigma 0.006 in rho_norm; current in
TORAX's <j.B>/B0 convention). One 2/1 mode. Scenarios:
  baseline   no seed, no drive: the width must stay 0; gives the q = 2 surface
  small      a 1 cm seed at t_seed: expected to decay (below the small-island width)
  seeded     an 8 cm seed at t_seed: expected to saturate (published ITER 2/1: 24 to 32 cm, Sauter 2010)
  fixed      seeded, P_eccd from t_on aimed at baseline's q = 2 surface (open loop)
  offset     the same, aimed 5 cm outside it
  tracking   seeded, the rule controller: full power above 3 cm, aimed at the measured surface
  dropout    tracking with the drive lost from t_on + 20 to t_on + 35 s (a gyrotron trip)
Reported per scenario: widths at a few times, the largest and final width, the driven current and its peak density
over the bootstrap current density at the surface (ITER suppression rule: at least 1.2, Sauter 2010).
Outputs: <out>/<scenario>.nc (TORAX's output with an 'islands' node) and <out>/summary.json.
Usage (TORAX's venv): python core/run_ntm.py [--only tracking]
"""
import argparse
import copy
import importlib.util
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import controllers as ctl  # noqa: E402
import ntm  # noqa: E402
import torax  # noqa: E402

ROOT = os.path.abspath(os.environ.get("XTORAX_HOME") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the workspace: runs/twin and external/ (README, "Where things live")

def torax_example(name):
    """TORAX's own example file: from a source checkout in external/torax if there is one, else from the installed package."""
    local = os.path.join(ROOT, "external", "torax", "torax", "examples", name)
    if os.path.exists(local):
        return local
    import importlib.util
    return os.path.join(list(importlib.util.find_spec("torax").submodule_search_locations)[0], "examples", name)



def base_config(t_final, n_rho=100, dt=0.5, width=0.006, efficiency=0.25, island_transport=True, physical_current_after=None):
    spec = importlib.util.spec_from_file_location(
        "ex", torax_example("iterhybrid_predictor_corrector.py"))
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    cfg = copy.deepcopy(ex.CONFIG)
    cfg["numerics"]["t_final"] = t_final
    cfg["numerics"]["fixed_dt"] = dt                  # the adaptive step shrinks as the cell size squared
    cfg["geometry"]["n_rho"] = n_rho
    cfg["time_step_calculator"] = {"calculator_type": "fixed"}
    if island_transport:                              # islands flatten the profile through the solver (ntm.QLKNNIsland)
        cfg["transport"]["model_name"] = "qlknn_island"
        cfg["pedestal"]["model_name"] = "set_T_ped_n_ped_elm"
        cfg["sources"]["generic_heat"]["model_name"] = "controlled"
        cfg["sources"]["gas_puff"]["model_name"] = "controlled"
    if physical_current_after is not None:            # TORAX's example speeds up current diffusion 200 times, "for testing purposes": only to settle
        t1 = float(physical_current_after)
        cfg["numerics"]["resistivity_multiplier"] = {0.0: cfg["numerics"]["resistivity_multiplier"], t1: cfg["numerics"]["resistivity_multiplier"], t1 + 1e-3: 1.0}
    cfg["sources"]["ecrh"] = {"model_name": "controlled", "gaussian_width": width, "is_explicit": True,
                              "current_drive_efficiency": {0.0: {0.0: efficiency, 1.0: efficiency}}}
    return cfg


def with_impurities(cfg, species):
    """Impurities given per species as n_Z / n_e, TORAX's mode that the harness can change as the run goes (impurities.py), with its Mavrin line radiation; no edge model here."""
    cfg["plasma_composition"].pop("Z_eff", None)
    cfg["plasma_composition"]["impurity"] = {"impurity_mode": "n_e_ratios", "species": dict(species)}
    cfg["sources"]["impurity_radiation"] = {"model_name": "mavrin_limited"}
    return cfg


def with_edge(cfg, neon=0.01, tungsten=1e-5, species=None):
    """TORAX's edge model on: a 1D parallel balance along the scrape-off layer (extended Lengyel) that gives the target temperature and heat flux and sets the core's separatrix
    temperature. Settings from TORAX's own ITER hybrid test (test_iterhybrid_predictor_corrector_mavrin_n_e_ratios_lengyel), which needs its impurities and Mavrin radiation."""
    cfg["plasma_composition"].pop("Z_eff", None)
    cfg["plasma_composition"]["impurity"] = {"impurity_mode": "n_e_ratios", "species": {"Ne": neon, "W": tungsten}}
    cfg["sources"]["impurity_radiation"] = {"model_name": "mavrin_fit"}
    cfg["edge"] = {"model_name": "extended_lengyel", "computation_mode": "forward", "fixed_impurity_concentrations": dict(species) if species else {"Ne": neon, "W": tungsten}, "impurity_sot": "core",
                   "connection_length_target": 30.0, "connection_length_divertor": 10.0, "angle_of_incidence_target": 3.0, "ratio_bpol_omp_to_bpol_avg": 4.0 / 3.0,
                   "toroidal_flux_expansion": 1.0, "use_enrichment_model": True, "diverted": True}
    return cfg


def summarize(recs, marks):
    t = np.array([r["t"] for r in recs])
    w = np.array([r.get("w_2_1", np.nan) for r in recs])
    at = lambda tt: float(w[min(np.searchsorted(t, tt), len(w) - 1)])
    res = dict(steps=len(t), w_max=float(np.nanmax(w)), w_end=float(w[-1]), w_at={str(int(m)): at(m) for m in marks},
               rho_q2_end=float(recs[-1].get("rho_2_1", np.nan)))
    on = [r for r in recs if r.get("j_cd_2_1", 0)]
    if on:
        res.update(I_cd_kA=float(np.median([r["I_cd_2_1"] for r in on]) / 1e3),
                   j_cd_over_j_bs=float(np.median([r["j_cd_2_1"] / r["j_bs_2_1"] for r in on])))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t_final", type=float, default=150.0)
    ap.add_argument("--t_seed", type=float, default=20.0)
    ap.add_argument("--t_on", type=float, default=80.0)
    ap.add_argument("--P_eccd", type=float, default=20.0)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default=os.path.join("runs", "torax_ntm"))
    args = ap.parse_args()
    out = os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    cfg = base_config(args.t_final)
    seed8, T, P = ((args.t_seed, 0.08),), args.t_on, args.P_eccd
    summary, rho_q2 = {}, None
    names = ["baseline", "small", "seeded", "fixed", "offset", "tracking", "dropout"]
    for name in names:
        if args.only and name not in args.only and not (name == "baseline" and rho_q2 is None):
            continue
        seeds = {"baseline": (), "small": ((args.t_seed, 0.01),)}.get(name, seed8)
        ctrl = {"fixed": lambda: ctl.OpenLoop(T, P, rho_q2), "offset": lambda: ctl.OpenLoop(T, P, rho_q2 + 0.025),
                "tracking": lambda: ctl.Tracking(P_max=P), "dropout": lambda: ctl.Dropout(ctl.Tracking(P_max=P), T + 20, T + 35)
                }.get(name, lambda: None)()
        t0 = time.perf_counter()
        tree, recs = ntm.run(torax.ToraxConfig.from_dict(cfg), [ntm.Mode(2, 1, seeds=seeds)], controller=ctrl)
        tree.to_netcdf(os.path.join(out, f"{name}.nc"))
        res = dict(wall_s=round(time.perf_counter() - t0, 1), **summarize(recs, (args.t_seed + 5, T - 1, T + 10, T + 30, args.t_final)))
        if name == "baseline":
            rho_q2 = float(np.nanmean([r["rho_2_1"] for r in recs if r["t"] > 0.5 * args.t_final]))
        summary[name] = res
        print(name, json.dumps(res), flush=True)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print("saved", os.path.join(args.out, "summary.json"))


if __name__ == "__main__":
    main()
