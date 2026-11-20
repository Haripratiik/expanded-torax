"""Does the reactor run with TORAX's edge model, and what does the edge change? (H36)

TORAX's extended Lengyel model (a 1D parallel heat and particle balance along the scrape-off layer, with impurity radiation) gives the target temperature and heat flux and sets the
separatrix boundary of the core solver. Settings from TORAX's own ITER hybrid test (test_iterhybrid_predictor_corrector_mavrin_n_e_ratios_lengyel: forward mode, 1% neon, 1e-5 tungsten,
Mavrin radiation). The same plasma without the edge model, then with it, then with a large island in it.
Usage (TORAX's venv): python core/check_edge.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ntm  # noqa: E402
import run_ntm  # noqa: E402
import torax  # noqa: E402


def config(edge, impurities, t_final):
    cfg = run_ntm.base_config(t_final)
    if impurities:
        run_ntm.with_edge(cfg)
    if not edge:
        cfg.pop("edge", None)
    return cfg


out = {}
for name, edge, imp, seeds in (("no edge, as in all the runs so far", False, False, ()), ("same impurities, no edge", False, True, ()), ("edge model on", True, True, ()), ("edge model on, 12 cm island at 40 s", True, True, ((40.0, 0.12),))):
    t0 = time.perf_counter()
    tree, recs = ntm.run(torax.ToraxConfig.from_dict(config(edge, imp, 70.0)), [ntm.Mode(2, 1, seeds=seeds)])
    sc = tree["scalars"].to_dataset()
    p = tree["profiles"].to_dataset()
    row = dict(P_fusion_MW=float(sc["P_fusion"].values[-1] / 1e6), Q=float(sc["Q_fusion"].values[-1]), W_thermal_MJ=float(sc["W_thermal_total"].values[-1] / 1e6), H98=float(sc["H98"].values[-1]),
               P_SOL_MW=float(sc["P_SOL_total"].values[-1] / 1e6), T_e_edge_keV=float(p["T_e"].values[-1, -1]), n_e_edge_1e19=float(p["n_e"].values[-1, -1] / 1e19), w_end_cm=float(100 * recs[-1].get("w_2_1", 0.0)), wall_s=round(time.perf_counter() - t0, 1))
    if "edge" in tree.children:
        e = tree["edge"].to_dataset()
        row["edge_outputs"] = {k: float(np.ravel(e[k].values)[-1]) for k in e.data_vars if np.issubdtype(e[k].dtype, np.number) and e[k].values.size and (e[k].ndim <= 1)}
    out[name] = row
    print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "edge_outputs"}), flush=True)
    if "edge_outputs" in row:
        print("   edge outputs:", json.dumps({k: float(f"{v:.4g}") for k, v in row["edge_outputs"].items()}), flush=True)
os.makedirs(os.path.join("runs", "twin"), exist_ok=True)
json.dump(out, open(os.path.join("runs", "twin", "edge_check.json"), "w"), indent=1)
