"""Each actuator and instability of the harness does what it says, from one settled ITER hybrid plasma, on branches. Prints a line per command: what it was meant to do and the measured change.
Usage (TORAX's venv): python core/check_plant.py [--only spi gas] [--edge]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twin  # noqa: E402

SPECIES = {"Ne": 0.01, "W": 1e-5, "Ar": 0.0, "N": 0.0, "Kr": 0.0}


def scenario(edge):
    return {"name": "check_plant", "t_final": 2000.0, "dt": 0.1, "edge": edge, "impurities": SPECIES, "physics": {"a_bs": 1.5, "a_cd": 0.5}, "gyrotrons": {"units": 20, "unit_MW": 1.0}, "decision": {"kind": "none"}, "events": []}


def read(h):
    o = h.observe()
    return {k: o.get(k) for k in ("t", "P_fusion_MW", "W_thermal_MJ", "P_SOL_MW", "n_line", "T_e0", "T_i0", "Z_eff_avg", "q95", "beta_N", "f_GW")} | dict(P_rad_MW=(o.get("P_rad") or 0.0) / 1e6, n20=(o.get("n_line") or 0.0) / 1e20, disrupted=o["disrupted"], finished=o["finished"], plant=o["plant"])


def show(name, a, b, keys):
    print(f"{name:22s} " + "  ".join(f"{k} {a[k]:.3g} -> {b[k]:.3g}" for k in keys if a.get(k) is not None and b.get(k) is not None) + f"  [t {b['t'] - a['t']:.2f} s" + (", ended" if b["finished"] else "") + "]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--edge", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()
    h = twin.Harness(scenario(args.edge), keep_history=False)
    h.apply(dict(cmd="step", until=40.0))
    h.apply(dict(cmd="snapshot", name="settled"))
    print(f"settled in {time.perf_counter() - t0:.0f} s", json.dumps({k: v for k, v in read(h).items() if k != "plant"}), flush=True)

    def case(name, cmds, then=2.0, keys=("P_fusion_MW", "W_thermal_MJ", "P_SOL_MW", "P_rad_MW", "n20", "T_e0", "Z_eff_avg")):
        if args.only and name not in args.only:
            return None
        h.apply(dict(cmd="restore", name="settled"))
        a, answer = read(h), {}
        for c in cmds:
            answer = h.apply(c)
            if answer.get("error"):
                print(name, "ERROR", answer)
        t1 = time.perf_counter()
        h.apply(dict(cmd="step", until=h.reactor.t + then))
        b = read(h)
        show(name, a, b, keys)
        print(f"    answer {json.dumps({k: v for k, v in answer.items() if k not in ('ok', 'cmd', 't')})}  impurity atoms {b['plant'].get('impurity_atoms')}  {time.perf_counter() - t1:.1f} s" + (f"  card {json.dumps(b['plant']['card'])}" if b["plant"]["card"] else ""), flush=True)
        return b

    case("quiet", [], 5.0)
    case("pellet_hfs", [dict(cmd="pellet", r_mm=4.0, side="hfs")], 0.2)
    case("pellet_lfs", [dict(cmd="pellet", r_mm=4.0, side="lfs")], 0.2)
    case("gas_D", [dict(cmd="gas", species="D", rate=3e22)], 8.0)
    case("gas_Ne", [dict(cmd="gas", species="Ne", rate=5e20, duration=5.0)], 8.0)
    case("gas_Ar", [dict(cmd="gas", species="Ar", rate=2e20, duration=5.0)], 8.0)
    case("laser_blowoff_Ti", [dict(cmd="laser_blowoff", species="Ti", atoms=1e19)], 2.0)
    case("wall_flake_W", [dict(cmd="wall_flake", species="W", atoms=3e19)], 8.0)
    case("nbi", [dict(cmd="nbi", P=30.0)], 8.0)
    case("icrh", [dict(cmd="icrh", P=20.0)], 8.0)
    case("ecrh", [dict(cmd="ecrh", P=10.0)], 8.0)
    case("rmp", [dict(cmd="rmp", level=0.8)], 8.0)
    case("ip", [dict(cmd="ip", I_MA=8.0, rate=1.0)], 8.0, keys=("P_fusion_MW", "W_thermal_MJ", "q95", "beta_N"))
    case("density_control", [dict(cmd="density_control", target=1.0)], 15.0, keys=("n20", "P_fusion_MW", "f_GW"))
    case("radiation_control", [dict(cmd="radiation_control", target_MW=60.0)], 15.0, keys=("P_SOL_MW", "P_rad_MW", "Z_eff_avg"))
    case("fishbone", [dict(cmd="fishbone", expelled=0.3, duration=8.0)], 6.0)
    case("alfven", [dict(cmd="alfven", redistributed=0.3, duration=8.0)], 6.0)
    case("marfe", [dict(cmd="marfe", duration=8.0)], 6.0)
    case("back_transition", [dict(cmd="back_transition", ramp=0.5)], 6.0)
    case("transport_barrier", [dict(cmd="transport_barrier", duration=8.0)], 6.0)
    case("rwm", [dict(cmd="rwm", seed=0.01, beta_N_limit=1.5)], 8.0, keys=("beta_N", "W_thermal_MJ"))
    case("vde", [dict(cmd="vde", kick_cm=5.0)], 3.0, keys=("W_thermal_MJ",))
    case("vde_held", [dict(cmd="vde", kick_cm=2.0, gain=50.0, v_max=5.0)], 3.0, keys=("W_thermal_MJ",))
    case("mitigation", [dict(cmd="island", w_cm=8.0), dict(cmd="mitigation", armed=1)], 8.0)
    for atoms in (1e21, 8e21, 2e22, 5e22, 2e23, 1.6e24):
        case(f"spi_Ne_{atoms:.0e}", [dict(cmd="spi", species="Ne", atoms=atoms)], 0.3)
    case("spi_Ne_1e22_H_1e24", [dict(cmd="spi", species="Ne", atoms=1e22, hydrogen=1e24)], 0.3)
    case("spi_D_1.9e24", [dict(cmd="spi", species="D", atoms=1.9e24)], 0.3)


if __name__ == "__main__":
    main()
