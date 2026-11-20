"""The coupled tools through the harness's own commands, as a script or an agent would use them: EPED-NN on the pedestal, FreeGSNKE on the shape, the vertical growth rate, a heating step, a snapshot and its branch, then both switched off. Each step prints what it read, and any step that does not do what it says stops the check.
Usage: python core/check_coupled.py [--settle 60]; with --mast, the same on a real MAST shot and the hands-off run
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_plant  # noqa: E402
import twin  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mast", action="store_true")
    ap.add_argument("--settle", type=float, default=60.0)
    ap.add_argument("--window", type=float, default=5.0)
    args = ap.parse_args()
    t0 = time.perf_counter()
    h = twin.Harness(check_plant.scenario(False), keep_history=False)
    r = h.reactor
    h.apply(dict(cmd="step", until=args.settle))
    o = h.observe()
    print(f"settled in {time.perf_counter() - t0:.0f} s: fusion {o['P_fusion_MW']:.0f} MW, W {o['W_thermal_MJ']:.0f} MJ, beta_N {o['beta_N']:.2f}, T_ped {r.get_param('pedestal.T_e_ped'):.2f} keV, top at {r.get_param('pedestal.rho_norm_ped_top'):.3f}", flush=True)

    t1 = time.perf_counter()
    h.apply(dict(cmd="free_boundary", machine="ITER", mode="control", interval=1.0, growth=True))
    fb = h.observe()["free_boundary"]
    print(f"1. free_boundary on in {time.perf_counter() - t1:.0f} s: shape { {k: round(v, 3) for k, v in fb['shape'].items()} }", flush=True)
    print("   coils, MA-turns:", {k: round(v / 1e6, 2) for k, v in fb["coils_A"].items()}, flush=True)
    print("   vertical stability:", fb["stability"], flush=True)
    assert fb["stability"] and fb["stability"]["growth_per_s"], "no growth rate"
    assert abs(fb["shape"]["a"] - 2.0) < 0.3, "wrong plasma"

    h.apply(dict(cmd="pedestal_model", name="EPED1NNmodel", interval=0.5))
    h.apply(dict(cmd="step", until=r.t + args.window))
    o = h.observe()
    print(f"2. pedestal_model on, {args.window} s later: EPED-NN { {k: (round(v, 3) if isinstance(v, float) else v) for k, v in o['eped'].items() if k in ('p_kPa', 'width', 'T_e_ped', 'n_e_ped', 'rho_top', 'extrapolation', 'worst')} }; fusion {o['P_fusion_MW']:.0f} MW, W {o['W_thermal_MJ']:.0f} MJ, beta_N {o['beta_N']:.2f}", flush=True)
    fb = o["free_boundary"]
    print(f"   shape after it: kappa {fb['shape']['kappa']:.3f}, R_out {fb['shape']['R_out']:.3f}; solves {fb['solves']}, rejected {fb['rejected']}; coils moved, kA-turns: { {k: round(v / 1e3) for k, v in fb['coils_moved_A'].items() if abs(v) > 1e3} }", flush=True)
    assert abs(r.get_param("pedestal.T_e_ped") - o["eped"]["T_e_ped"]) < 1e-6, "the pedestal did not take the model's height"
    assert not r.sim_error or "NO_ERROR" in str(r.sim_error), r.sim_error

    h.apply(dict(cmd="snapshot", name="a"))
    W_a = h.observe()["W_thermal_MJ"]
    h.apply(dict(cmd="nbi", P=30.0))
    h.apply(dict(cmd="step", until=r.t + args.window))
    o = h.observe()
    print(f"3. 30 MW more for {args.window} s: W {W_a:.0f} to {o['W_thermal_MJ']:.0f} MJ; kappa {o['free_boundary']['shape']['kappa']:.3f}", flush=True)
    h.apply(dict(cmd="restore", name="a"))
    print(f"4. back to the snapshot: W {h.observe()['W_thermal_MJ']:.0f} MJ against {W_a:.0f}", flush=True)
    assert abs(h.observe()["W_thermal_MJ"] - W_a) < 1e-6, "the branch did not come back"

    h.apply(dict(cmd="pedestal_model", name="off"))
    h.apply(dict(cmd="free_boundary", mode="off"))
    h.apply(dict(cmd="step", until=r.t + 1.0))
    o = h.observe()
    assert "free_boundary" not in o and "eped" not in o, "still on"
    print(f"5. both off, and it runs on: t {r.t:.1f}, fusion {o['P_fusion_MW']:.0f} MW, {r.sim_error}", flush=True)
    print("all steps did what they say", flush=True)


def mast(shot=20689, window=0.03):     # a development shot: never one of the chip project's held-out shots
    """The same command on a real MAST shot, and the hands-off run the front end loads: FreeGSNKE on the shot's coil traces, against EFIT's boundary."""
    import numpy as np
    import natural
    import ntm
    t0 = time.perf_counter()
    h = natural.hands_off(shot)
    r = h.reactor
    h.apply(dict(cmd="step", until=r.t + window))
    o = h.observe()["free_boundary"]
    print(f"hands-off {shot}, {window * 1e3:.0f} ms in {time.perf_counter() - t0:.0f} s: R_out {o['shape']['R_out']:.3f}, kappa {o['shape']['kappa']:.2f}; solves {o['solves']}, rejected {o['rejected']}; EFIT's boundary has {len(o['measured_lcfs'])} points, out to {max(x[0] for x in o['measured_lcfs']):.3f}", flush=True)
    assert o["solves"] >= 2 and abs(o["shape"]["R_out"] - max(x[0] for x in o["measured_lcfs"])) < 0.1, "the boundary went elsewhere"
    h.free.close()
    h.free = None
    ntm.TURBULENCE[:] = 1.0
    t0 = time.perf_counter()
    h = twin.Harness(dict(name="check", mast=dict(shot=shot, model="bgb", past=0.0, natural=True, shots=natural.SHOTS), beams=True, dt=1e-3, edge=False, elms=False, impurities={"C": 0.0167, "Ne": 0.0, "Ar": 0.0}, physics={"a_bs": 1.5, "a_cd": 0.5}, rotation=False, gyrotrons={"units": 4, "unit_MW": 0.25, "mirror_rate": 2.0}, sensors={"sigma_w_cm": 0.0, "sigma_rho_cm": 0.0, "latency_steps": 0, "seed": 1}, decision={"kind": "none"}, events=[]), keep_history=False)
    r = h.reactor
    h.apply(dict(cmd="step", until=r.t + 0.005))
    h.apply(dict(cmd="free_boundary", machine="MAST", interval=0.01))
    h.apply(dict(cmd="step", until=r.t + window))
    o = h.observe()["free_boundary"]
    print(f"harness command on {shot}, {window * 1e3:.0f} ms in {time.perf_counter() - t0:.0f} s: R_out {o['shape']['R_out']:.3f}, kappa {o['shape']['kappa']:.2f}; solves {o['solves']}, rejected {o['rejected']}", flush=True)
    assert o["solves"] >= 2, "no solves went through"
    h.apply(dict(cmd="free_boundary", mode="off"))
    h.apply(dict(cmd="step", until=r.t + 0.005))
    assert "free_boundary" not in h.observe() and not r.sim_error or "NO_ERROR" in str(r.sim_error), "off did not hold"
    print("MAST: all steps did what they say", flush=True)


if __name__ == "__main__":
    if "--mast" in sys.argv:
        mast()
    else:
        main()
