"""The harness's quench command as a script or an agent would use it (H78): on a MAST hands-off shot, the fast 2D quench (FreeGSNKE's static equilibria:
which way the plasma goes); on the ITER check plasma, DREAM with neon and deuterium arriving 30 ms after the onset. Each step prints what it read; anything
missing stops the check.
Usage (TORAX's venv): python core/check_quench.py [--only mast|iter] [--shot 20689] [--t_mast 0.02]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twin  # noqa: E402


def need(out, keys, what):
    missing = [k for k in keys if k not in out or out[k] is None]
    if missing:
        raise SystemExit(f"{what}: missing {missing} in {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["mast", "iter"])
    ap.add_argument("--shot", type=int, default=20689)      # a development shot, never a held-out one
    ap.add_argument("--t_mast", type=float, default=0.02, help="s run on the MAST shot before the quench")
    ap.add_argument("--settle", type=float, default=60.0, help="s run on ITER before the quench")
    args = ap.parse_args()
    if args.only != "iter":
        import natural
        t0 = time.perf_counter()
        h = natural.hands_off(args.shot)
        h.apply(dict(cmd="step", until=h.reactor.t + args.t_mast))
        out = h.apply(dict(cmd="quench"))
        need(out, ("dZ_cm", "dR_cm", "group", "start"), "MAST quench")
        print(f"MAST {args.shot} at {h.reactor.t:.3f} s, beta_pol {out['betap']:.3f} from the run: the equilibrium point moves dR {out['dR_cm']:+.1f} cm, dZ {out['dZ_cm']:+.2f} cm; {out['group']}, "
              f"direction {out['direction']}, measured quench for that way {out['cq_80_20_ms_measured_on_mast']} ms; start holds EFIT within {out['start']['dist_cm']:.1f} cm; "
              f"{out['wall_s']:.0f} s for the quench, {time.perf_counter() - t0:.0f} s in all", flush=True)
        card = (h.plant.card or {}).get("quench_2d")
        need(card or {}, ("dZ_cm",), "the disruption card")
        print("on the disruption card: quench_2d", flush=True)
    if args.only != "mast":
        import check_plant
        t0 = time.perf_counter()
        h = twin.Harness(check_plant.scenario(False), keep_history=False)
        h.apply(dict(cmd="step", until=args.settle))
        out = h.apply(dict(cmd="quench", material="neon+deuterium", arrive_ms=30))
        need(out, ("Ire_max_MA", "plasma"), "ITER quench")
        p = out["plasma"]
        print(f"ITER-like run at {h.reactor.t:.0f} s (Ip {p['Ip'] / 1e6:.1f} MA, T0 {p['T0'] / 1e3:.1f} keV, n {p['ne']:.2e}): DREAM, neon and deuterium 30 ms after the onset: "
              f"runaways {out['Ire_max_MA']:.2f} MA at most, current quench {out.get('t_cq_ms')} ms; limits {out['limits']}; {time.perf_counter() - t0:.0f} s", flush=True)
        need((h.plant.card or {}).get("quench_dream") or {}, ("Ire_max_MA",), "the disruption card")
        print("on the disruption card: quench_dream", flush=True)
    print("check_quench: all steps did what they say", flush=True)


if __name__ == "__main__":
    main()
