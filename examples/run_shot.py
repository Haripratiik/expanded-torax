"""A MAST shot hands-off from Python, with the recommended version: the run natural.py makes, stepped here. Needs MAST shot files
in the workspace (README.md, "MAST shots"), and FreeGSNKE.
Usage (from the repository): python examples/run_shot.py [shot] [ms]"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
import natural  # noqa: E402

shot = int(sys.argv[1]) if len(sys.argv) > 1 else 20689
span = float(sys.argv[2]) / 1e3 if len(sys.argv) > 2 else 0.03
h = natural.hands_off(shot)                         # natural.RECOMMENDED; FreeGSNKE on it as h.free
r = h.reactor
print(f"shot {shot}, version {natural.RECOMMENDED}, start at t = {r.t:.4f} s", flush=True)
end = r.t + span
while r.t < end:
    h.apply({"cmd": "step", "n": 5})
    rec = r.records[-1]
    print(f"t {r.t:.4f}  n_line {rec['n_line']:.2e}  T_e0 {rec['T_e0']:.3f} keV  W {rec['W_thermal'] / 1e3:.1f} kJ  q95 {rec['q95']:.2f}  H {rec.get('H', 0):.0f}  "
          f"radiated (bolometer reading) {rec.get('rad_bolo', float('nan')) / 1e3:.0f} kW", flush=True)
o = h.observe()
print("free boundary:", {k: round(v, 3) for k, v in o["free_boundary"]["shape"].items()} if "free_boundary" in o else "off")
if h.free is not None:
    h.free.close()
