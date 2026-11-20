"""MAST's gas signals for the twin (H85), from FAIR-MAST level 1 (aga, fairmast.py get aga): the fast ion gauge's molecular density at the wall (fig, D2 per m^3)
and the valve flows (inboard, outboard, total injected; electrons per second and electrons). Writes runs/twin/gas/<shot>.npz, plain numpy, for the KN1D venv.
Usage: $XTORAX_FAIRMAST_PYTHON -I core/gas_extract.py [shots]   (default: the 25 twin shots that have aga downloaded)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fairmast  # noqa: E402

OUT = os.path.join(fairmast.ROOT, "runs", "twin", "gas")


def extract(shot):
    import zarr
    g = zarr.open_group(os.path.join(fairmast.DATA, f"{shot}.zarr", "aga"), mode="r")
    keys = set(g.array_keys())
    arr = lambda k: np.asarray(g[k][...], float) if k in keys else np.array([])

    def time_of(k):                     # each signal carries its own time base in its attributes' dims; FAIR-MAST level 1 keeps them as time, time1, ...
        dims = (g[k].attrs.get("dims") or g[k].attrs.get("_ARRAY_DIMENSIONS") or ["time"])
        d = dims[0] if dims else "time"
        return arr(d) if d in keys else arr("time")
    out = {}
    for k in ("fig", "inboard_total", "outboard_total", "integ_gas", "group1", "group2", "group3", "group4", "group5"):
        if k in keys:
            out[k] = arr(k)
            out[f"t_{k}"] = time_of(k)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, f"{shot}.npz"), **out)
    return out


if __name__ == "__main__":
    shots = [int(a) for a in sys.argv[1:]] or [s for s in fairmast.TWIN if os.path.isdir(os.path.join(fairmast.DATA, f"{s}.zarr", "aga"))]
    dev = fairmast.development()
    for s in shots:
        if dev is not None and s not in dev:
            raise SystemExit(f"refused, not a development shot: {s}")
        o = extract(s)
        fig = o.get("fig", np.array([]))
        tf = o.get("t_fig", np.array([]))
        on = (tf > 0.1) & (tf < 0.3)
        print(f"{s}: fig {len(fig)} samples, median 0.1 to 0.3 s {np.nanmedian(fig[on]) if on.any() else float('nan'):.2e} D2/m3; "
              f"inboard {np.nanmax(o.get('inboard_total', [np.nan])):.2e} e/s max, outboard {np.nanmax(o.get('outboard_total', [np.nan])):.2e} e/s max", flush=True)
