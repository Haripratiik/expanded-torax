"""EFIT's signals the whole-shot test (H82) needs and the shot files lack, read once from the local MAST store (XTORAX_MAST_STORE): whmd, the shape, beta_pol, beta_N, l_i.
Read only. When the workspace keeps a list of development shots (runs/twin/dev_shots.json), the script stops on any other shot.
Usage (the zarr environment, from the repository): $XTORAX_FAIRMAST_PYTHON -I core/whole_shot_efit.py 20689 16458 ...
Writes runs/twin/whole82/real/<shot>.npz and <shot>_beta.npz.
"""
import json
import os
import sys

import numpy as np
import zarr

sys.dont_write_bytecode = True
STORE = os.path.expanduser(os.environ.get("XTORAX_MAST_STORE", "~/data/mast/tokamark_v1"))   # FAIR-MAST's TokaMark v1 store (zarr)
ROOT = os.path.abspath(os.environ.get("XTORAX_HOME") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "runs", "twin", "whole82", "real")


def development():
    """runs/twin/dev_shots.json's shots if the workspace keeps that list, else None (any shot)."""
    path = os.path.join(ROOT, "runs", "twin", "dev_shots.json")
    return set(json.load(open(path))["development"]) if os.path.exists(path) else None


def main():
    allowed = development()
    os.makedirs(OUT, exist_ok=True)
    for shot in [int(a) for a in sys.argv[1:]]:
        assert allowed is None or shot in allowed, f"{shot} is not a development shot"
        eq = zarr.open_group(os.path.join(STORE, f"{shot}.zarr"), mode="r")["equilibrium"]
        f = lambda k: eq[k][:].astype(float)
        np.savez_compressed(os.path.join(OUT, f"{shot}.npz"), t_eq=f("time"), whmd=f("whmd"), z_mag=f("magnetic_axis_z"), q95=f("q95"), kappa=f("elongation"), a=f("minor_radius"),
                            r_mag=f("magnetic_axis_r"), delta_u=f("triangularity_upper"), delta_l=f("triangularity_lower"))
        np.savez_compressed(os.path.join(OUT, f"{shot}_beta.npz"), t_eq=f("time"), beta_pol=f("beta_pol"), beta_normal=f("beta_normal"), li=f("li"))
        print(shot, len(f("time")), flush=True)


if __name__ == "__main__":
    main()
