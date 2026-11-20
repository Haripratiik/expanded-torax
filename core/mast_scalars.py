"""Per-shot scalars on EFIT's 5 ms grid for the laws behind the natural test (H43): what a trigger may read from the plasma state. WSL, read-only on the store.
Usage: python core/mast_scalars.py
Writes runs/twin/mast_natural2/scalars.npz: one array per field and shot, keyed <shot>_<field>.
"""
import json
import os
import sys

import numpy as np
import zarr

sys.dont_write_bytecode = True
STORE = os.path.expanduser(os.environ.get("XTORAX_MAST_STORE", "~/data/mast/tokamark_v1"))   # FAIR-MAST's TokaMark v1 store (zarr)
OUT = os.path.join("runs", "twin", "mast_natural2")


def cell(tf, v, t):
    edges = np.concatenate([[t[0] - 0.0025], 0.5 * (t[1:] + t[:-1]), [t[-1] + 0.0025]])
    idx = np.searchsorted(edges, tf) - 1
    ok = (idx >= 0) & (idx < len(t)) & np.isfinite(v)
    n = np.bincount(idx[ok], minlength=len(t))
    return np.where(n > 0, np.bincount(idx[ok], weights=v[ok], minlength=len(t)) / np.maximum(n, 1), np.nan)


def main():
    shots = [int(k) for k in json.load(open(os.path.join("runs", "twin", "mast_catalog", "catalog.json")))["shots"]]
    out = {}
    for shot in shots:
        try:
            r = zarr.open_group(os.path.join(STORE, f"{shot}.zarr"), mode="r")
            eq, su = r["equilibrium"], r["summary"]
            t = eq["time"][:].astype(float)
            d = dict(t=t, beta_n=eq["beta_normal"][:], li=eq["li"][:], q95=eq["q95"][:], q0=eq["q_axis"][:], W_dia=eq["whmd"][:], B=np.abs(eq["bvac_rmag"][:]), a=eq["minor_radius"][:], kappa=eq["elongation"][:], R=eq["magnetic_axis_r"][:], Z=eq["magnetic_axis_z"][:], V_loop=eq["vloop_static"][:])
            q = eq["q"][:]
            d["q_min"] = np.nanmin(np.where(q > 0, q, np.nan), axis=0)
            tf = su["time"][:]
            for name, key in (("Ip", "ip"), ("P_nbi", "power_nbi"), ("nel", "line_average_n_e"), ("nGW", "greenwald_density"), ("P_rad", "power_radiated"), ("neutrons", "neutron_rates_total")):
                if key in su:
                    d[name] = cell(tf, su[key][:].astype(float), t)
            if "gas_injection" in r and "total_injected" in r["gas_injection"]:
                d["gas"] = cell(r["gas_injection"]["time"][:].astype(float), r["gas_injection"]["total_injected"][:].astype(float), t)
            for k, v in d.items():
                out[f"{shot}_{k}"] = np.asarray(v, np.float32 if k != "t" else np.float64)
        except Exception as err:
            print(shot, "failed", repr(err)[:200], flush=True)
    np.savez_compressed(os.path.join(OUT, "scalars.npz"), **out)
    print("done", len(shots), flush=True)


if __name__ == "__main__":
    main()
