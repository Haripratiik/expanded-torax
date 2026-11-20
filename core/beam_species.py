"""MAST's measured beam power split into full, half and third energy, per beam line (SS, SW) and shot, from FAIR-MAST level 1 (anb, fairmast.py get anb), averaged
over the time each line injects more than 0.2 MW, and the shot's power-weighted mix of both lines. Writes runs/twin/beam_species.json for the twin (beams.Mixed).
Usage: $XTORAX_FAIRMAST_PYTHON -I core/beam_species.py"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fairmast  # noqa: E402

out = {}
for shot in fairmast.TWIN:
    try:
        d = fairmast.read(shot, "anb")
    except Exception as e:
        print(shot, "unreadable", e)
        continue
    t = d["time"]
    row = {}
    tot = np.zeros(3)
    for line in ("ss", "sw"):
        p = np.array([np.nan_to_num(d.get(f"{line}_{c}_power", np.zeros_like(t))) for c in ("full", "half", "third")])
        s = np.nan_to_num(d.get(f"{line}_sum_power", p.sum(axis=0)))
        on = s > 0.2
        if on.sum() < 5:
            row[line] = None
            continue
        e = p[:, on].sum(axis=1)
        tot += e
        row[line] = dict(fractions=(e / e.sum()).round(4).tolist(), mean_MW=float(s[on].mean()), on_s=[float(t[on][0]), float(t[on][-1])])
    if tot.sum() > 0:
        row["mix"] = (tot / tot.sum()).round(4).tolist()
    out[shot] = row
    f = lambda r: "off" if r is None else f"{100 * r['fractions'][0]:4.1f}:{100 * r['fractions'][1]:4.1f}:{100 * r['fractions'][2]:4.1f} ({r['mean_MW']:.2f} MW)"
    print(f"{shot}: SS {f(row['ss'])}   SW {f(row['sw'])}   mix {row.get('mix')}")
json.dump(out, open(os.path.join(fairmast.ROOT, "runs", "twin", "beam_species.json"), "w"), indent=1)
