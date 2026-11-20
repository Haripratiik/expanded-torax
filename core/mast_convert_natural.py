"""Shot files for the natural test (H43): the 38 shots of runs/twin/mast_catalog/natural_test_shots.json, each over its window, through mast.convert, plus the same extras as mast_extras.py.
Usage (WSL, read-only on the store): python core/mast_convert_natural.py
Writes runs/twin/mast_natural2/shots/<shot>.npz, runs/twin/mast_natural2/conversion.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
import mast  # noqa: E402

OUT = os.path.join(mast.ROOT, "runs", "twin", "mast_natural2")


def main():
    picked = json.load(open(os.path.join(mast.ROOT, "runs", "twin", "mast_catalog", "natural_test_shots.json")))["shots"]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "conversion.json")
    infos = json.load(open(path)) if os.path.exists(path) else {}
    for s in picked:
        if str(s["shot"]) in infos:
            continue
        entry = dict(shot=s["shot"], t0=s["window"][0], t1=s["window"][1], campaign=s.get("campaign", ""), shape=s.get("shape", ""), heating=s.get("heating", ""))
        try:
            infos[str(s["shot"])] = mast.convert(entry, OUT)
        except Exception as err:
            print(s["shot"], "failed", repr(err)[:300], flush=True)
        json.dump(infos, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
