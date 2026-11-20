"""FAIR-MAST level 1 data for the twin (H85): download a diagnostic group of development shots from the public store (s3.echo.stfc.ac.uk, bucket mast,
level1/shots/<shot>.zarr/<group>/), and read it back. Development shots only: runs/twin/dev_shots.json (written by dev_shots.py from the chip project's own split) lists them,
and any other shot is refused, so none of the chip project's held-out shots is ever fetched.

Groups used so far: anb (the beams' analysed power per line, SS and SW, split into full, half and third energy, MW); aga (gas: the fast ion gauge in D2 per m^3, valve flows in electrons
per second); abm (the bolometers).
Usage (its own venv, external/fairmast-venv: zarr 2, numcodecs, requests):
  $XTORAX_FAIRMAST_PYTHON -I core/fairmast.py get anb 22235 20689 ...
  $XTORAX_FAIRMAST_PYTHON -I core/fairmast.py get anb --twin        (the 25 shots of the H82 and d84 sets)
Writes external/data/fairmast_l1/<shot>.zarr/<group>/ (plain Zarr v2 directory stores, data only).
"""
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import requests

ROOT = os.path.abspath(os.environ.get("XTORAX_HOME") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the workspace: runs/twin and external/ (README, "Where things live")
DATA = os.path.join(ROOT, "external", "data", "fairmast_l1")
ENDPOINT = "https://s3.echo.stfc.ac.uk"
BUCKET = "mast"
TWIN = [20689, 16458, 21947, 22786, 20304, 24042, 22248, 15018, 24627, 17859, 20428,
        16228, 22235, 21428, 24608, 16276, 20863, 22056, 24515, 16148, 18886, 24064, 18464, 15977, 22092]
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def development():
    """runs/twin/dev_shots.json's shots if the workspace keeps that list, else None (any shot)."""
    path = os.path.join(ROOT, "runs", "twin", "dev_shots.json")
    return set(json.load(open(path))["development"]) if os.path.exists(path) else None


def keys(prefix):
    """Every object under a prefix, with its size, from S3's ListObjectsV2 (anonymous)."""
    out, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        r = requests.get(f"{ENDPOINT}/{BUCKET}", params=q, timeout=60)
        r.raise_for_status()
        t = ET.fromstring(r.content)
        for c in t.findall(f"{NS}Contents"):
            out.append((c.find(f"{NS}Key").text, int(c.find(f"{NS}Size").text)))
        if (t.findtext(f"{NS}IsTruncated") or "false") != "true":
            return out
        token = t.findtext(f"{NS}NextContinuationToken")


def get(group, shots):
    dev = development()
    bad = [s for s in shots if dev is not None and s not in dev]
    if bad:
        raise SystemExit(f"refused, not development shots: {bad}")
    total = 0
    for s in shots:
        prefix = f"level1/shots/{s}.zarr/{group}/"
        ks = keys(prefix)
        if not ks:
            print(f"{s}: no {group} in level 1", flush=True)
            continue
        def fetch(item):
            k, size = item
            dest = os.path.join(DATA, *k.split("/")[2:])
            if os.path.exists(dest) and os.path.getsize(dest) == size:
                return 0
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            for attempt in range(4):
                try:
                    r = requests.get(f"{ENDPOINT}/{BUCKET}/{k}", timeout=120)
                    r.raise_for_status()
                    break
                except requests.RequestException:
                    if attempt == 3:
                        raise
                    time.sleep(2 * (attempt + 1))
            open(dest, "wb").write(r.content)
            return size
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(8) as pool:                  # the store answers one small file at a time; eight at once
            n = sum(pool.map(fetch, ks))
        # the shot's own root metadata, so zarr can open <shot>.zarr as a group
        for meta in (".zgroup", ".zattrs"):
            dest = os.path.join(DATA, f"{s}.zarr", meta)
            if not os.path.exists(dest):
                r = requests.get(f"{ENDPOINT}/{BUCKET}/level1/shots/{s}.zarr/{meta}", timeout=60)
                if r.ok:
                    open(dest, "wb").write(r.content)
        total += n
        print(f"{s}: {group} {len(ks)} files, {n / 1e6:.2f} MB fetched", flush=True)
    print(f"done: {total / 1e6:.1f} MB in all", flush=True)


def read(shot, group):
    """The group as a dict of numpy arrays (zarr v2 directory store)."""
    import zarr
    g = zarr.open_group(os.path.join(DATA, f"{shot}.zarr", group), mode="r")
    return {k: g[k][...] for k in g.array_keys()}


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "get":
        group = sys.argv[2]
        shots = TWIN if "--twin" in sys.argv else [int(a) for a in sys.argv[3:]]
        get(group, shots)
    else:
        print(__doc__)
