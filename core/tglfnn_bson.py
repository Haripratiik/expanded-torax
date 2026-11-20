"""Read TurbulentTransport.jl's network files (Julia BSON) as plain data, with no Julia and nothing executed from them (H70). Julia's precompile of Flux failed on the Windows machine this was developed on (an lld error), so the format is read directly: standard BSON, with BSON.jl's tags for arrays, structs, symbols and shared references.

  dump <model>       the tree of tags and keys, to see the layout
  export <model>...  runs/twin/tglfnn/<model>.json: per ensemble member, the network as a plain tree and the scalings, names and bounds
  check              the port against the package's own regression values

Needs TurbulentTransport.jl's repository in the workspace's external/ (git clone https://github.com/ProjectTorreyPines/TurbulentTransport.jl).
The transport option tglfnn_st (tglfnn_st.py) reads three models: sat3_em_mastu+nstx_azf-1_withnegD, sat3_em_mastunearedge+nstxnearedge_azf-1_withnegD
and sat3_em_mastuedge+nstxedge_azf-1_withnegD; check also needs sat0_em_d3d and sat3_em_d3d_azf-1.

Usage (TORAX's environment, from the repository): python -I core/tglfnn_bson.py dump|export <model> ... | check
"""
import json
import os
import struct
import sys

import numpy as np

ROOT = os.path.abspath(os.environ.get("XTORAX_HOME") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the workspace (README.md, "Set environment variables")
MODELS = os.path.join(ROOT, "external", "TurbulentTransport.jl", "models")
OUT = os.path.join(ROOT, "runs", "twin", "tglfnn")


# ---------- standard BSON
def _cstring(b, i):
    j = b.index(b"\x00", i)
    return b[i:j].decode("utf-8"), j + 1


def _doc(b, i):
    n = struct.unpack_from("<i", b, i)[0]
    end = i + n - 1
    i += 4
    out = {}
    while i < end:
        t = b[i]
        i += 1
        key, i = _cstring(b, i)
        if t == 0x01:
            v = struct.unpack_from("<d", b, i)[0]
            i += 8
        elif t == 0x02:
            m = struct.unpack_from("<i", b, i)[0]
            v = b[i + 4:i + 4 + m - 1].decode("utf-8")
            i += 4 + m
        elif t in (0x03, 0x04):
            v, i = _doc(b, i)
            if t == 0x04:
                v = [v[k] for k in sorted(v, key=int)]
        elif t == 0x05:
            m = struct.unpack_from("<i", b, i)[0]
            v = bytes(b[i + 5:i + 5 + m])
            i += 5 + m
        elif t == 0x08:
            v = bool(b[i])
            i += 1
        elif t == 0x09:
            v = struct.unpack_from("<q", b, i)[0]
            i += 8
        elif t == 0x0A:
            v = None
        elif t == 0x10:
            v = struct.unpack_from("<i", b, i)[0]
            i += 4
        elif t == 0x12:
            v = struct.unpack_from("<q", b, i)[0]
            i += 8
        else:
            raise ValueError(f"BSON type {t:#x} not handled at {i}")
        out[key] = v
    return out, end + 1


def load_raw(path):
    b = open(path, "rb").read()
    d, _ = _doc(b, 0)
    return d


# ---------- BSON.jl's tags
NUMERIC = {"Float64": "<f8", "Float32": "<f4", "Float16": "<f2", "Int64": "<i8", "Int32": "<i4", "UInt8": "u1", "Bool": "?"}


def type_name(t):
    if isinstance(t, dict) and t.get("tag") == "datatype":
        return ".".join(t["name"])
    if isinstance(t, dict) and t.get("tag") == "ref":
        return "ref"
    return str(t)


class Decoder:
    """Resolves BSON.jl's tags into numpy arrays, lists, dicts and (type name, fields) pairs. Shared objects (tag "backref") come from the file's _backrefs."""

    def __init__(self, raw):
        self.refs = raw.get("_backrefs", [])
        self.cache = {}

    def raw(self, x):
        """Follow shared references to the raw tagged object."""
        while isinstance(x, dict) and x.get("tag") == "backref":
            x = self.refs[x["ref"] - 1]
        return x

    def tname(self, t):
        t = self.raw(t)
        if isinstance(t, dict) and t.get("tag") == "datatype":
            return ".".join(t["name"])
        return str(t)

    def __call__(self, x):
        if isinstance(x, list):
            return [self(a) for a in x]
        if not isinstance(x, dict):
            return x
        tag = x.get("tag")
        if tag == "backref":
            k = x["ref"]
            if k not in self.cache:
                self.cache[k] = self(self.refs[k - 1])
            return self.cache[k]
        if tag == "array":
            et = self.tname(x["type"]).split(".")[-1]
            size = [int(s) for s in x["size"]]
            if isinstance(x["data"], (bytes, bytearray)) and et in NUMERIC:
                a = np.frombuffer(x["data"], dtype=NUMERIC[et])
                return a.reshape(size, order="F") if len(size) > 1 else a.copy()
            return [self(a) for a in x["data"]]
        if tag == "symbol":
            return ":" + x["name"]
        if tag == "struct":
            t = self.raw(x["type"])
            name = self.tname(t)
            fields = [self(a) for a in x.get("data", [])]
            if name.endswith("Base.Dict") and len(fields) == 2:
                return {(k if not isinstance(k, np.ndarray) else str(k)): v for k, v in zip(fields[0], fields[1])}
            return dict(_type=name, _params=[self.tname(p) for p in (t.get("params", []) if isinstance(t, dict) else [])], _fields=fields)
        if tag == "datatype":
            return dict(_datatype=self.tname(x))
        if tag in ("tuple", "svec"):
            return tuple(self(a) for a in x["data"])
        if tag == "dict":
            keys, vals = self(x["keys"]), self(x["values"])
            return {(k if not isinstance(k, str) else k): v for k, v in zip(keys, vals)}
        if tag == "ref":
            return dict(_ref=x.get("path"))
        return {k: self(v) for k, v in x.items()}


def show(x, depth=0, name="", limit=6):
    pad = "  " * depth
    if isinstance(x, np.ndarray):
        print(f"{pad}{name}: array {x.dtype} {x.shape}")
    elif isinstance(x, dict) and "_type" in x:
        print(f"{pad}{name}: struct {x['_type']} {x['_params'][:3] if x['_params'] else ''} ({len(x['_fields'])} fields)")
        for i, f in enumerate(x["_fields"][:12]):
            show(f, depth + 1, f"field {i}", limit)
    elif isinstance(x, dict):
        print(f"{pad}{name}: dict {list(x.keys())[:12]}")
        for k in list(x.keys())[:limit if depth > 0 else 40]:
            show(x[k], depth + 1, str(k), limit)
    elif isinstance(x, (list, tuple)):
        print(f"{pad}{name}: list of {len(x)}" + (f", first {str(x[:4])[:120]}" if x and not isinstance(x[0], (dict, list, tuple, np.ndarray)) else ""))
        if x and isinstance(x[0], (dict, list, tuple, np.ndarray)):
            for i, a in enumerate(x[:limit]):
                show(a, depth + 1, f"[{i}]", limit)
    else:
        print(f"{pad}{name}: {str(x)[:120]}")


def decoded(model):
    raw = load_raw(os.path.join(MODELS, model + ".bson"))
    dec = Decoder(raw)
    top = {k: v for k, v in raw.items() if k != "_backrefs"}
    return dec(top) if "tag" in top else {k: dec(v) for k, v in top.items()}


def dump(model):
    show(decoded(model), name=model)


ACTIVATIONS = ("relu", "tanh", "sigmoid", "gelu", "elu", "swish", "softplus", "identity", "leakyrelu", "selu", "celu", "tanh_fast", "sigmoid_fast")


def _fname(t):
    """A serialized function's name: NNlib.#elu gives elu, Main.Base.#+ gives +."""
    return t.split("#")[-1] if "#" in t else t.split(".")[-1]


def _node(x):
    """A Flux model as a plain tree: dense (weight, bias, activation), chain (parts), parallel (operator, branches), function (name)."""
    if isinstance(x, dict) and "_type" in x:
        t = x["_type"]
        if t.endswith("Flux.Dense"):
            w, b, f = x["_fields"][:3]
            return dict(kind="dense", weight=np.asarray(w, float).tolist(), bias=np.asarray(b, float).tolist(), activation=_fname(f["_type"]))
        if t.endswith("Flux.Chain"):
            return dict(kind="chain", parts=[_node(a) for a in x["_fields"][0]])
        if t.endswith("Flux.Parallel"):
            return dict(kind="parallel", op=_fname(x["_fields"][0]["_type"]), branches=[_node(a) for a in x["_fields"][1]])
        if not x["_fields"]:
            return dict(kind="function", name=_fname(t))
        raise ValueError(f"layer {t} not handled")
    raise ValueError(f"not a layer: {str(x)[:80]}")


def _member(d):
    m = dict(net=_node(d[":fluxmodel"]))
    for k in ("xnames", "ynames", "xm", "xσ", "ym", "yσ", "xbounds", "ybounds", "yscale", "vexb_convention", "name"):
        key = ":" + k
        if key in d:
            v = d[key]
            m[k.replace("σ", "s")] = np.asarray(v, float).tolist() if isinstance(v, np.ndarray) else v
    return m


# ---------- the same arithmetic as TurbulentTransport.jl's flux_array, in numpy, to check the port
def act(name, x):
    if name == "elu":
        return np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))
    if name == "identity":
        return x
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "tanh":
        return np.tanh(x)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    raise ValueError(name)


def run_net(n, x):
    k = n["kind"]
    if k == "dense":
        return act(n["activation"], np.asarray(n["weight"]) @ x + np.asarray(n["bias"]))
    if k == "chain":
        for p in n["parts"]:
            x = run_net(p, x)
        return x
    if k == "parallel":
        outs = [run_net(b, x) for b in n["branches"]]
        if n["op"] != "+":
            raise ValueError(n["op"])
        return sum(outs)
    if k == "function":
        return act(n["name"], x)
    raise ValueError(k)


def run_member(m, x):
    x = np.asarray(x, float).copy()
    for i, name in enumerate(m["xnames"]):
        if "_log10" in name:
            x[i] = np.log10(x[i])
    z = run_net(m["net"], (x - np.asarray(m["xm"])) / np.asarray(m["xs"]))
    y = z * np.asarray(m["ys"]) + np.asarray(m["ym"])
    ys = m.get("yscale") or []
    return np.asarray(ys) * np.sinh(y) if len(ys) else y


def mid_input(m):
    """The input the package's tests use: each feature at the middle of its training bounds."""
    b = np.asarray(m["xbounds"])
    mid = 0.5 * (b[:, 0] + b[:, 1])
    return np.array([10 ** v if "_log10" in n else v for n, v in zip(m["xnames"], mid)])


def check():
    """Against the package's own regression values (test/fixtures.jl): sat0_em_d3d's first member, and the mean of sat3_em_d3d_azf-1's members, at the middle of the bounds."""
    export(["sat0_em_d3d", "sat3_em_d3d_azf-1"], quiet=True)
    a = json.load(open(os.path.join(OUT, "sat0_em_d3d.json")))["members"][0]
    got = run_member(a, mid_input(a))
    want = np.array([0.020119015925778916, 0.23107953401865972, 0.7410156911426711, 0.6620782527214439])
    print("single member: got", np.round(got, 6).tolist(), "want", np.round(want, 6).tolist(), "largest relative difference", float(np.max(np.abs(got - want) / np.abs(want))))
    e = json.load(open(os.path.join(OUT, "sat3_em_d3d_azf-1.json")))["members"]
    got = np.mean([run_member(m, mid_input(e[0])) for m in e], axis=0)
    want = np.array([-0.019443083518266357, 0.04080973584865324, 0.23715745308655625, 0.07256906359102389])
    print("ensemble mean:", len(e), "members, got", np.round(got, 6).tolist(), "want", np.round(want, 6).tolist(), "largest relative difference", float(np.max(np.abs(got - want) / np.abs(want))))


def export(models, quiet=False):
    os.makedirs(OUT, exist_ok=True)
    for model in models:
        d = decoded(model)
        keys = list(d.keys())
        if all(str(k).isdigit() for k in keys):
            members = [_member(d[k]) for k in sorted(keys, key=int)]
        else:
            members = [_member(d)]
        json.dump(dict(model=model, members=members), open(os.path.join(OUT, model + ".json"), "w"))
        if quiet:
            continue
        m = members[0]
        print(f"{model}: {len(members)} members, network {m['net']['kind']} of {len(m['net'].get('parts', []))} parts")
        print("  inputs", len(m.get("xnames", [])), ":", " ".join(m.get("xnames", [])))
        print("  outputs", len(m.get("ynames", [])), ":", " ".join(m.get("ynames", [])))
        print("  yscale", len(m.get("yscale", []) or []), "; vexb", m.get("vexb_convention"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1] == "dump":
        dump(sys.argv[2])
    elif sys.argv[1] == "check":
        check()
    else:
        export(sys.argv[2:])
