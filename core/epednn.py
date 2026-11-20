"""EPED-NN, the open-source pedestal model (ProjectTorreyPines/EPEDNN.jl, Apache-2.0: a neural network over the EPED peeling-ballooning and kinetic-ballooning model, trained on a multi-machine database), run in Python from the weights epednn_export.jl writes out. No Julia at run time, and no fitting here.
Two models: "EPED1NNmodel", pressure and width; "EPED1NNensemble", pressure with its uncertainty, and separatrix inputs, trained further on ITER-scale runs. Both are a power law in log space plus a network. Each call says how far outside the training bounds it is, and a far point should not be believed: all spherical tokamak plasmas are.
Inputs in the model's units: a (m), betan, bt (T), delta (effective: 2/3 of the smaller plus 1/3 of the larger of upper and lower), ip (MA), kappa, m (ion mass, amu), neped (1e19 m^-3), r (m), zeffped; for the ensemble also nesep_ratio and tesep (eV).
Check: python core/epednn.py check, against the numbers EPEDNN.jl gave for the same inputs (data/epednn/check.json).
"""
import json
import os
import sys

import numpy as np
from scipy.special import erf

ROOT = os.path.abspath(os.environ.get("XTORAX_HOME") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the workspace: runs/twin and external/ (README, "Where things live")
FOLDER = os.environ.get("XTORAX_EPEDNN") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "epednn")   # the weights, shipped in data/epednn
SOLUTIONS = ("GH_H", "GH_meta", "GH_superH", "G_H", "G_meta", "G_superH", "H_H", "H_meta", "H_superH")


def gelu_tanh(x):
    """NNlib's gelu, the approximate form EPED1NNmodel was built with."""
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def gelu_erf(x):
    """The exact form the ensemble was trained with."""
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


class Model:
    def __deepcopy__(self, memo):
        return self                                    # a branch shares it: it holds a process, a pipe or fixed weights, and is never copied

    def __init__(self, name="EPED1NNmodel"):
        m = json.load(open(os.path.join(FOLDER, name + ".json")))
        self.name, self.xnames, self.ynames = name, m["xnames"], m["ynames"]
        self.xm, self.xs, self.ym, self.ys = (np.array(m[k], float) for k in ("xm", "xs", "ym", "ys"))
        self.bounds, self.yp = np.array(m["xbounds"], float), np.array(m["yp"], float)
        self.nets = [[(np.array(l["weight"], float), np.array(l["bias"], float)) for l in net] for net in json.load(open(os.path.join(FOLDER, name + "_nets.json")))]
        self.activation = gelu_erf if name == "EPED1NNensemble" else gelu_tanh

    def vector(self, **inputs):
        return np.array([float(inputs[k]) for k in self.xnames])

    def extrapolation(self, x):
        """How far outside the training bounds, as a fraction of each input's range; the largest and its name."""
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        d = np.maximum(0.0, np.maximum((lo - x) / (hi - lo), (x - hi) / (hi - lo)))
        d[d < 1e-6] = 0.0
        return float(d.max()), self.xnames[int(np.argmax(d))]

    def members(self, x):
        """Outputs of each network, one row per network: pressure in MPa, and for the single model also the width in psi_N."""
        xx = np.array(x, float)
        xx[self.xnames.index("delta")] += 1.0
        xx = np.abs(xx)
        y0 = 10.0 ** (self.yp[:, 0] + self.yp[:, 1:] @ np.log10(xx))
        n = (xx - self.xm) / self.xs
        out = []
        for net in self.nets:
            h = n
            for k, (w, b) in enumerate(net):
                h = w @ h + b
                if k < len(net) - 1:
                    h = self.activation(h)
            y = (y0 + h * self.ys + self.ym) ** 2
            count = 9 if len(y) == 18 else len(y)
            y[:count] *= xx[self.xnames.index("neped")]
            out.append(y)
        return np.array(out)

    def __call__(self, solution="GH_H", **inputs):
        """Pedestal pressure in Pa, half width in psi_N (from the analytic width law if the model gives none, then None here), and the warnings."""
        x = self.vector(**inputs)
        y = self.members(x)
        far, worst = self.extrapolation(x)
        if y.shape[1] == 18:
            i = SOLUTIONS.index(solution)
            return dict(p=float(y[0, i]) * 1e6, width=float(y[0, 9 + i]), sigma=None, extrapolation=far, worst=worst)
        return dict(p=float(y[:, 0].mean()) * 1e6, width=None, sigma=float(y[:, 0].std()) * 1e6, extrapolation=far, worst=worst)


def effective_delta(lower, upper):
    return (2.0 / 3.0) * min(lower, upper) + (1.0 / 3.0) * max(lower, upper)


def width_law(beta_pol_ped):
    """EPED1's analytic width: half width in psi_N."""
    return 0.076 * np.sqrt(beta_pol_ped)


def check():
    rows = json.load(open(os.path.join(FOLDER, "check.json")))
    single, ensemble = Model("EPED1NNmodel"), Model("EPED1NNensemble")
    names = ["a", "betan", "bt", "delta", "ip", "kappa", "m", "neped", "r", "zeffped", "nesep_ratio", "tesep"]
    worst_single, worst_ens, worst_far = 0.0, 0.0, 0.0
    for r in rows:
        inputs = dict(zip(names, r["x"]))
        y = single.members(single.vector(**inputs))[0]
        worst_single = max(worst_single, float(np.max(np.abs(y - np.array(r["single"])) / np.abs(np.array(r["single"])))))
        e = ensemble.members(ensemble.vector(**inputs))[:, 0]
        worst_ens = max(worst_ens, abs(e.mean() - r["ensemble_height"]) / r["ensemble_height"])
        worst_far = max(worst_far, abs(single.extrapolation(single.vector(**inputs))[0] - r["single_extrapolation"]))
    print(f"{len(rows)} points against Julia: largest relative difference, single {worst_single:.2e}, ensemble {worst_ens:.2e}; extrapolation {worst_far:.2e}")
    iter15 = dict(a=2.0, betan=2.0, bt=5.3, delta=0.49, ip=15.0, kappa=1.85, m=2.5, neped=7.0, r=6.2, zeffped=1.5, nesep_ratio=0.25, tesep=75.0)
    print("the package's own regression value at ITER 15 MA, 0.07717931845709872 MPa:", single(**iter15)["p"] / 1e6, "width 0.03263137598834528:", single(**iter15)["width"])
    print("bounds of the single model:", {n: tuple(np.round(b, 2)) for n, b in zip(single.xnames, single.bounds)})


if __name__ == "__main__":
    if sys.argv[1:] == ["check"]:
        check()
