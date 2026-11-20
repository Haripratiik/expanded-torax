"""TGLF-NN for spherical tokamaks, in TORAX (H70, docs/digital-twin.md "Improving it").

The networks are ProjectTorreyPines' (TurbulentTransport.jl, Apache-2.0; Neiser et al.): TGLF SAT3, electromagnetic, trained on MAST-U and NSTX profiles,
positive and negative triangularity, one network for the core (r/a below 0.881), one for the near edge (to 0.975) and one for the edge, each the mean of
20 members, blended by radius as that package does. They were written out as plain numbers by tglfnn_bson.py, whose port matches the package's own test
values to 1e-14. Nothing of ours is fitted: the inputs are TORAX's TGLF inputs (tglf_based_transport_model, GACODE's definitions and units, which are
the ones the networks were trained on), and the fluxes go back through TORAX's own conversion.

What TORAX does not have is set to zero: squareness and its shear, the vertical shift and its gradient, and the parallel flow and its shear. ExB shear is
TORAX's. MAST's plasmas rotate, and with no flow shear the networks give more transport than they would with it; the anchor at the start takes up the
level, and the networks say how transport changes afterwards. Inputs are held inside the training bounds, which the package only warns about.

Model names: "tglfnn-st" alone, "tglfnn-st_table" with the turbulence table and the islands' conduction (ntm.IslandTransport), as the other models of
mast_anchor.BASE; "combined_table" for TORAX's combined model under the same table. On MAST the networks' inputs leave their training range beyond about
rho 0.65 (collisionality above, beta below, q above what MAST-U and NSTX gave them), and there they give fluxes thousands of gyro-Bohm units wide. So
mast_anchor's "tglfnn_st" uses them inside RHO_SWITCH and Bohm-gyroBohm outside, through TORAX's combined model.
"""
import dataclasses
import json
import os
from typing import Annotated, Literal

import jax
import jax.numpy as jnp
import numpy as np
from torax import transport
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import pydantic_model as transport_pydantic
from torax._src.transport_model import tglf_based_transport_model
from torax._src.transport_model import tglfnn_ukaea_transport_model

ROOT = os.path.abspath(os.environ.get("XTORAX_HOME") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the workspace: runs/twin and external/ (README, "Where things live")
FOLDER = os.path.join(ROOT, "runs", "twin", "tglfnn")
MODELS = ("sat3_em_mastu+nstx_azf-1_withnegD", "sat3_em_mastunearedge+nstxnearedge_azf-1_withnegD", "sat3_em_mastuedge+nstxedge_azf-1_withnegD")
RHO_SWITCH = 0.6                                         # mast_anchor's tglfnn_st: the networks inside, Bohm-gyroBohm outside (see above)
BLEND = (0.881, 0.975)                                   # RMIN_LOC where the near-edge, then the edge network take over (TurbulentTransport.jl)
ZERO = ("DZMAJDX_LOC", "S_ZETA_LOC", "ZETA_LOC", "ZMAJ_LOC", "VPAR_1", "VPAR_SHEAR_1")


def _tree(n):
    """A network as nested tuples of arrays, the structure kept in the tags."""
    k = n["kind"]
    if k == "dense":
        return ("dense", jnp.asarray(n["weight"]), jnp.asarray(n["bias"]), n["activation"])
    if k == "chain":
        return ("chain", tuple(_tree(p) for p in n["parts"]))
    if k == "parallel":
        return ("parallel", n["op"], tuple(_tree(b) for b in n["branches"]))
    return ("function", n["name"])


def _act(name, x):
    if name == "elu":
        return jnp.where(x > 0, x, jnp.expm1(jnp.minimum(x, 0.0)))
    if name == "identity":
        return x
    if name == "relu":
        return jnp.maximum(x, 0.0)
    if name == "tanh":
        return jnp.tanh(x)
    raise ValueError(name)


def _run(node, x):
    """x: (features, points)."""
    k = node[0]
    if k == "dense":
        return _act(node[3], node[1] @ x + node[2][:, None])
    if k == "chain":
        for p in node[1]:
            x = _run(p, x)
        return x
    if k == "parallel":
        if node[1] != "+":
            raise ValueError(node[1])
        return sum(_run(b, x) for b in node[2])
    return _act(node[1], x)


@dataclasses.dataclass(frozen=True, eq=False)
class Network:
    xnames: tuple
    ynames: tuple
    log: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    xm: np.ndarray
    xs: np.ndarray
    ym: np.ndarray
    ys: np.ndarray
    yscale: np.ndarray
    members: tuple

    @classmethod
    def load(cls, name):
        d = json.load(open(os.path.join(FOLDER, name + ".json")))["members"]
        m = d[0]
        b = np.asarray(m["xbounds"], float)
        return cls(tuple(m["xnames"]), tuple(m["ynames"]), np.array(["_log10" in x for x in m["xnames"]]), b[:, 0], b[:, 1], np.asarray(m["xm"], float), np.asarray(m["xs"], float), np.asarray(m["ym"], float), np.asarray(m["ys"], float), np.asarray(m.get("yscale") or [], float), tuple(_tree(a["net"]) for a in d))

    def __call__(self, feats):
        """feats: name (without _log10) to an array over faces. Returns the outputs (ynames, faces), the members' mean, as flux_array does."""
        rows = []
        for name, is_log in zip(self.xnames, self.log):
            v = feats[name.replace("_log10", "")]
            rows.append(jnp.log10(jnp.maximum(v, 1e-30)) if is_log else v)
        x = jnp.clip(jnp.stack(rows), self.lo[:, None], self.hi[:, None])
        z = (x - self.xm[:, None]) / self.xs[:, None]
        out = 0.0
        for m in self.members:
            y = _run(m, z) * self.ys[:, None] + self.ym[:, None]
            out = out + (self.yscale[:, None] * jnp.sinh(y) if len(self.yscale) else y)
        return out / len(self.members)


@dataclasses.dataclass(frozen=True, eq=False)
class TGLFNNSTTransportModel(tglf_based_transport_model.TGLFBasedTransportModel):
    """The spherical-tokamak networks, on TORAX's TGLF inputs."""

    networks: tuple = dataclasses.field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "networks", tuple(Network.load(n) for n in MODELS))
        super().__post_init__()

    def features(self, ti):
        f = {name: getattr(ti, name) for name in ("AS_2", "AS_3", "BETAE", "DEBYE", "DELTA_LOC", "DRMAJDX_LOC", "KAPPA_LOC", "P_PRIME_LOC", "Q_LOC", "Q_PRIME_LOC", "RLNS_1", "RLNS_2", "RLNS_3", "RLTS_1", "RLTS_2", "RLTS_3", "RMAJ_LOC", "RMIN_LOC", "S_DELTA_LOC", "S_KAPPA_LOC", "TAUS_2", "TAUS_3", "VEXB_SHEAR", "XNUE", "ZEFF")}
        f["Q_LOC"] = jnp.abs(f["Q_LOC"])
        for name in ZERO:
            f[name] = jnp.zeros_like(f["RMIN_LOC"])
        return f

    def fluxes(self, ti):
        """Electron particle flux and the two heat fluxes, in TGLF's gyro-Bohm units, on the faces."""
        f = self.features(ti)
        outs = [net(f) for net in self.networks]
        names = self.networks[0].ynames
        r = f["RMIN_LOC"]
        pick = lambda y: jnp.where(r < BLEND[0], y[0], jnp.where(r < BLEND[1], y[1], y[2]))
        get = lambda key: pick([o[names.index(key)] for o in outs])
        return get("OUT_G_elec"), get("OUT_Q_elec"), get("OUT_Q_ions")

    def call_implementation(self, transport, runtime_params, geo, core_profiles, pedestal_model_output):
        del pedestal_model_output
        ti = self._prepare_tglf_inputs(transport=transport, geo=geo, core_profiles=core_profiles, poloidal_velocity_multiplier=runtime_params.neoclassical.poloidal_velocity_multiplier)
        g_e, q_e, q_i = self.fluxes(ti)
        return self._make_core_transport(ion_heat_flux_GB=q_i, electron_heat_flux_GB=q_e, electron_particle_flux_GB=g_e, tglf_inputs=ti, transport=transport, geo=geo, core_profiles=core_profiles)


class TGLFNNST(transport_pydantic.TGLFNNukaeaTransportModel):
    """The configuration: as TORAX's TGLF-NN, with the spherical-tokamak networks in its place."""
    model_name: Annotated[Literal["tglfnn-st"], torax_pydantic.JAX_STATIC] = "tglfnn-st"

    def build_transport_model(self):
        return TGLFNNSTTransportModel()

    def build_runtime_params(self, t):
        return super().build_runtime_params(t)


transport.register_transport_model(TGLFNNST)


def _register_table():
    import ntm

    class TGLFNNSTTable(TGLFNNST):
        """With the turbulence table and the islands' conduction: anchored at the start, then the networks say how transport changes."""
        model_name: Annotated[Literal["tglfnn-st_table"], torax_pydantic.JAX_STATIC] = "tglfnn-st_table"

        def build_transport_model(self):
            return ntm.IslandTransport(inner=TGLFNNSTTransportModel())

    transport.register_transport_model(TGLFNNSTTable)

    class CombinedTable(transport_pydantic.CombinedTransportModel):
        """TORAX's combined model (each part on its own range of rho) with the turbulence table and the islands' conduction on the sum."""
        model_name: Annotated[Literal["combined_table"], torax_pydantic.JAX_STATIC] = "combined_table"

        def build_transport_model(self):
            return ntm.IslandTransport(inner=super().build_transport_model())

    transport.register_transport_model(CombinedTable)


_register_table()
