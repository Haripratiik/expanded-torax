"""Island model for TORAX: neoclassical tearing modes (NTMs) evolved alongside TORAX's profiles (H34, docs/ideas.md).

TORAX has no island model. This module runs TORAX's own compiled step from its own loop (TORAX is not modified) and,
after every step, for each mode m/n:
  1. finds the rational surface q = m/n (outermost crossing) on TORAX's q profile;
  2. reads TORAX's quantities there (bootstrap current and pressure just outside the island, where TORAX's profiles are
     not flattened): bootstrap current density, conductivity (physical: TORAX's resistivity_multiplier
     only enters its current-diffusion coefficients), magnetic shear, pressure and its gradient, electron temperature
     and heat diffusivity, and the current-drive profile of the ECCD source (total driven current, peak location, 1/e
     width);
  3. advances the island width w with the modified Rutherford equation, sub-stepped over TORAX's time step;
  4. flattens the temperatures across the island. Inside an island heat runs along field lines, which short-circuits the radial gradient
     (Fitzpatrick 1995): the gradient falls by w_d^2 / (w^2 + w_d^2), the same factor as in the bootstrap term below. So with the transport
     model 'qlknn_island' the heat diffusivity across the island is raised by chi_0 (w / w_d)^2, chi_0 being the local value with no island,
     and TORAX's own solver flattens the profile. (The first version flattened T_e and T_i after each step, keeping the energy; the
     solver then rebuilt the gradient within the next 0.5 s step, since heat crosses a few cm in about 10 ms, so fusion power did not respond
     properly: it rose with width up to 16 cm. flatten_on=True still does that, for the results made with it.)
Islands start at w0 or at seed events (time, width), the way a sawtooth crash or an ELM seeds one.

The equation (Kong et al. 2022, PPCF 64 044008, eq. 3, SI units, all at q = m/n; rbar = a sqrt(Phi / Phi_b), which is
TORAX's rho_norm times a_minor):
  (tau_R / rbar) dw/dt = rbar Delta'_0 + rbar Delta'_BS + rbar Delta'_GGJ + rbar Delta'_CD
  tau_R = mu0 rbar^2 / (1.22 eta_neo)   (the 1.22 is Rutherford's factor, applied once)
  rbar Delta'_0  = -m by default (Sauter et al. 1997); a constant
  rbar Delta'_BS = a_bs rbar (2 mu0 R0 q / (s B0)) j_bs w / (w^2 + w_d^2)   (Kong 2022 eqs. 4 to 7, large aspect ratio)
  rbar Delta'_GGJ = -6 a_ggj rbar D_R / sqrt(w^2 + 0.2 w_d^2),  D_R = -(q^2 - 1) L_q^2 beta / (rbar L_p),
                    beta = 2 mu0 p / B0^2, L_q = rbar / s, L_p = p / (dp / drbar)   (Kong 2022 eqs. 9, 10)
  rbar Delta'_CD = -a_cd (16 mu0 R0 q / (pi s B0)) (I_cd / w_dep^2) N_cd G_cd   (Kong 2022 eq. 11, continuous drive)
                   N_cd = 0.25 / (1 + (2/3) (w / w_dep)^2),
                   G_cd = 1.6 [1 - tanh((0.75 x - 0.3) / 0.2)] / [1 - tanh(-1.5) + 2 x^3] - 0.6 exp(-x^2),
                   x = |rbar_dep - rbar| / max(w, w_dep)   (Kong et al. 2019, NF 59 076035, eqs. 4 to 8)
  w_d = (5.1 rbar (eps s n)^(-1/2))^(4/3) (chi_perp n / (v_Te R0 L_q))^(1/3)   (Sauter 2002, PPCF 44 1999, convective
        form; chi_perp is TORAX's electron heat diffusivity, v_Te = sqrt(2 T_e / m_e)), unless given
Coefficients default to Kong 2022's ASDEX Upgrade fit (a_bs 1.5, a_ggj 1, a_cd 0.65; theory 3.2, 1, 1); polarization
and heating terms are off, as in that fit. Currents are signed along the plasma current, so co-current drive and
bootstrap current count positive. The pressure is the thermal pressure of electrons and main ions.
With Mode(rotation=Rotation()), the island also rotates and can lock (Nies, Reiman and Fisch 2022, Nucl. Fusion 62 086044, eq. 4, as in the
the companion chip project's sim/iter_ntm; integrated at 0.1 ms between TORAX's steps on the width of that step):
  dom/dt = [om0 - om (1 + C_M w/a)] / tau_M0 - (w/a)^3 / tau_A0^2 [ (C_1/m) om tau_w / ((om tau_w)^2 + 1) + (m^2/256) (a/L_q)^2 (w_vac/w)^2 sin(phi - phi_EF) ]
A growing island's wall drag overcomes the rotation and it locks, below 0.4 Hz. A lock is a risk, not a certainty: while locked, the discharge ends as a disruption at quench_rate per second (lock_ends=True;
None: at once). On MAST 38 to 45% of clean onsets that lock quench within 20 ms (the companion chip project report, section 10), about 1 s at the 50 times slower pace of a resistive scaling to ITER, so about 0.5 per second.
Near a lock (below 5 Hz) the error field also enters the width: 2 m (w_vac/w)^2 cos(phi - phi_EF).
With elms=ELMs(), run() adds Type I ELMs as a crash and recovery of the pedestal top (the transport model of TORAX has none; its pedestal is an input). Each crash takes drop_T and drop_n of the
pedestal's temperature and density; the top then recovers linearly until the next, cycling about the configured height so the mean plasma is unchanged. The period is the energy of one ELM over the share
of the exhaust power that ELMs carry: (drop_T + drop_n) W_ped / (power_share P_SOL), share 0.2 to 0.4 on many machines (Herrmann 2002). This keeps the observed trend, frequency rising with power. TORAX's own
sawtooth model is switched on through its config ('mhd'). Sawtooth crashes and ELMs reach inject() as events, so either can seed an island, which is how NTMs start in machines.
Limits: the ECCD term assumes continuous drive and a Gaussian deposition (no O-point modulation yet); the axisymmetric
effect of the drive on Delta' is left to TORAX's own current diffusion; what follows a lock is not simulated: the discharge just ends.
Run inside TORAX's venv (external/torax-venv).
"""
import copy
import dataclasses
from typing import Annotated, Literal

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from torax import pedestal as pedestal_api
from torax import sources
from torax import transport
from torax._src import constants
from torax._src import math_utils
from torax._src.physics import collisions
from torax._src.physics import psi_calculations
from torax._src.physics.radiation import radiation as radiation_lib
from torax._src.sources import electron_cyclotron_source as ecs
from torax._src.sources import formulas
from torax._src.sources import gas_puff_source
from torax._src.sources.impurity_radiation_heat_sink import impurity_radiation_mavrin_fit as mavrin
from torax._src.sources import generic_current_source
from torax._src.sources import generic_ion_el_heat_source as generic_heat
from torax._src import state as state_lib
from torax._src.orchestration import run_simulation as run_simulation_lib
from torax._src.output_tools import output
from torax._src.pedestal_model import pydantic_model as pedestal_pydantic
from torax._src.torax_pydantic import interpolated_param_1d
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import pydantic_model as transport_pydantic

MU0, QE, ME = 4e-7 * np.pi, 1.602176634e-19, 9.1093837015e-31
COMMAND = {"P": 0.0, "rho": 0.5}     # ECCD power (MW) and aim (rho_norm): set by the controller, read by the source
MAX_MODES, CHI_ADD_MAX = 4, 100.0    # m^2/s
ISLANDS = np.zeros((MAX_MODES, 4))   # per mode: surface, half width (rho_norm), diffusivity added to electrons and to ions; set by run()
PEDESTAL = np.ones(2)                # pedestal-top temperature and density over the configured ones: the ELM cycle and any script's scale
PEDESTAL_TI = np.ones(1)             # a further scale on the ion temperature alone
TURB_RHO = np.linspace(0.0, 1.0, 41)  # knots of the turbulence table
PARTICLE = np.zeros((2, len(TURB_RHO)))     # row 0: 1 where particle transport is tied to the heat channels, D = eta chi_e chi_i / (chi_e + chi_i), eta from 1 on the axis to 0.3 at the edge (Garzotti 2003, as in TORAX's Bohm-gyroBohm); row 1: V / D there, 1/m. Row 0 at 0: the transport model's own D and V, times the table
TURBULENCE = np.ones((3, len(TURB_RHO)))   # multipliers of the turbulent transport against rho_norm: electron heat, ion heat, particles. Scripts and agents write here
CHI_START = np.zeros((2, len(TURB_RHO)))   # the base model's own chi_e, chi_i at the start, for the anchor that adds (H70)
ANCHOR_ADD = np.zeros(1)             # 1: the anchor adds what the base model lacked at the start, (multiplier - 1) chi_start, and keeps that fixed; 0: it multiplies, as before
HEAT = np.ones(1)                    # multiplier of the generic heating, live
BEAM_CURRENT = np.zeros(len(TURB_RHO))   # toroidal current density the beams drive against rho_norm (A/m^2), live: beams.Beams writes here
NE_CORONAL = 1e19                    # TORAX's coronal limit of n_e tau, m^-3 s
NE_TAU = np.full(len(TURB_RHO), NE_CORONAL)   # the impurities' residence parameter n_e tau against rho_norm (m^-3 s), live: coupled.py writes it each step when asked (noncoronal, H84); at the coronal limit, TORAX's own radiation
D_LAW_MIN = 0.05        # m^2/s, TORAX's default D_e_min (transport_model pydantic base): the floor the published particle law keeps
PARTICLE_LAW = np.zeros(5)           # published particle transport (H85, docs/lit/42), on when [0] is 1: D = [1] * 2 chi_e chi_i / (chi_e + chi_i) in the core and in L-mode (the AUG and JET database's D / chi_eff, Angioni 2007),
                                     # [2] in place of [1] beyond the pedestal top in H-mode ([4] = 1; micro-tearing and ETG ratios); the turbulent pinch in L-mode V = -[3] D (dq/dr) / q, at most 2 D / R inward (Garzotti 2003's
                                     # form, Angioni 2009's curvature bound), none in H-mode, where only the neoclassical pinch acts. Drawn per run by coupled.py from the published ranges; overrides PARTICLE
LMODE_EDGE = np.zeros(1)             # 1: L-mode, so no transport barrier outside the pedestal top: the transport model's own coefficients run to the edge. TORAX masks them there and clips to chi_min, which is an H-mode barrier, and in L-mode it traps the ohmic heat of a current ramp in a thin hot layer that then carries the current (H83). 0: TORAX's masking, as before
EXTRA_HEAT = np.zeros((2, len(TURB_RHO)))   # heat density added to electrons and to ions against rho_norm (W/m^3), live: named actuators and effects write here (actuators.py, effects.py)
EXTRA_PARTICLES = np.zeros(len(TURB_RHO))   # electron source density against rho_norm (per m^3 per s), live
STEP_DT = np.array([0.1])            # the step being taken, for the radiation limiter


@dataclasses.dataclass
class Rotation:
    f0: float = 420.0                    # Hz, the unperturbed rotation (the paper's ITER case; TORAX does not evolve rotation here)
    C_M: float = 12.0
    C_1: float = 1 / 80
    tau_w: float = 0.014                 # s
    tau_M0: float = 3.7                  # s
    tau_A0: float = 3e-6                 # s
    w_vac: float = 0.0125                # m
    phi_ef: float = 0.0
    f_lock: float = 0.4                  # Hz
    dt: float = 1e-4                     # s
    quench_rate: float | None = None     # per second while locked; None: a lock ends the discharge at once


@jax.jit
def _spin(om, phi, w, a, L_q, m, om0, C_M, C_1, tau_w, tau_M0, tau_A0, w_vac, phi_ef, h, n):
    def body(_, x):
        om_, phi_ = x
        otw = om_ * tau_w
        dom = (om0 - om_ * (1 + C_M * w / a)) / tau_M0 - (w / a) ** 3 / tau_A0 ** 2 * (
            C_1 / m * otw / (otw ** 2 + 1) + m ** 2 / 256 * (a / L_q) ** 2 * (w_vac / w) ** 2 * jnp.sin(phi_ - phi_ef))
        return om_ + h * dom, jnp.mod(phi_ + h * om_, 2 * jnp.pi)
    return jax.lax.fori_loop(0, n, body, (om, phi))


def spin(om, phi, w, loc, r, dt):
    """The island's rotation (rad/s) and phase after dt at width w."""
    out = _spin(float(om), float(phi), float(w), loc["a"], loc["rbar"] / loc["s"], float(loc["m"]), 2 * np.pi * r.f0, r.C_M, r.C_1, r.tau_w, r.tau_M0, r.tau_A0, r.w_vac, r.phi_ef,
                r.dt, int(round(dt / r.dt)))
    return float(out[0]), float(out[1])


def _command():
    return np.array([COMMAND["P"], COMMAND["rho"]], np.float64)


def ecrh_controlled(runtime_params, geo, source_name, core_profiles, unused_profiles, unused_conductivity):
    """TORAX's electron-cyclotron source (its Lin-Liu formula and efficiency profile) with power and aim read from
    COMMAND every step through a callback, and the current returned as <j.B>/B0 via TORAX's own conversion, the
    convention its current equation expects (SourceProfiles.total_psi_sources). TORAX 1.4.3's own ECRH source returns
    <j.B> instead: on the ITER hybrid case it drove 480 to 680 kA at 20 MW where its efficiency definition gives about
    90 kA, a factor near B0 = 5.3 T (2026-11-01; to confirm before reporting upstream)."""
    sp = runtime_params.sources[source_name]
    cmd = jax.pure_callback(_command, jax.ShapeDtypeStruct((2,), jnp.float64))
    p = formulas.gaussian_profile(center=cmd[1], width=sp.gaussian_width, total=cmd[0] * 1e6, geo=geo)
    Te, ne = core_profiles.T_e.value, core_profiles.n_e.value
    c = constants.CONSTANTS
    j_tor = (16 * jnp.pi * c.epsilon_0 ** 2 * Te * 1e3 * sp.current_drive_efficiency * p
             / (c.q_e ** 2 * collisions.calculate_log_lambda_ee(Te, ne) * ne))
    return p, psi_calculations.j_toroidal_to_j_parallel(j_tor, geo, runtime_params.numerics.min_rho_norm)


class ECRHControlled(ecs.ElectronCyclotronSourceConfig):
    model_name: Annotated[Literal["controlled"], torax_pydantic.JAX_STATIC] = "controlled"

    @property
    def model_func(self):
        return ecrh_controlled


sources.register_source_model_config(ECRHControlled, "ecrh")


def _heat():
    return HEAT.copy()


def _extra_heat():
    return EXTRA_HEAT.copy()


def heat_controlled(runtime_params, geo, source_name, core_profiles, profiles, conductivity):
    """TORAX's generic heating times HEAT[0], plus the table EXTRA_HEAT, both of which a script can change between steps."""
    ion, el = generic_heat.default_formula(runtime_params, geo, source_name, core_profiles, profiles, conductivity)
    f = jax.pure_callback(_heat, jax.ShapeDtypeStruct((1,), jnp.float64))[0]
    extra = jax.pure_callback(_extra_heat, jax.ShapeDtypeStruct(EXTRA_HEAT.shape, jnp.float64))
    at = lambda row: jnp.interp(geo.rho_norm, jnp.asarray(TURB_RHO), extra[row])
    return ion * f + at(1), el * f + at(0)


class GenericHeatControlled(generic_heat.GenericIonElHeatSourceConfig):
    model_name: Annotated[Literal["controlled"], torax_pydantic.JAX_STATIC] = "controlled"

    @property
    def model_func(self):
        return heat_controlled


sources.register_source_model_config(GenericHeatControlled, "generic_heat")


def _beam_current():
    return BEAM_CURRENT.copy()


def beam_current(runtime_params, geo, source_name, unused_state, unused_profiles, unused_conductivity):
    """The current neutral beams drive, net of the electrons' response, from beams.Beams every step: TORAX's generic current source with its Gaussian replaced by that profile."""
    j = jax.pure_callback(_beam_current, jax.ShapeDtypeStruct(BEAM_CURRENT.shape, jnp.float64))
    return (psi_calculations.j_toroidal_to_j_parallel(jnp.interp(geo.rho_norm, jnp.asarray(TURB_RHO), j), geo, runtime_params.numerics.min_rho_norm),)


class BeamCurrent(generic_current_source.GenericCurrentSourceConfig):
    model_name: Annotated[Literal["beams"], torax_pydantic.JAX_STATIC] = "beams"

    @property
    def model_func(self):
        return beam_current


sources.register_source_model_config(BeamCurrent, "generic_current")


def _step_dt():
    return STEP_DT.copy()


def _ne_tau():
    return NE_TAU.copy()


NE_TAU_FLOOR = 1e16      # m^-3 s: the lowest n_e tau that Mavrin's non-coronal fits cover (docs/lit/37); below it they are not extrapolated


def set_residence(reactor):
    """NE_TAU from the plasma now (H84): tau = L_T^2 / D, the time an impurity takes to diffuse across one electron temperature gradient length, with the plasma's own particle
    diffusivity (impurity D taken as the main ions', an assumption; a floor at TORAX's chi_min, 0.05 m^2/s). Long in the core, so coronal there; a fraction of a millisecond at an L-mode edge,
    where the rates are non-coronal (docs/lit/39, section 3). No number fitted to a machine."""
    st = reactor.state
    geo, cp = st.geometry, st.core_profiles
    rho, rf = np.asarray(geo.rho_norm), np.asarray(geo.rho_face_norm)
    Te, ne = np.asarray(cp.T_e.value), np.asarray(cp.n_e.value)
    dT = np.gradient(Te, rho) / max(float(geo.a_minor), 1e-3)            # keV per m
    L_T = np.clip(Te / np.maximum(np.abs(dT), 1e-6), 1e-3, 1.0)
    tr = getattr(st, "core_transport", None)
    D = np.interp(rho, rf, np.asarray(tr.d_face_el)) if tr is not None and getattr(tr, "d_face_el", None) is not None else np.ones_like(rho)
    ne_tau = np.clip(ne * L_T ** 2 / np.maximum(D, 0.05), NE_TAU_FLOOR, NE_CORONAL)
    NE_TAU[:] = np.interp(TURB_RHO, rho, ne_tau)
    return ne_tau


def radiation_limited(runtime_params, geo, source_name, core_profiles, profiles, conductivity):
    """TORAX's Mavrin line radiation, which cannot take more than half the electrons' energy in one step. With much cold impurity the time scale at the edge falls to microseconds and an unlimited sink sends T_e through zero; the real cooling
    curve falls away at a few eV, where Mavrin's fit, flat below 0.1 keV, does not. Only the fastest collapse is touched.
    The cooling rate takes the residence parameter n_e tau from NE_TAU (H84): at its coronal limit everywhere, as before, this is TORAX's own sink; below it, TORAX's non-coronal Mavrin rates
    (radiation.calculate_cooling_rate with ne_tau), which differ from the coronal ones below about 0.15 keV for carbon."""
    names = runtime_params.plasma_composition.impurity_names
    fractions = jnp.stack([core_profiles.impurity_fractions[s] for s in names])
    nt = jax.pure_callback(_ne_tau, jax.ShapeDtypeStruct(NE_TAU.shape, jnp.float64))
    ne_tau = jnp.interp(geo.rho_norm, jnp.asarray(TURB_RHO), nt)
    LZ = jnp.zeros_like(core_profiles.T_e.value)
    for i, s in enumerate(names):
        LZ += fractions[i] * radiation_lib.calculate_cooling_rate(core_profiles.T_e.value, s, ne_tau=ne_tau)
    rad = -(LZ * core_profiles.n_e.value * core_profiles.n_impurity.value * core_profiles.impurity_density_scaling * runtime_params.sources[source_name].radiation_multiplier)       # as mavrin.impurity_radiation_mavrin_fit, with ne_tau
    dt = jax.pure_callback(_step_dt, jax.ShapeDtypeStruct((1,), jnp.float64))[0]
    return (-jnp.minimum(-rad, 0.75 * core_profiles.n_e.value * core_profiles.T_e.value * 1e3 * QE / dt),)


class RadiationLimited(mavrin.ImpurityRadiationHeatSinkMavrinFitConfig):
    model_name: Annotated[Literal["mavrin_limited"], torax_pydantic.JAX_STATIC] = "mavrin_limited"

    @property
    def model_func(self):
        return radiation_limited


sources.register_source_model_config(RadiationLimited, "impurity_radiation")


def _extra_particles():
    return EXTRA_PARTICLES.copy()


def puff_controlled(runtime_params, geo, source_name, core_profiles, profiles, conductivity):
    """TORAX's gas puff plus the table EXTRA_PARTICLES, which a script can change between steps."""
    base = gas_puff_source.calc_puff_source(runtime_params, geo, source_name, core_profiles, profiles, conductivity)
    extra = jax.pure_callback(_extra_particles, jax.ShapeDtypeStruct(EXTRA_PARTICLES.shape, jnp.float64))
    return (base[0] + jnp.interp(geo.rho_norm, jnp.asarray(TURB_RHO), extra),)


class GasPuffControlled(gas_puff_source.GasPuffSourceConfig):
    model_name: Annotated[Literal["controlled"], torax_pydantic.JAX_STATIC] = "controlled"

    @property
    def model_func(self):
        return puff_controlled


sources.register_source_model_config(GasPuffControlled, "gas_puff")


def _islands():
    return ISLANDS.copy()


def _turbulence():
    return TURBULENCE.copy()


def _anchor_add():
    return np.r_[ANCHOR_ADD, CHI_START.ravel()]


def _particle():
    return PARTICLE.copy()


def _lmode_edge():
    return LMODE_EDGE.copy()


def _particle_law():
    return PARTICLE_LAW.copy()


@dataclasses.dataclass(frozen=True, eq=False)
class IslandTransport(transport.TransportModel):
    """Any TORAX transport model times the turbulence table, with the islands' extra heat conduction added after its smoothing, which would otherwise spread a few cm over tens."""
    inner: transport.TransportModel = None

    def __call__(self, runtime_params, geo, core_profiles, pedestal_model_output):
        base = self.inner(runtime_params, geo, core_profiles, pedestal_model_output)
        # L-mode (LMODE_EDGE): outside the pedestal top the model's own coefficients, clipped as TORAX clips them, in place of the masked chi_min
        lm = jax.pure_callback(_lmode_edge, jax.ShapeDtypeStruct((1,), jnp.float64))
        out = geo.rho_face_norm > pedestal_model_output.rho_norm_ped_top - 1e-7

        def edge_open(_):            # the model evaluated again without TORAX's mask, only when the rule is on: a network model is not run twice otherwise
            tp = runtime_params.transport
            raw = self.inner._apply_clipping(tp, self.inner.zero_out_disabled_channels(tp, self.inner.call_implementation(tp, runtime_params, geo, core_profiles, pedestal_model_output)))
            return (jnp.where(out, raw.chi_face_el, base.chi_face_el), jnp.where(out, raw.chi_face_ion, base.chi_face_ion),
                    jnp.where(out, raw.d_face_el, base.d_face_el), jnp.where(out, raw.v_face_el, base.v_face_el))

        def edge_masked(_):
            return (base.chi_face_el, base.chi_face_ion, base.d_face_el, base.v_face_el)

        ce0, ci0, d0, v0 = jax.lax.cond(lm[0] > 0.5, edge_open, edge_masked, None)
        base = dataclasses.replace(base, chi_face_el=ce0, chi_face_ion=ci0, d_face_el=d0, v_face_el=v0)
        table = jax.pure_callback(_islands, jax.ShapeDtypeStruct((MAX_MODES, 4), jnp.float64))
        x = geo.rho_face_norm[None, :]
        rho, half = table[:, 0:1], table[:, 1:2]
        mask = 0.5 * (jnp.tanh((x - rho + half) / 0.005) - jnp.tanh((x - rho - half) / 0.005))
        turb = jax.pure_callback(_turbulence, jax.ShapeDtypeStruct(TURBULENCE.shape, jnp.float64))
        at = lambda row: jnp.interp(geo.rho_face_norm, jnp.asarray(TURB_RHO), turb[row])
        part = jax.pure_callback(_particle, jax.ShapeDtypeStruct(PARTICLE.shape, jnp.float64))
        ce, ci = base.chi_face_el * at(0), base.chi_face_ion * at(1)
        add = jax.pure_callback(_anchor_add, jax.ShapeDtypeStruct((1 + CHI_START.size,), jnp.float64))
        start = add[1:].reshape(CHI_START.shape)
        at_start = lambda row: jnp.interp(geo.rho_face_norm, jnp.asarray(TURB_RHO), start[row])
        ce = jnp.where(add[0] > 0.5, jnp.maximum(base.chi_face_el + (at(0) - 1.0) * at_start(0), 0.02), ce)      # a stiff base model's own response is kept at its own size; multiplying it would scale that up with the anchor
        ci = jnp.where(add[0] > 0.5, jnp.maximum(base.chi_face_ion + (at(1) - 1.0) * at_start(1), 0.02), ci)
        tied = (1.0 - 0.7 * geo.rho_face_norm) * ce * ci / jnp.maximum(ce + ci, 1e-6) * at(2)
        on = jnp.interp(geo.rho_face_norm, jnp.asarray(TURB_RHO), part[0]) > 0.5
        d_e = jnp.where(on, tied, base.d_face_el * at(2))
        v_e = jnp.where(on, tied * jnp.interp(geo.rho_face_norm, jnp.asarray(TURB_RHO), part[1]), base.v_face_el * at(2))
        law = jax.pure_callback(_particle_law, jax.ShapeDtypeStruct((5,), jnp.float64))
        d_law = jnp.where((law[4] > 0.5) & out, law[2], law[1]) * 2.0 * ce * ci / jnp.maximum(ce + ci, 1e-6) * at(2)
        d_law = jnp.maximum(d_law, D_LAW_MIN)      # TORAX's own D_e_min (H87): the law replaces D after TORAX's clipping, so 0.05 to 0.1 of a pedestal diffusivity at its floor fell far under it,
                                                   # cut an H-mode edge cell off from its neighbours and emptied it until TORAX stopped (22786, 18464 in v17 and v18)
        q = jnp.abs(core_profiles.q_face)
        dq_dr = jnp.gradient(q, jnp.maximum(geo.r_mid_face, 1e-6))
        v_law = jnp.maximum(-law[3] * d_law * dq_dr / jnp.maximum(q, 1e-3), -2.0 * d_law / geo.R_major)
        v_law = jnp.where(law[4] > 0.5, 0.0, v_law)
        d_e = jnp.where(law[0] > 0.5, d_law, d_e)
        v_e = jnp.where(law[0] > 0.5, v_law, v_e)
        return dataclasses.replace(base, chi_face_el=ce + (mask * table[:, 2:3]).sum(0), chi_face_ion=ci + (mask * table[:, 3:4]).sum(0), d_face_el=d_e, v_face_el=v_e)

    def call_implementation(self, *args):
        raise NotImplementedError


class QLKNNIsland(transport_pydantic.QLKNNTransportModel):
    model_name: Annotated[Literal["qlknn_island"], torax_pydantic.JAX_STATIC] = "qlknn_island"

    def build_transport_model(self):
        return IslandTransport(inner=super().build_transport_model())


transport.register_transport_model(QLKNNIsland)


class BohmGyroBohmTable(transport_pydantic.BohmGyroBohmTransportModel):
    """Bohm-gyroBohm with the turbulence table and islands' conduction, as above."""
    model_name: Annotated[Literal["bohm-gyrobohm_table"], torax_pydantic.JAX_STATIC] = "bohm-gyrobohm_table"

    def build_transport_model(self):
        return IslandTransport(inner=super().build_transport_model())


transport.register_transport_model(BohmGyroBohmTable)


class TGLFNNTable(transport_pydantic.TGLFNNukaeaTransportModel):
    """The UKAEA TGLF surrogate with the turbulence table and islands' conduction: anchored at the start, then TGLF says how transport changes. TORAX's own wrapper of it, with no numbers added here."""
    model_name: Annotated[Literal["tglfnn-ukaea_table"], torax_pydantic.JAX_STATIC] = "tglfnn-ukaea_table"

    def build_transport_model(self):
        return IslandTransport(inner=super().build_transport_model())


transport.register_transport_model(TGLFNNTable)


class ConstantTable(transport_pydantic.ConstantTransportModel):
    """Constant diffusivities with the turbulence table and islands' conduction: with an anchored table, frozen transport."""
    model_name: Annotated[Literal["constant_table"], torax_pydantic.JAX_STATIC] = "constant_table"

    def build_transport_model(self):
        return IslandTransport(inner=super().build_transport_model())


transport.register_transport_model(ConstantTable)


def _pedestal_scale():
    return np.r_[PEDESTAL, PEDESTAL_TI]


@dataclasses.dataclass(frozen=True, eq=False)
class ELMPedestal(pedestal_api.PedestalModel):
    """A pedestal model with its top scaled by the ELM cycle."""
    inner: pedestal_api.PedestalModel = None

    def _call_implementation(self, *args):
        out = self.inner._call_implementation(*args)
        f = jax.pure_callback(_pedestal_scale, jax.ShapeDtypeStruct((3,), jnp.float64))
        return dataclasses.replace(out, T_e_ped=out.T_e_ped * f[0], T_i_ped=out.T_i_ped * f[0] * f[2], n_e_ped=out.n_e_ped * f[1])


class SetTpedNpedELM(pedestal_pydantic.SetTpedNped):
    model_name: Annotated[Literal["set_T_ped_n_ped_elm"], torax_pydantic.JAX_STATIC] = "set_T_ped_n_ped_elm"

    def build_pedestal_model(self):
        inner = super().build_pedestal_model()
        return ELMPedestal(formation_model=inner.formation_model, saturation_model=inner.saturation_model, inner=inner)


pedestal_api.register_pedestal_model(SetTpedNpedELM)


@dataclasses.dataclass
class ELMs:
    drop_T: float = 0.10                 # share of the pedestal temperature lost at a crash
    drop_n: float = 0.05                 # and of its density
    power_share: float = 0.3             # share of the exhaust power carried by ELMs
    jitter: float = 0.2                  # relative scatter of the period
    seed: int = 0
    law: object = None                   # None: the period from the exhaust power, as above. Otherwise law(reactor, t) gives (interval in s, drop_T, drop_n) for the next ELM, or None while there are no ELMs: measured frequencies (natural.py)


@dataclasses.dataclass
class Mode:
    m: int
    n: int
    w0: float = 0.0                      # width at the start (m)
    seeds: tuple = ()                    # ((time in s, width in m), ...): the width is raised to the seed at that time
    prescribed: object = None            # width(t) in m: the island follows this and the equation is not used (replays)
    rotation: Rotation | None = None     # None: islands neither rotate nor lock


@dataclasses.dataclass
class Coefficients:
    a_bs: float = 1.5
    a_ggj: float = 1.0
    a_cd: float = 0.65
    rdelta0: float | None = None         # rbar Delta'_0; None: -m
    w_sat: float | None = None           # with it, rbar Delta' = rdelta0 (1 - w / w_sat): a tearing mode's own saturation (m)
    w_d: float | None = None             # fixed small-island width (m); None: Sauter 2002 from TORAX's profiles
    w_cap: bool = False                  # an island's width kept within the plasma: rho_s +- w / 2a inside 0 and 1 (H69). Without it a locked island could grow past the edge in a run's last milliseconds, to 183 cm on MAST
    k_pol: float | None = None           # the small-island threshold: w_small = k_pol ion banana widths at the outboard midplane (La Haye 2012: 2 to 3 measured on DIII-D and NSTX), in place of w_d in the bootstrap term; None: w_d's soft form, as before version 5


def _interp(x, xp, fp):
    return float(np.interp(x, np.asarray(xp), np.asarray(fp)))


def local(st, mode, eccd, w=0.0):
    """TORAX's quantities at the outermost q = m/n surface of state st, or None if there is no such surface. With an
    island of width w, the bootstrap current and pressure are read just outside its edges (mean of the two sides): the
    equation needs their unperturbed values, and inside the island TORAX's profiles are already flattened."""
    cp, geo = st.core_profiles, st.geometry
    rf, q = np.asarray(geo.rho_face_norm), np.asarray(cp.q_face)
    qt = mode.m / mode.n
    cross = np.where((q[:-1] - qt) * (q[1:] - qt) <= 0)[0]
    if len(cross) == 0:
        return None
    i = cross[-1]
    rho = float(rf[i] + (qt - q[i]) * (rf[i + 1] - rf[i]) / (q[i + 1] - q[i]))
    if _interp(rho, rf, np.asarray(cp.s_face)) < 0.05:       # the island equation needs shear: where q is flat or reversed it gives no island (and a NaN from the square root of s, which took TORAX down with it on MAST's reversed profiles)
        return None
    a, R0, B0 = float(geo.a_minor), float(geo.R_major), float(geo.B_0)
    rc = np.asarray(geo.rho_norm)
    j_tot = np.asarray(cp.j_total)
    sgn = np.sign(_interp(rho, rc, j_tot)) or 1.0
    T_e, T_i = np.asarray(cp.T_e.value), np.asarray(cp.T_i.value)
    n_e, n_i = np.asarray(cp.n_e.value), np.asarray(cp.n_i.value)
    p = 1e3 * QE * (n_e * T_e + n_i * T_i)
    dp = np.gradient(p, rc * a)
    j_bs = np.asarray(st.core_sources.bootstrap_current.j_parallel_bootstrap)
    side = w / (2 * a) + float(rc[1] - rc[0]) if w > 0 else 0.0
    at = lambda f: 0.5 * (_interp(rho - side, rc, f) + _interp(rho + side, rc, f))
    loc = dict(rho=rho, rbar=rho * a, a=a, R0=R0, B0=B0, q=qt, m=mode.m, n=mode.n,
               s=_interp(rho, rf, np.asarray(cp.s_face)),
               j_bs=sgn * at(j_bs),
               eta=1.0 / _interp(rho, rc, np.asarray(cp.sigma)),
               p=at(p), dpdr=at(dp), T_e=_interp(rho, rc, T_e), T_i=_interp(rho, rc, T_i), j_total=abs(_interp(rho, rc, j_tot)),
               chi_e=_interp(rho, rf, np.asarray(st.core_transport.chi_face_el)), chi_i=_interp(rho, rf, np.asarray(st.core_transport.chi_face_ion)),
               I_cd=0.0, rho_dep=rho, w_dep=1.0)
    # for the threshold term: the ion banana width at the outboard midplane, with the poloidal field there (version 5). Leaves the numbers above alone
    try:
        R_o, psi_ = np.asarray(geo.R_out_face), np.asarray(cp.psi.value)
        Ro_s, dRo = _interp(rho, rf, R_o), _interp(rho, rf, np.gradient(R_o, rf))
        B_th = abs(_interp(rho, rc, np.gradient(psi_, rc)) / dRo) / (2 * np.pi * Ro_s)
        A_i = float(np.ravel(np.asarray(cp.A_i))[0])
        loc["w_bi"] = float(np.sqrt(max(loc["rbar"] / R0, 1e-4)) * np.sqrt(2 * A_i * 1.67262192e-27 * 1e3 * QE * loc["T_i"]) / (QE * B_th))
    except Exception:
        loc["w_bi"] = 0.0
    j_cd = st.core_sources.psi.get(eccd)
    if j_cd is not None:
        j_cd = np.asarray(j_cd)
        if np.abs(j_cd).max() > 0:
            k = int(np.argmax(np.abs(j_cd)))
            above = np.where(np.abs(j_cd) >= np.abs(j_cd[k]) / np.e)[0]
            loc.update(I_cd=sgn * float(math_utils.area_integration(j_cd, geo)), rho_dep=float(rc[k]), j_cd_peak=sgn * float(j_cd[k]),
                       w_dep=max(float(rc[above[-1]] - rc[above[0]]), float(rc[1] - rc[0])) * a)
    return loc


def small_island_width(loc):
    eps, L_q = loc["rbar"] / loc["R0"], loc["rbar"] / loc["s"]
    v_te = np.sqrt(2 * loc["T_e"] * 1e3 * QE / ME)
    return (5.1 * loc["rbar"] / np.sqrt(eps * loc["s"] * loc["n"])) ** (4 / 3) \
        * (max(loc["chi_e"], 1e-3) * loc["n"] / (v_te * loc["R0"] * L_q)) ** (1 / 3)


def rates(w, loc, c):
    """dw/dt (m/s) and the four terms of rbar Delta' (dimensionless) at width w."""
    rb, q, s, R0, B0 = loc["rbar"], loc["q"], loc["s"], loc["R0"], loc["B0"]
    w_d = c.w_d if c.w_d is not None else small_island_width(loc)
    L_q, L_p = rb / s, loc["p"] / loc["dpdr"] if loc["dpdr"] != 0 else -np.inf
    beta = 2 * MU0 * loc["p"] / B0 ** 2
    d0 = c.rdelta0 if c.rdelta0 is not None else -float(loc["m"])
    if c.w_sat is not None:
        d0 = d0 * (1 - w / c.w_sat)
    bs = c.a_bs * rb * (2 * MU0 * R0 * q / (s * B0)) * loc["j_bs"] * w / (w ** 2 + w_d ** 2)
    if c.k_pol:                          # La Haye 2012 eq. 2.1 in place of the soft w_d form, not on top of it: 1 / w - w_small^2 / 3 w^3, which lumps the small-island effects, with w_small = k_pol ion banana widths
        w_small = c.k_pol * loc.get("w_bi", 0.0)
        w_c = max(w, 0.2 * w_small, 1e-4)
        bs = c.a_bs * rb * (2 * MU0 * R0 * q / (s * B0)) * loc["j_bs"] * (1 / w_c - w_small ** 2 / (3 * w_c ** 3))
    D_R = -(q ** 2 - 1) * L_q ** 2 * beta / (rb * L_p)
    ggj = -6 * c.a_ggj * rb * D_R / np.sqrt(w ** 2 + 0.2 * w_d ** 2)
    cd = 0.0
    if loc["I_cd"] != 0:
        wdep = loc["w_dep"]
        x = abs(loc["rho_dep"] - loc["rho"]) * loc["a"] / max(w, wdep)
        N = 0.25 / (1 + (2 / 3) * (w / wdep) ** 2)
        G = 1.6 * (1 - np.tanh((0.75 * x - 0.3) / 0.2)) / (1 - np.tanh(-1.5) + 2 * x ** 3) - 0.6 * np.exp(-x ** 2)
        cd = -c.a_cd * (16 * MU0 * R0 * q / (np.pi * s * B0)) * loc["I_cd"] / wdep ** 2 * N * G
    tau_R = MU0 * rb ** 2 / (1.22 * loc["eta"])
    return rb / tau_R * (d0 + bs + ggj + cd), dict(delta0=d0, bs=bs, ggj=ggj, cd=cd, w_d=w_d, tau_R=tau_R)


def advance(w, loc, c, dt, substeps):
    """w after dt (Heun's method on frozen profiles, w kept at or above 0)."""
    h = dt / substeps
    for _ in range(substeps):
        k1 = rates(w, loc, c)[0]
        k2 = rates(max(w + h * k1, 0.0), loc, c)[0]
        w = max(w + 0.5 * h * (k1 + k2), 0.0)
    return w


def flatten(cp, geo, rho, w):
    """T_e and T_i flattened over |rho_norm - rho| < w / (2 a), energy in the region kept."""
    rc, a = np.asarray(geo.rho_norm), float(geo.a_minor)
    inside = np.abs(rc - rho) < w / (2 * a)
    if inside.sum() < 2:
        return cp
    dv = np.asarray(geo.vpr) * np.asarray(geo.drho_norm)            # cell volumes; drho_norm is per cell
    new = {}
    for T, n in (("T_e", "n_e"), ("T_i", "n_i")):
        Tv, nv = np.array(getattr(cp, T).value), np.asarray(getattr(cp, n).value)
        Tv[inside] = (nv * Tv * dv)[inside].sum() / (nv * dv)[inside].sum()
        new[T] = dataclasses.replace(getattr(cp, T), value=jnp.asarray(Tv, dtype=getattr(cp, T).value.dtype))
    return dataclasses.replace(cp, **new)



def set_island(k, loc, w, c, chi0):
    """The extra heat conduction of mode k at width w, for the transport model. chi0[k] holds the local diffusivities from before the island."""
    if loc is None or w <= 0.0:
        ISLANDS[k] = 0.0
        return
    if w < 0.005 or k not in chi0:
        chi0[k] = (loc["chi_e"], loc["chi_i"])
    w_d = c.w_d if c.w_d is not None else small_island_width(loc)
    f = (w / w_d) ** 2
    ISLANDS[k] = (loc["rho"], w / (2 * loc["a"]), min(chi0[k][0] * f, CHI_ADD_MAX), min(chi0[k][1] * f, CHI_ADD_MAX))


def rollout(state, ppo, step_fn, modes, widths, c, command, n, eccd="ecrh", substeps=20, flatten_on=False, chi0=None, provider=None):
    """The widths after n more steps from (state, ppo) under a fixed command (P in MW, aim in rho_norm), with no seed events: what
    would have happened. The run itself is untouched: TORAX's states are immutable and COMMAND is restored."""
    saved, saved_islands, chi0 = dict(COMMAND), ISLANDS.copy(), dict(chi0 or {})
    COMMAND.update(P=float(command[0]), rho=float(command[1]))
    w = list(widths)
    for _ in range(n):
        t_prev = float(state.t)
        state, ppo = step_fn(state, ppo, max_dt=np.inf, runtime_params_overrides=provider or step_fn.runtime_params_provider)
        for k, md in enumerate(modes):
            loc = local(state, md, eccd, w[k])
            if loc is None:
                w[k] = 0.0
                continue
            w[k] = advance(w[k], loc, c, float(state.t) - t_prev, substeps)
            set_island(k, loc, w[k], c, chi0)
            if flatten_on and w[k] > 0:
                state = dataclasses.replace(state, core_profiles=flatten(state.core_profiles, state.geometry, loc["rho"], w[k]))
    COMMAND.update(saved)
    ISLANDS[:] = saved_islands
    return w


class Reactor:
    """TORAX with islands, one step at a time: the harness under the digital twin (H39). step() advances one TORAX step and returns its record. Between steps anything may be edited: the plasma's
    profiles, its islands, transport, heating, the pedestal. snapshot() and restore() branch a run: TORAX's states are immutable, so a branch costs nothing to keep. The keyword arguments are run()'s.
    One reactor per process: COMMAND, ISLANDS, PEDESTAL, TURBULENCE and HEAT are read by TORAX's compiled step through callbacks."""

    def __init__(self, torax_config, modes, coeffs=None, eccd="ecrh", substeps=20, flatten_on=None, controller=None, inject=None, keep_history=True, lock_ends=True, elms=None):
        COMMAND.update(P=0.0, rho=0.5)
        ISLANDS[:] = 0.0
        PEDESTAL[:] = 1.0
        PEDESTAL_TI[:] = 1.0
        TURBULENCE[:] = 1.0
        PARTICLE[:] = 0.0
        HEAT[:] = 1.0
        EXTRA_HEAT[:] = 0.0
        BEAM_CURRENT[:] = 0.0
        EXTRA_PARTICLES[:] = 0.0
        LMODE_EDGE[:] = 0.0
        PARTICLE_LAW[:] = 0.0
        NE_TAU[:] = NE_CORONAL
        self.torax_config, self.modes, self.eccd, self.substeps, self.controller, self.inject = torax_config, modes, eccd, substeps, controller, inject
        self.keep_history, self.lock_ends, self.elms = keep_history, lock_ends, elms
        self.chi0 = {}
        self.elm_rng = np.random.default_rng(elms.seed) if elms else None
        self.elm_last, self.elm_period = float("-inf"), None
        self.elm_drop, self.elm_times = (0.0, 0.0), []
        self.quench_rng = np.random.default_rng(1 if elms is None else elms.seed + 1)
        self.c = coeffs or Coefficients()
        # the old way, only for a transport model that knows nothing of islands
        self.flatten_on = torax_config.transport.model_name != "qlknn_island" if flatten_on is None else flatten_on
        self.state, self.ppo, self.step_fn = run_simulation_lib.prepare_simulation(torax_config)
        self.provider, self.max_dt = self.step_fn.runtime_params_provider, np.inf    # TORAX's parameters, which set_param can change, and a limit on the step (for fast events)
        self.fixed_dt = self.get_param("numerics.fixed_dt")
        self.history, self.ppo_history = [self.state], [self.ppo]
        self.widths = [float(md.w0) for md in modes]
        self.locs = [None for _ in modes]
        self.spins = [(2 * np.pi * md.rotation.f0, 0.0) if md.rotation else None for md in modes]
        self.disrupted = False
        self.done_seeds = [set() for _ in modes]
        self.ped_scale = np.ones(2)
        self.ped_extra = np.ones(2)                    # a second scale on the pedestal, for what the plant does to it: fuelling, pump-out, the current
        self.carry = {}                                # values from before the step, for the step's record: what plant models computed
        self.retries, self.n_retried = 0, 0
        self.geo_overrides = None                      # a geometry provider that replaces the configured one: a free-boundary solver's equilibrium (freeboundary.py)
        self.before_step, self.after_solve = [], []    # hooks: f(reactor) before TORAX's step; f(reactor, t) right after it
        self.records = [dict(t=float(self.state.t), P_fusion=float(self.ppo.P_fusion), Q_fusion=float(self.ppo.Q_fusion), W_thermal=float(self.ppo.W_thermal_total), H98=float(self.ppo.H98),
                             **{f"w_{md.m}_{md.n}": w for md, w in zip(modes, self.widths)})]   # TORAX's outputs start at t0
        self.sim_error = state_lib.SimError.NO_ERROR
        self.finished = False

    @property
    def t(self):
        return float(self.state.t)

    def step(self):
        """One TORAX step and everything that follows it. Returns its record, or None once the discharge has ended."""
        if self.finished or self.step_fn.is_done(self.state.t):
            self.finished = True
            return None
        for hook in self.before_step:
            hook(self)
        modes, c = self.modes, self.c
        t_prev = float(self.state.t)
        last_good = self.state, self.ppo
        STEP_DT[0] = min(self.max_dt, self.fixed_dt)
        attempt_dt = self.max_dt
        for attempt in range(self.retries + 1):        # a step that fails can be tried again, smaller
            if self.geo_overrides is not None and not jnp.issubdtype(jnp.asarray(last_good[0].geometry.Phi_b_dot).dtype, jnp.floating):      # a state from a fixed geometry holds this as an integer, and TORAX's sawtooth branch needs one type on both sides
                last_good = (dataclasses.replace(last_good[0], geometry=dataclasses.replace(last_good[0].geometry, Phi_b_dot=jnp.asarray(last_good[0].geometry.Phi_b_dot, dtype=jnp.float64))), last_good[1])
            self.state, self.ppo = self.step_fn(*last_good, max_dt=attempt_dt, runtime_params_overrides=self.provider, geo_overrides=self.geo_overrides)
            self.sim_error = self.state.check_for_errors()
            if self.sim_error == state_lib.SimError.NO_ERROR and self.state.core_profiles.below_minimum_temperature(self.provider.numerics.T_minimum_eV):
                self.sim_error = state_lib.SimError.LOW_TEMPERATURE_COLLAPSE
            if self.sim_error == state_lib.SimError.NO_ERROR:
                self.n_retried += attempt
                break
            attempt_dt = min(attempt_dt, self.fixed_dt) / 4
            STEP_DT[0] = attempt_dt
        if self.sim_error != state_lib.SimError.NO_ERROR:
            self.state, self.ppo = last_good           # the last valid plasma stays readable
            self.finished = True
            return None
        state, ppo = self.state, self.ppo
        t, dt = float(state.t), float(state.t) - t_prev
        rec = dict(t=t, P_fusion=float(ppo.P_fusion), Q_fusion=float(ppo.Q_fusion), W_thermal=float(ppo.W_thermal_total), H98=float(ppo.H98), P_SOL=float(ppo.P_SOL_total),
                   P_rad=-float(ppo.P_radiation_e), P_brems=-float(ppo.P_bremsstrahlung_e), P_LH=float(ppo.P_LH), beta_N=float(ppo.beta_N), f_GW=float(ppo.fgw_n_e_line_avg), q95=float(ppo.q95), li3=float(ppo.li3), n_line=float(ppo.n_e_line_avg),
                   T_e0=float(state.core_profiles.T_e.value[0]), T_i0=float(state.core_profiles.T_i.value[0]), n_e0=float(state.core_profiles.n_e.value[0]), f_bs=float(ppo.f_bootstrap),
                   P_alpha=float(ppo.P_alpha_total), P_ohmic=float(ppo.P_ohmic_e), Z_eff_avg=float(np.mean(np.asarray(state.core_profiles.Z_eff))), **self.carry)
        crash = bool(state.solver_numeric_outputs.sawtooth_crash)
        elm = False
        elms = self.elms
        PEDESTAL[:] = self.ped_scale * self.ped_extra
        if elms and elms.law is not None:
            elm = self._elm_law(rec, t)
        elif elms:
            if self.elm_period is None or t - self.elm_last >= self.elm_period:
                cp, geo = state.core_profiles, state.geometry
                rc = np.asarray(geo.rho_norm)
                top = int(np.argmin(np.abs(rc - 0.9)))
                W_ped = 3e3 * QE * float(cp.n_e.value[top]) * float(cp.T_e.value[top]) * float(geo.volume_face[-1])
                dW = (elms.drop_T + elms.drop_n) * W_ped
                self.elm_period = max(dW / max(elms.power_share * float(ppo.P_SOL_total), 1e6) * (1 + elms.jitter * self.elm_rng.normal()), 1e-3)
                self.elm_last, elm = t, True
                rec.update(elm=1.0, dW_elm_MJ=dW / 1e6, elm_period=self.elm_period)
            phase = (t - self.elm_last) / self.elm_period
            PEDESTAL[:] = self.ped_scale * self.ped_extra * (1 + elms.drop_T * (phase - 0.5), 1 + elms.drop_n * (phase - 0.5))
        if crash:
            rec["sawtooth"] = 1.0
        for hook in self.after_solve:
            hook(self, t)
        widths = self.widths
        if self.inject is not None:
            for k, ws in self.inject(t, list(widths), dict(sawtooth=crash, elm=elm)):
                widths[k] = max(widths[k], ws)
        for k, md in enumerate(modes):
            for j, (ts, ws) in enumerate(md.seeds):
                if j not in self.done_seeds[k] and t >= ts:
                    widths[k] = max(widths[k], ws)
                    self.done_seeds[k].add(j)
            loc = self.locs[k] = local(state, md, self.eccd, widths[k])
            tag = f"{md.m}_{md.n}"
            if loc is None:
                widths[k] = 0.0
                ISLANDS[k] = 0.0
                rec.update({f"w_{tag}": 0.0, f"rho_{tag}": np.nan})
                continue
            if md.prescribed is not None:
                widths[k] = float(md.prescribed(t))
            else:
                widths[k] = advance(widths[k], loc, c, dt, self.substeps)
            if md.rotation is not None:
                r = md.rotation
                if widths[k] < 0.003:            # gone: the next island starts from the plasma's rotation
                    widths[k], self.spins[k] = 0.0, (2 * np.pi * r.f0, self.spins[k][1])
                else:
                    self.spins[k] = spin(*self.spins[k], widths[k], loc, r, dt)
                    if abs(self.spins[k][0]) < 2 * np.pi * 5.0 and widths[k] > 0:
                        widths[k] = max(widths[k] + dt * loc["rbar"] / (MU0 * loc["rbar"] ** 2 / (1.22 * loc["eta"])) * 2 * md.m * (r.w_vac / widths[k]) ** 2 * np.cos(self.spins[k][1] - r.phi_ef), 0.0)
                f = self.spins[k][0] / (2 * np.pi)
                rec[f"f_{tag}"] = f
                if widths[k] > 0 and abs(f) < r.f_lock:
                    rec[f"locked_{tag}"] = 1.0
                    if self.lock_ends and (r.quench_rate is None or self.quench_rng.random() < 1 - np.exp(-r.quench_rate * dt)):
                        self.disrupted = True
                        rec.update(disrupted=1.0, disrupted_mode=float(k))
            if c.w_cap:
                widths[k] = min(widths[k], 2.0 * loc["a"] * min(loc["rho"], 1.0 - loc["rho"]))
            set_island(k, loc, widths[k], c, self.chi0)
            dwdt, terms = rates(widths[k], loc, c)
            rec.update({f"w_{tag}": widths[k], f"rho_{tag}": loc["rho"], f"dwdt_{tag}": dwdt, f"j_bs_{tag}": loc["j_bs"],
                        f"I_cd_{tag}": loc["I_cd"], f"j_cd_{tag}": loc.get("j_cd_peak", 0.0), f"rho_dep_{tag}": loc["rho_dep"], f"w_dep_{tag}": loc["w_dep"],
                        **{f"{name}_{tag}": v for name, v in terms.items()}})
            if self.flatten_on and widths[k] > 0:
                self.state = state = dataclasses.replace(state, core_profiles=flatten(state.core_profiles, state.geometry, loc["rho"], widths[k]))
        if self.controller is not None:
            tags = [f"{md.m}_{md.n}" for md in modes]
            obs = dict(t=t, w=list(widths), rho=[rec.get(f"rho_{g}", np.nan) for g in tags],
                       dwdt=[rec.get(f"dwdt_{g}", np.nan) for g in tags], P=COMMAND["P"], aim=COMMAND["rho"],
                       j_bs=[rec.get(f"j_bs_{g}", 0.0) / 1e6 for g in tags], loc=list(self.locs), f=[rec.get(f"f_{g}", np.nan) for g in tags],
                       rollout=lambda P_, aim_, n: rollout(self.state, self.ppo, self.step_fn, modes, widths, c, (P_, aim_), n, self.eccd, self.substeps, self.flatten_on, self.chi0, self.provider))
            P, aim = self.controller(obs)
            COMMAND.update(P=float(P), rho=float(aim))
            rec.update(P_cmd=float(P), aim_cmd=float(aim), **getattr(self.controller, "extra", {}))
        self.records.append(rec)
        if self.keep_history:
            self.history.append(self.state)
            self.ppo_history.append(self.ppo)
        if self.disrupted:
            self.finished = True
        return rec

    # edits: TORAX's own parameters, between steps
    def set_param(self, path, value):
        """Change a parameter of TORAX from now on: a dotted path into its runtime parameters ("sources.gas_puff.S_total", "profile_conditions.Ip", "pedestal.T_e_ped"), with a number. A time trace becomes that number
        at all its knots. The compiled step does not need rebuilding, as long as the array shape stays."""
        node = self.provider.get_node_from_path(path)
        if hasattr(node, "time") and hasattr(node, "value"):           # a time-varying scalar
            update = interpolated_param_1d.TimeVaryingScalarUpdate(value=jnp.full_like(jnp.asarray(node.value), float(value)))
        else:
            update = jnp.asarray(value, dtype=jnp.asarray(node).dtype)
        self.provider = self.provider.update_provider_from_mapping({path: update})

    def get_param(self, path):
        node = self.provider.get_node_from_path(path)
        return float(np.ravel(np.asarray(node.value if hasattr(node, "value") else node))[-1])

    def set_ratio(self, species, rho, ratio):
        """The impurity density against electron density for one species, a profile on knots rho: TORAX's composition, dilution, Z_eff and radiation follow."""
        update = interpolated_param_2d.TimeVaryingArrayUpdate(value=jnp.asarray(ratio)[None, :], rho_norm=jnp.asarray(rho), time=jnp.zeros(1))
        self.provider = self.provider.update_provider_from_mapping({f"plasma_composition.impurity.species.{species}": update})

    # edits: the plasma itself, between steps
    def zone(self, rho, grid=None, smooth=0.01):
        """1 inside rho = (a, b), falling to 0 over smooth, on the cell grid."""
        x = np.asarray(self.state.geometry.rho_norm) if grid is None else grid
        return 0.5 * (np.tanh((x - rho[0]) / smooth) - np.tanh((x - rho[1]) / smooth))

    def edit_profile(self, name, factor, rho=(0.0, 1.0), smooth=0.01):
        """Multiply T_e, T_i or n_e by factor inside the zone; the solver takes it from there."""
        cp = self.state.core_profiles
        var = getattr(cp, name)
        value = np.asarray(var.value) * (1 + (factor - 1) * self.zone(rho, smooth=smooth))
        self.state = dataclasses.replace(self.state, core_profiles=dataclasses.replace(cp, **{name: dataclasses.replace(var, value=jnp.asarray(value, dtype=var.value.dtype))}))

    def flatten_zone(self, rho, density=False):
        """Flatten T_e and T_i inside the zone, keeping its energy: a forced sawtooth-like crash, whatever the q profile says. With density, n_e and n_i too, keeping their counts."""
        cp, geo = self.state.core_profiles, self.state.geometry
        rc = np.asarray(geo.rho_norm)
        inside = (rc >= rho[0]) & (rc <= rho[1])
        dv = np.asarray(geo.vpr) * np.asarray(geo.drho_norm)
        new = {}
        if density:
            E = {T: (np.asarray(getattr(cp, n).value) * np.asarray(getattr(cp, T).value) * dv)[inside].sum() for T, n in (("T_e", "n_e"), ("T_i", "n_i"))}
            for n in ("n_e", "n_i"):
                nv = np.array(getattr(cp, n).value)
                nv[inside] = (nv * dv)[inside].sum() / dv[inside].sum()
                new[n] = dataclasses.replace(getattr(cp, n), value=jnp.asarray(nv, dtype=getattr(cp, n).value.dtype))
            for T, n in (("T_e", "n_e"), ("T_i", "n_i")):
                Tv, nv = np.array(getattr(cp, T).value), np.asarray(new[n].value)
                Tv[inside] = E[T] / (nv * dv)[inside].sum()
                new[T] = dataclasses.replace(getattr(cp, T), value=jnp.asarray(Tv, dtype=getattr(cp, T).value.dtype))
            self.state = dataclasses.replace(self.state, core_profiles=dataclasses.replace(cp, **new))
            return
        for T, n in (("T_e", "n_e"), ("T_i", "n_i")):
            Tv, nv = np.array(getattr(cp, T).value), np.asarray(getattr(cp, n).value)
            Tv[inside] = (nv * Tv * dv)[inside].sum() / (nv * dv)[inside].sum()
            new[T] = dataclasses.replace(getattr(cp, T), value=jnp.asarray(Tv, dtype=getattr(cp, T).value.dtype))
        self.state = dataclasses.replace(self.state, core_profiles=dataclasses.replace(cp, **new))

    def _elm_law(self, rec, t):
        """ELMs as a renewal process with the intervals and sizes from a law: several can fall inside one step, and each time is kept. Returns whether one happened."""
        count, law = 0, self.elms.law
        if self.elm_period is None:                    # not in an ELMing phase: the first ELM comes one interval after the law allows them
            drawn = law(self, t)
            if drawn is None:
                return False
            self.elm_period, self.elm_drop, self.elm_last = drawn[0], drawn[1:], t
        while t - self.elm_last >= self.elm_period:
            self.elm_last += self.elm_period
            self.elm_times.append(self.elm_last)
            count += 1
            drawn = law(self, self.elm_last)
            if drawn is None:
                self.elm_period = None
                break
            self.elm_period, self.elm_drop = drawn[0], drawn[1:]
        if self.elm_period is not None:
            phase = (t - self.elm_last) / self.elm_period
            PEDESTAL[:] = self.ped_scale * self.ped_extra * (1 + self.elm_drop[0] * (phase - 0.5), 1 + self.elm_drop[1] * (phase - 0.5))
        if count:
            rec.update(elm=float(count))
        return count > 0

    def snapshot(self):
        """Everything needed to come back here. Decision models are deep-copied, so a branch does not disturb them."""
        return dict(state=self.state, ppo=self.ppo, widths=list(self.widths), locs=list(self.locs), spins=list(self.spins), done_seeds=[set(x) for x in self.done_seeds], chi0=dict(self.chi0),
                    elm=(self.elm_last, self.elm_period, self.elm_rng.bit_generator.state if self.elm_rng else None), quench=self.quench_rng.bit_generator.state, command=dict(COMMAND),
                    islands=ISLANDS.copy(), pedestal=PEDESTAL.copy(), ped_scale=self.ped_scale.copy(), ped_extra=self.ped_extra.copy(), turbulence=TURBULENCE.copy(), heat=HEAT.copy(), extra_heat=EXTRA_HEAT.copy(), beam_current=BEAM_CURRENT.copy(), lmode_edge=LMODE_EDGE.copy(), particle_law=PARTICLE_LAW.copy(), ne_tau=NE_TAU.copy(), extra_particles=EXTRA_PARTICLES.copy(), c=dataclasses.replace(self.c), disrupted=self.disrupted,
                    provider=self.provider, max_dt=self.max_dt, sim_error=self.sim_error,
                    finished=self.finished, n_records=len(self.records), n_history=len(self.history), controller=copy.deepcopy(self.controller),
                    # what else a step reads that a step or a command changes (package 1.1): the readings carried to the next record (the bolometer's between its
                    # updates), the free boundary's geometry, the drawn ELM size and the ELM list, the step and its settings, the hooks, and every module array
                    carry=copy.deepcopy(self.carry), geo_overrides=getattr(self, "geo_overrides", None), elm_drop=copy.copy(self.elm_drop), n_elm_times=len(self.elm_times),
                    elms_fields=dataclasses.replace(self.elms) if self.elms is not None else None, retries=(self.retries, self.n_retried),
                    step=(self.step_fn, self.fixed_dt, self.flatten_on), hooks=(list(self.before_step), list(self.after_solve)),
                    arrays={k: v.copy() for k, v in globals().items() if k.isupper() and isinstance(v, np.ndarray)})

    def restore(self, snap):
        if "arrays" in snap:                           # snapshots from package 1.1 on
            for k, v in snap["arrays"].items():
                if globals()[k].shape == v.shape:
                    globals()[k][...] = v
                else:
                    globals()[k] = v.copy()
            self.carry.clear()
            self.carry.update(copy.deepcopy(snap["carry"]))
            self.geo_overrides = snap["geo_overrides"]
            self.elm_drop = copy.copy(snap["elm_drop"])
            del self.elm_times[snap["n_elm_times"]:]
            if self.elms is not None and snap["elms_fields"] is not None:
                for f in dataclasses.fields(self.elms):
                    setattr(self.elms, f.name, getattr(snap["elms_fields"], f.name))
            self.retries, self.n_retried = snap["retries"]
            self.step_fn, self.fixed_dt, self.flatten_on = snap["step"]
            self.before_step[:], self.after_solve[:] = snap["hooks"]
        self.state, self.ppo, self.widths, self.locs, self.spins = snap["state"], snap["ppo"], list(snap["widths"]), list(snap["locs"]), list(snap["spins"])
        self.done_seeds, self.chi0 = [set(x) for x in snap["done_seeds"]], dict(snap["chi0"])
        self.elm_last, self.elm_period = snap["elm"][:2]
        if self.elm_rng:
            self.elm_rng.bit_generator.state = snap["elm"][2]
        self.quench_rng.bit_generator.state = snap["quench"]
        COMMAND.update(snap["command"])
        ISLANDS[:], PEDESTAL[:], TURBULENCE[:], HEAT[:] = snap["islands"], snap["pedestal"], snap["turbulence"], snap["heat"]
        EXTRA_HEAT[:], EXTRA_PARTICLES[:] = snap["extra_heat"], snap["extra_particles"]
        BEAM_CURRENT[:] = snap.get("beam_current", 0.0)
        LMODE_EDGE[:] = snap.get("lmode_edge", 0.0)
        PARTICLE_LAW[:] = snap.get("particle_law", 0.0)
        NE_TAU[:] = snap.get("ne_tau", NE_CORONAL)
        self.provider, self.max_dt, self.sim_error = snap["provider"], snap["max_dt"], snap["sim_error"]
        self.ped_scale, self.c, self.disrupted, self.finished = snap["ped_scale"].copy(), dataclasses.replace(snap["c"]), snap["disrupted"], snap["finished"]
        self.ped_extra = snap["ped_extra"].copy()
        del self.records[snap["n_records"]:]
        del self.history[snap["n_history"]:]
        del self.ppo_history[snap["n_history"]:]
        self.controller = copy.deepcopy(snap["controller"])

    def finish(self):
        """(TORAX's output DataTree with an 'islands' node, or None without history; the per-step records)."""
        if not self.keep_history:
            return None, self.records
        hist = output.StateHistory(state_history=self.history, post_processed_outputs_history=tuple(self.ppo_history),
                                   sim_error=self.sim_error, torax_config=self.torax_config)
        tree = hist.simulation_output_to_xr()
        keys = sorted({k for r in self.records for k in r if k != "t"})
        ds = xr.Dataset({k: ("time", np.array([r.get(k, np.nan) for r in self.records], float)) for k in keys},
                        coords={"time": np.array([r["t"] for r in self.records])})
        tree["islands"] = xr.DataTree(ds)
        return tree, self.records


def run(torax_config, modes, coeffs=None, eccd="ecrh", substeps=20, flatten_on=None, max_steps=100000, controller=None, inject=None, keep_history=True, lock_ends=True, elms=None):
    """TORAX with islands, to the end. Returns (TORAX's output DataTree with an 'islands' node, list of per-step records). With keep_history=False the
    states are not kept and the tree is None, for long runs: the records then hold what is needed (fusion power, Q, stored energy each step).
    inject(t, widths, events) -> [(mode index, width in m), ...], if given, is called after every step: islands raised on demand, for events
    that depend on the run (Mode.seeds are the ones known in advance); events says whether a sawtooth crash or an ELM just happened.
    controller(obs) -> (P in MW, aim in rho_norm) is called after every step with obs = dict(t, w, rho, dwdt, P, aim, j_bs, loc, rollout)
    (per-mode lists for w, rho, dwdt, j_bs in MA/m^2 and loc, the dict from local(); rollout(P, aim, n) runs the plant forward from
    here under another command, see rollout()); its command acts from the next step, through the 'controlled' ECRH source."""
    r = Reactor(torax_config, modes, coeffs, eccd, substeps, flatten_on, controller, inject, keep_history, lock_ends, elms)
    for _ in range(max_steps):
        if r.step() is None:
            break
    return r.finish()
