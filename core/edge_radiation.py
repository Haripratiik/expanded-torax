"""Power radiated in the scrape-off layer (SOL) and divertor, from TORAX's own extended Lengyel model.

What it computes. TORAX ships the extended Lengyel model of Body, Kallenbach and Eich, "A simple, accurate model for detachment access",
Nucl. Fusion 65 (2025) 086002, https://doi.org/10.1088/1741-4326/ade4d9 (arXiv https://arxiv.org/abs/2504.05486), in
external/torax/torax/_src/edge. It follows one flux tube, one heat-flux width (lambda_q) wide, from the outer midplane to the outer target:
Spitzer conduction with carbon radiation (Mavrin 2017 non-coronal rates) down to the convection-conduction interface, then a convective
layer near the target whose power and momentum losses are fits in the target temperature (Table 1 of the paper), and the sheath. lambda_q
itself comes out of the model (Eich 2020 turbulence-broadened width, Body eq. 49). edge_radiation() runs it in forward mode: given the
carbon concentration, it returns the target temperature, and from the solved flux tube the share of the tube's power radiated by carbon.

The model has no "radiated power" output. It is computed here from the solved state with the paper's own relations (eqs. 38 and 39):
  main SOL (outer midplane to the divertor entrance):   q_div^2 = q_u^2 - k L_INT(T_div..T_u),  k = 2 kappa_e n_sep^2 T_u^2
  divertor leg (entrance to the convection layer):      q_cc^2 = (q_div / b)^2 - k L_INT(T_cc..T_div)
The tube's area grows by b (the divertor broadening) at the entrance, so the power radiated is A (q_u - q_div) above and A (q_div - b q_cc)
below; carbon radiation is the fraction 1 - b q_cc / q_u of the tube's power (Body eqs. 46 and 47 write the same as
1 - f_pow = (1 - f_rad) / b). The convective layer then removes the fraction b q_cc f_pow,c(T_t) / q_u. That loss is NOT counted as
radiation: in the 1D model the fits came from, its sinks are hydrogen ionization at an effective 15 eV per event, charge exchange, and
1% nitrogen radiation (Body eq. 13 and section 5), so it is mostly energy a bolometer does not see as light. It is returned apart
(P_convective_loss) as an upper bound on what else that layer could add.

Inputs, by where they come from.
  Physics, from the plasma state: P_SOL (power crossing the separatrix), the line-averaged density (or a separatrix density), R0, a,
    kappa, delta (separatrix values), the vacuum toroidal field at R0, Ip, q95, and the carbon concentration (or Z_eff, read as carbon).
  Published rules (each can be overridden):
    n_sep = 0.3 n_line: the ITER CDA design value, 30% of the average density, shown against the multi-machine edge database in
      G.D. Porter et al., J. Nucl. Mater. 266-269 (1999) 917 (separatrix density versus line average, L- and H-mode, C-Mod, AUG, DIII-D,
      JET, JT-60U). The fraction rises with density there.
    L_par = pi q95 R0, the textbook outer-midplane-to-target length (P.C. Stangeby, The Plasma Boundary of Magnetic Fusion Devices, 2000).
      For MAST the published modelling value is 10 m at q = 7.1 (F. Militello et al., J. Nucl. Mater. 438 (2013) S530,
      https://arxiv.org/abs/1307.2391), about half of pi q95 R0: an option, not the default.
    L_div = L_par / 4: the X-point-to-target share Body et al. use for ASDEX Upgrade (5 m of 20 m). No MAST value was found.
    kappa95 = kappa / 1.12, delta95 = delta / 1.5: ITER Physics Design Guidelines 1989 (N.A. Uckan, IAEA/ITER/DS/10), the rule PROCESS
      uses (https://ukaea.github.io/PROCESS/physics-models/plasma_geometry); PROCESS's own MAST fit gives nearly the same numbers.
    Share of P_SOL in the modelled divertor = 0.45: MAST double nulls put 1 part in 10 on the inner targets (J.-W. Ahn et al.,
      J. Nucl. Mater. 290-293 (2001) 820, as quoted by S.F. Paul et al., J. Nucl. Mater. 337-339 (2005) 251), and the outer 0.9 is split
      between the upper and lower outer divertors. TORAX's default, 2/3, is for a single null.
    B_pol at the outer midplane over its separatrix average: TORAX's default 4/3 (Body eq. 51) is for conventional aspect ratio. MAST's
      EFIT gives 0.7 to 1.1, so pass the equilibrium's value when there is one (TORAX's integrated model does the same with its geometry).
    Divertor broadening b = 3, n_e tau = 0.5e17 m^-3 s (non-coronal residence parameter), sheath transmission 8: Body et al. and TORAX's
      defaults, set for ASDEX Upgrade.
  Assumptions (no published value applied):
    The carbon concentration in the SOL equals the core's (the twin runs 1.67% of n_e, Z_eff 1.5). No enrichment: TORAX's enrichment
      regression (Kallenbach 2024) is fitted to recycling seed gases and gives about 50 for carbon at 1 Pa, which is not physical for a
      wall-sourced impurity. Published MAST range: charge exchange gives about 0.1% carbon in the core, bremsstrahlung Z_eff up to 2
      (3.3% if all carbon) outside mid-radius (S.S. Henderson et al., https://arxiv.org/abs/1508.00424).
    The radiated fraction found for the near-SOL flux tube holds for all of P_SOL (the far SOL and the inner divertors included).
    T_i = T_e, no flows at the separatrix, toroidal flux expansion 1, and every other model setting at TORAX's default.

Not covered: radiation inside the separatrix (that is the core model's job), hydrogen line radiation in the divertor, and anything a
bolometer sees from charge-exchange neutrals.

Usage:
  from edge_radiation import edge_radiation
  out = edge_radiation(P_SOL=1.3e6, n_line=2.4e19, R0=0.82, a=0.58, kappa=1.81, delta=0.36, B0=0.50, Ip=0.68e6, q95=6.3, c_carbon=0.0167)
  out.P_rad, out.T_e_target
Runs in TORAX's venv (external/torax-venv). The check against MAST's bolometers is edge_radiation_check.py.
"""
import dataclasses
import math

import torax  # noqa: F401  (switches JAX to float64, as every TORAX run uses)
import jax
import jax.numpy as jnp
import numpy as np
from torax._src.edge import collisional_radiative_models as crm
from torax._src.edge import divertor_sol_1d as dsol
from torax._src.edge import extended_lengyel_defaults as eld
from torax._src.edge import extended_lengyel_enums as ele
from torax._src.edge import extended_lengyel_formulas as elf
from torax._src.edge import extended_lengyel_standalone as els

MU0 = 4e-7 * math.pi
A_D, A_C, Z_C = 2.014, 12.011, 6.0

# Published rules, defaults of edge_radiation(); the docstring above gives each source.
N_SEP_FRACTION = 0.3
L_DIV_FRACTION = 0.25
DIVERTOR_SHARE_DOUBLE_NULL = 0.45
KAPPA95_DIVISOR, DELTA95_DIVISOR = 1.12, 1.5


@dataclasses.dataclass
class EdgeRadiation:
    """Result for one plasma state. Powers in W, temperatures in eV, lengths in m unless named otherwise."""
    P_rad: float                 # carbon radiation in the SOL and divertor down to the convection layer: the estimate
    P_rad_main_sol: float        # the part between the outer midplane and the divertor entrance (X-point)
    P_rad_divertor: float        # the part in the divertor leg, entrance to the convection layer
    P_convective_loss: float     # convective layer near the target: ionization, charge exchange, impurity radiation; not counted in P_rad
    P_target: float              # what reaches the targets
    T_e_target: float
    T_e_separatrix: float
    T_e_divertor_entrance: float
    q_parallel: float            # upstream parallel heat flux, W/m^2
    lambda_q_mm: float           # heat-flux width at the outer midplane from the model (times lambda_q_factor)
    alpha_t: float
    n_sep: float                 # m^-3
    L_par: float
    L_div: float
    f_rad: float                 # P_rad / P_SOL
    physics_outcome: int         # 0 success; 2: so much power lost that the solution is fully detached
    numerics_error: int          # Newton-Raphson: 0 converged, 1 not, 2 coarse
    multiple_roots: bool
    roots_T_e_target: tuple      # every distinct target temperature the multistart found
    q_cc_mismatch: float         # |q_cc from the Lengyel integral / q_cc from the target side - 1|: 0 when the solution is consistent
    inputs: dict


@jax.jit
def _accounting(params, state, c_carbon, ne_tau):
    """Heat fluxes along the solved tube (Body eqs. 38 and 39) and the temperatures where its regions meet."""
    m = dsol.DivertorSOL1D(params=params, state=state)
    T_u, T_div, T_cc = m.T_e_separatrix, m.divertor_entrance_electron_temp, m.electron_temp_at_cc_interface
    conc = {"C": c_carbon}
    L_div_u = crm.calculate_weighted_L_INT(conc, start_temp=T_div / 1e3, stop_temp=T_u / 1e3, ne_tau=ne_tau)
    L_cc_div = crm.calculate_weighted_L_INT(conc, start_temp=T_cc / 1e3, stop_temp=T_div / 1e3, ne_tau=ne_tau)
    k = 2.0 * state.kappa_e * params.separatrix_electron_density ** 2 * T_u ** 2
    q_u, b = state.q_parallel, params.divertor_broadening_factor
    q_div = jnp.sqrt(jnp.maximum(q_u ** 2 - k * L_div_u, 0.0))
    q_cc = jnp.sqrt(jnp.maximum((q_div / b) ** 2 - k * L_cc_div, 0.0))
    return dict(T_u=T_u, T_div=T_div, T_cc=T_cc, q_div=q_div, q_cc=q_cc, q_cc_target_side=m.parallel_heat_flux_at_cc_interface,
                f_pow_conv=elf.calc_power_loss_in_convection_layer(state.T_e_target))


def _lambda_q_omp(T_sep_eV, alpha_t, average_ion_mass, B_pol_avg, R0, a, bpol_ratio):
    """The model's own heat-flux width at the outer midplane, m (Body eqs. 49, 50, 53)."""
    rho_s_pol = math.sqrt(T_sep_eV * average_ion_mass * 1.66053906660e-27 / 1.602176634e-19) / B_pol_avg
    lam_avg = 0.6 * (1.0 + 2.1 * alpha_t ** 1.7) * rho_s_pol
    return lam_avg / (bpol_ratio * (R0 + a) / R0)


def edge_radiation(P_SOL, R0, a, kappa, delta, B0, Ip, q95, n_line=None, n_sep=None, c_carbon=None, Z_eff=None, *,
                   bpol_omp_over_avg=eld.RATIO_BPOL_OMP_TO_BPOL_AVG, n_sep_fraction=N_SEP_FRACTION, divertor_share=DIVERTOR_SHARE_DOUBLE_NULL,
                   L_par=None, L_div_fraction=L_DIV_FRACTION, broadening=eld.DIVERTOR_BROADENING_FACTOR, ne_tau=eld.NE_TAU,
                   kappa95=None, delta95=None, lambda_q_factor=1.0, lambda_q_mm=None, whole_sol=True, solver="hybrid"):
    """SOL and divertor radiated power for one plasma state; see the module docstring for what each input is and where it comes from.

    P_SOL [W], R0 and a [m] (geometric centre and minor radius, so R0 + a is the outer midplane), kappa and delta at the separatrix,
    B0 [T] the vacuum toroidal field at R0, Ip [A], q95; n_line [m^-3] (then n_sep = n_sep_fraction n_line) or n_sep; c_carbon = n_C/n_e
    or Z_eff (carbon fully stripped: c = (Z_eff - 1) / 30).
    lambda_q_factor multiplies the model's heat-flux width; lambda_q_mm imposes a width at the outer midplane instead (found by a few
    passes, as the model's own width moves with the solution). Both act through the share of P_SOL entering the tube, which in TORAX's
    model enters only q_parallel = P_SOL share (1 - 1/e) pitch / (2 pi (R0 + a) lambda_q), so the substitution is exact.
    whole_sol: the tube's radiated fraction applied to all of P_SOL (default) or only to the power inside the modelled tube.
    """
    if c_carbon is None:
        if Z_eff is None:
            raise ValueError("give c_carbon or Z_eff")
        c_carbon = (Z_eff - 1.0) / (Z_C * (Z_C - 1.0))
    if n_sep is None:
        if n_line is None:
            raise ValueError("give n_line or n_sep")
        n_sep = n_sep_fraction * n_line
    L_par = math.pi * q95 * R0 if L_par is None else L_par
    L_div = L_div_fraction * L_par
    kappa95 = kappa / KAPPA95_DIVISOR if kappa95 is None else kappa95
    delta95 = delta / DELTA95_DIVISOR if delta95 is None else delta95
    n_D = 1.0 - Z_C * c_carbon                     # quasi-neutrality, per electron
    average_ion_mass = A_D * n_D + A_C * c_carbon  # sum(A_i n_i) / n_e, as TORAX's integrated model defines it
    mean_ion_charge_state = 1.0 / (n_D + c_carbon)  # n_e / sum(n_i)
    inputs = dict(P_SOL=P_SOL, R0=R0, a=a, kappa=kappa, delta=delta, B0=B0, Ip=Ip, q95=q95, n_line=n_line, n_sep=n_sep, c_carbon=c_carbon,
                  bpol_omp_over_avg=bpol_omp_over_avg, divertor_share=divertor_share, L_par=L_par, L_div=L_div, broadening=broadening, ne_tau=ne_tau,
                  kappa95=kappa95, delta95=delta95, lambda_q_factor=lambda_q_factor, lambda_q_mm=lambda_q_mm, whole_sol=whole_sol)
    if P_SOL <= 0.0:
        return EdgeRadiation(0.0, 0.0, 0.0, 0.0, 0.0, float("nan"), float("nan"), float("nan"), 0.0, float("nan"), float("nan"), n_sep, L_par, L_div,
                             0.0, -1, -1, False, (), float("nan"), inputs)
    common = dict(power_crossing_separatrix=float(P_SOL), separatrix_electron_density=float(n_sep), fixed_impurity_concentrations={"C": float(c_carbon)},
                  main_ion_charge=1.0, magnetic_field_on_axis=float(B0), plasma_current=float(Ip), connection_length_target=float(L_par),
                  connection_length_divertor=float(L_div), major_radius=float(R0), minor_radius=float(a), elongation_psi95=float(kappa95),
                  triangularity_psi95=float(delta95), average_ion_mass=float(average_ion_mass), mean_ion_charge_state=float(mean_ion_charge_state),
                  ratio_bpol_omp_to_bpol_avg=float(bpol_omp_over_avg), divertor_broadening_factor=float(broadening), ne_tau=float(ne_tau))
    mode = {"hybrid": ele.SolverMode.HYBRID, "fixed_point": ele.SolverMode.FIXED_POINT, "newton_raphson": ele.SolverMode.NEWTON_RAPHSON}[solver]
    shaping = float(elf.calc_shaping_factor(kappa95, delta95))
    B_pol_avg = MU0 * Ip / (2.0 * math.pi * a * shaping)

    factor = float(lambda_q_factor)
    passes = 1 if lambda_q_mm is None else 6
    for _ in range(passes):
        # a wider tube carries the same power at a lower parallel flux: dividing the share by the factor is the same as multiplying lambda_q
        out = els.run_extended_lengyel_standalone(**common, fraction_of_P_SOL_to_divertor=float(divertor_share) / factor,
                                                  computation_mode=ele.ComputationMode.FORWARD, solver_mode=mode)
        lam = factor * _lambda_q_omp(float(out.T_e_separatrix) * 1e3, float(out.alpha_t), average_ion_mass, B_pol_avg, R0, a, bpol_omp_over_avg)
        if lambda_q_mm is None:
            break
        ratio = lambda_q_mm * 1e-3 / lam
        if abs(ratio - 1.0) < 0.005:
            break
        factor *= ratio

    params = els._construct_parameters(**common, seed_impurity_weights={}, fraction_of_P_SOL_to_divertor=float(divertor_share) / factor,
                                       sheath_heat_transmission_factor=eld.SHEATH_HEAT_TRANSMISSION_FACTOR, angle_of_incidence_target=eld.ANGLE_OF_INCIDENCE_TARGET,
                                       SOL_conduction_fraction=eld.SOL_CONDUCTION_FRACTION, ratio_of_molecular_to_ion_mass=eld.RATIO_MOLECULAR_TO_ION_MASS,
                                       T_wall=eld.T_WALL, mach_separatrix=eld.MACH_SEPARATRIX, T_i_T_e_ratio_separatrix=eld.T_I_T_E_RATIO_SEPARATRIX,
                                       n_e_n_i_ratio_separatrix=eld.N_E_N_I_RATIO_SEPARATRIX, T_i_T_e_ratio_target=eld.T_I_T_E_RATIO_TARGET,
                                       n_e_n_i_ratio_target=eld.N_E_N_I_RATIO_TARGET, mach_target=eld.MACH_TARGET, toroidal_flux_expansion=eld.TOROIDAL_FLUX_EXPANSION)
    state = dsol.ExtendedLengyelState(q_parallel=out.q_parallel, alpha_t=out.alpha_t, kappa_e=out.kappa_e, T_e_target=out.T_e_target,
                                      c_z_prefactor=out.c_z_prefactor)
    acc = {k: float(v) for k, v in _accounting(params, state, jnp.asarray(float(c_carbon)), jnp.asarray(float(ne_tau))).items()}
    q_u = float(out.q_parallel)
    b = float(broadening)
    f_main = 1.0 - acc["q_div"] / q_u
    f_divleg = (acc["q_div"] - b * acc["q_cc"]) / q_u
    f_conv = b * acc["q_cc"] * acc["f_pow_conv"] / q_u
    f_target = b * acc["q_cc"] * (1.0 - acc["f_pow_conv"]) / q_u
    P_ref = P_SOL if whole_sol else P_SOL * divertor_share * (1.0 - math.exp(-1.0))
    inputs["lambda_q_factor_used"] = factor
    numerics = out.solver_status.numerics_outcome
    err = int(np.asarray(numerics.error)) if hasattr(numerics, "error") else 0
    unique = out.get_unique_roots()
    roots = tuple(float(x) for x in np.ravel(np.asarray(unique.T_e_target)) if np.isfinite(x)) if unique is not None else ()
    return EdgeRadiation(
        P_rad=(f_main + f_divleg) * P_ref, P_rad_main_sol=f_main * P_ref, P_rad_divertor=f_divleg * P_ref, P_convective_loss=f_conv * P_ref,
        P_target=f_target * P_ref, T_e_target=float(out.T_e_target), T_e_separatrix=acc["T_u"], T_e_divertor_entrance=acc["T_div"], q_parallel=q_u,
        lambda_q_mm=lam * 1e3, alpha_t=float(out.alpha_t), n_sep=float(n_sep), L_par=L_par, L_div=L_div, f_rad=(f_main + f_divleg) * P_ref / P_SOL,
        physics_outcome=int(np.asarray(out.solver_status.physics_outcome)), numerics_error=err, multiple_roots=bool(np.asarray(out.multiple_roots_found)),
        roots_T_e_target=roots, q_cc_mismatch=abs(acc["q_cc"] / max(acc["q_cc_target_side"], 1e-30) - 1.0), inputs=inputs)


if __name__ == "__main__":
    r = edge_radiation(P_SOL=1.28e6, n_line=2.35e19, R0=0.819, a=0.581, kappa=1.806, delta=0.365, B0=0.497, Ip=0.676e6, q95=6.31, c_carbon=0.0167,
                       bpol_omp_over_avg=0.83)
    print({k: v for k, v in dataclasses.asdict(r).items() if k != "inputs"})
