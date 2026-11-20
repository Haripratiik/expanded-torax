"""The scrape-off layer (SOL) for the twin (H86): its separatrix and target state from what the core puts out, by textbook physics and multi-machine numbers.

Power, the two-point model (P.C. Stangeby, The Plasma Boundary of Magnetic Fusion Devices, IOP 2000, chapter 5), one outer leg's flux tube, T_e = T_i:
  q_par = f_div P_SOL / (2 pi R_omp lambda_q B_pol,omp / B_omp)
  T_u^(7/2) = T_t^(7/2) + (7/2) q_par L / kappa_0e,   n_u T_u = 2 n_t T_t,   q_par = gamma e n_t T_t c_s(T_t)
  lambda_q = 0.63 B_pol,omp^-1.19 mm: Eich et al., Nucl. Fusion 53 (2013) 093031, regression 14 (multi-machine H-mode, MAST among the machines).
  lambda_T = (7/2) lambda_q, the conduction-limited relation; L = pi q95 R0, the textbook outer-midplane-to-target length (MAST's published modelling
  value, 10 m at q95 7.1, Militello et al. 2013, is about half: an option). f_div = 0.45: MAST double nulls put 1 part in 10 on the inner targets
  (Ahn et al. 2001, quoted by Paul et al. 2005), the outer 0.9 shared by two outer legs. gamma = 7 (sheath, T_e = T_i), kappa_0e = 2000.
Particles, the simple SOL (Stangeby 2000, section 1.8): particles enter the SOL across the separatrix and from neutrals ionized in it, and leave along
  the field at the sound speed over the connection length:
  n_u lambda_n / tau_par = Gamma_out / A_sep + S_SOL,   lambda_n = sqrt(D_perp tau_par),   tau_par = L / c_s(T_u)
  so n_u = (Gamma_out / A_sep + S_SOL) sqrt(tau_par / D_perp). Gamma_out is the core's outflux (TORAX's integrated particle source less the change
  of its particle content, smoothed over SMOOTH_S), S_SOL the ionization outside the separatrix per unit area (KN1D), D_perp = 1 m^2/s the textbook
  working value (Stangeby 2000; an option). The recycling at the targets closes on itself in the divertor and is not a source upstream here.
Returns the separatrix density and temperature, the decay lengths for KN1D's slab, and the target state (the recycling flux sets the hydrogen
radiation of the divertor, radiation.py). Assumptions, none set on MAST: one outer leg stands for all, no momentum or power loss along it (no
radiation in the two-point balance), no flows upstream, the SOL source spread uniformly over the separatrix.
"""
import math

import numpy as np

QE, MD, MU0 = 1.602176634e-19, 2.0141 * 1.66053906660e-27, 4e-7 * np.pi
KAPPA0, GAMMA = 2000.0, 7.0
SMOOTH_S = 0.005


def lambda_q_eich14(B_pol_omp):
    """Heat-flux width at the outer midplane in m, Eich 2013 regression 14."""
    return 0.63e-3 * max(B_pol_omp, 0.02) ** -1.19


def c_s(T_eV):
    return math.sqrt(2.0 * QE * max(T_eV, 0.1) / MD)


def two_point(q_par, L, n_u):
    """Upstream and target temperature (eV) and target density for a parallel heat flux q_par (W/m^2), length L (m) and upstream density n_u (m^-3)."""
    T_u = (3.5 * q_par * L / KAPPA0) ** (2.0 / 7.0)
    T_t = T_u
    for _ in range(200):                                   # damped fixed point on T_t; converges in a few tens of passes from both regimes
        T_u = (min(T_t, T_u) ** 3.5 + 3.5 * q_par * L / KAPPA0) ** (2.0 / 7.0)
        T_new = (2.0 * q_par / (GAMMA * QE * n_u * T_u * math.sqrt(2.0 * QE / MD))) ** 2
        T_new = min(max(T_new, 0.1), T_u)
        if abs(T_new - T_t) < 1e-6 * T_u:
            T_t = T_new
            break
        T_t = 0.5 * T_t + 0.5 * T_new
    n_t = n_u * T_u / (2.0 * T_t)
    return T_u, T_t, n_t


class SOL:
    def __init__(self, D_perp=1.0, f_div=0.45, L_par=None, relax=0.5, n_floor=2e17):
        self.D_perp, self.f_div, self.L_par, self.relax, self.n_floor = D_perp, f_div, L_par, relax, n_floor
        self.N_prev = self.t_prev = None
        self.G_out = None
        self.S_sol = 0.0                                   # ionization outside the separatrix per unit area (m^-2 s^-1), from the last KN1D call
        self.n_u = None
        self.last = {}

    def geometry(self, r):
        geo, cp = r.state.geometry, r.state.core_profiles
        R0, a, B0 = float(geo.R_major), float(geo.a_minor), abs(float(geo.B_0))
        kappa = float(np.asarray(geo.elongation_face)[-1]) if hasattr(geo, "elongation_face") else 1.8
        R_o = np.asarray(geo.R_out_face) if hasattr(geo, "R_out_face") else None
        psi = np.asarray(cp.psi.face_value()) if hasattr(cp.psi, "face_value") else None
        if R_o is not None and psi is not None and len(R_o) == len(psi):
            R_omp = float(R_o[-1])
            B_pol = abs((psi[-1] - psi[-2]) / max(R_o[-1] - R_o[-2], 1e-6)) / (2 * np.pi * R_omp)     # psi in Wb, total: B_p = |grad psi| / (2 pi R)
        else:
            R_omp = R0 + a
            B_pol = MU0 * float(r.ppo.I_total if hasattr(r.ppo, "I_total") else 1e6) / (2 * np.pi * a * math.sqrt(0.5 * (1 + kappa ** 2)))
        B_t = B0 * R0 / R_omp
        A_sep = 4 * np.pi ** 2 * R0 * a * math.sqrt(0.5 * (1 + kappa ** 2))
        return dict(R0=R0, a=a, kappa=kappa, R_omp=R_omp, B_pol=B_pol, B_omp=math.hypot(B_t, B_pol), A_sep=A_sep)

    def outflux(self, r):
        """The core's particle outflux (per s): TORAX's integrated source less the change in its particle content, smoothed."""
        geo, cp = r.state.geometry, r.state.core_profiles
        dv = np.asarray(geo.vpr) * float(np.asarray(geo.drho_norm).ravel()[0])
        N = float(np.sum(np.asarray(cp.n_e.value) * dv))
        S = float(getattr(r.ppo, "S_total", 0.0))
        t = float(r.t)
        if self.N_prev is None or t <= self.t_prev:
            G = max(S, 0.0)
        else:
            G = S - (N - self.N_prev) / (t - self.t_prev)
            a = 1.0 - math.exp(-(t - self.t_prev) / SMOOTH_S)
            G = (self.G_out if self.G_out is not None else G) + a * (G - (self.G_out if self.G_out is not None else G))
        self.N_prev, self.t_prev, self.G_out = N, t, G
        return max(G, 0.0), N, S

    def __call__(self, r, q95=None):
        g = self.geometry(r)
        P_SOL = max(float(getattr(r.ppo, "P_SOL_total", 0.0)), 1e4)
        q95 = float(getattr(r.ppo, "q95", 6.0)) if q95 is None else q95
        L = self.L_par if self.L_par is not None else math.pi * q95 * g["R0"]
        lam_q = lambda_q_eich14(g["B_pol"])
        q_par = self.f_div * P_SOL / (2 * np.pi * g["R_omp"] * lam_q * g["B_pol"] / g["B_omp"])
        G_out, N, S_core = self.outflux(r)
        n_u = self.n_u if self.n_u is not None else 1e19
        for _ in range(30):                                # n_u and T_u together: tau_par depends on T_u, T_u on n_u
            T_u, T_t, n_t = two_point(q_par, L, n_u)
            tau = L / c_s(T_u)
            n_new = max((G_out / g["A_sep"] + self.S_sol) * math.sqrt(tau / self.D_perp), self.n_floor)
            if abs(n_new - n_u) < 1e-4 * n_u:
                n_u = n_new
                break
            n_u = math.sqrt(n_u * n_new)
        T_u, T_t, n_t = two_point(q_par, L, n_u)
        tau = L / c_s(T_u)
        self.n_u = n_u if self.n_u is None else self.n_u + self.relax * (n_u - self.n_u)
        lam_n = math.sqrt(self.D_perp * tau)
        A_par = 2 * np.pi * g["R_omp"] * lam_q * g["B_pol"] / g["B_omp"]       # one outer leg's flux tube, cross-section along the field
        G_t = 2.0 * n_t * c_s(T_t) * A_par / self.f_div * 0.5                   # both outer legs carry 0.9 of P_SOL; the target flux of all four legs, by the same share
        self.last = dict(sol_n_sep=self.n_u, sol_T_sep=T_u, sol_T_t=T_t, sol_n_t=n_t, sol_lam_n=lam_n, sol_lam_q=lam_q, sol_lam_T=3.5 * lam_q, sol_q_par=q_par,
                         sol_L=L, sol_G_out=G_out, sol_S_sol=self.S_sol, sol_G_target=G_t, sol_B_pol=g["B_pol"], sol_tau_par=tau)
        return self.last
