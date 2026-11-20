"""Porcelli's sawtooth trigger, thermal terms, as a critical shear for TORAX's simple trigger (H84).

From Porcelli, Boucher and Rosenbluth 1996 (PPCF 38 2163), eqs. 1 to 16 and Appendices A and B, as read from the paper (docs/lit/40, section 1.2). Not the NTCC module, whose terms
forbid modifying it. Each step the twin computes, at the q = 1 surface:
  dW-hat = dW_Bussac + dW_el + dW_KO   (Appendix B; each term goes as 1/s1)
    dW_Bussac = -(9 pi / s1) (li1 - 1/2) eps1^2 (beta_p1^2 - beta_pc^2),  beta_p1 = (2 mu0 / B_p1^2) (<p>_1 - p(r1)),  beta_pc = 0.3 (1 - 5 r1 / (3 a))
    dW_el     = -(18 pi / s1) (li1 - 1/2)^3 ((kappa1 - 1) / 2)^2
    dW_KO     = 0.6 c_p eps1^(1/2) beta_i0 / s1,  c_p = (5/2) int_0^1 x^(3/2) p_i(x) / p_i0 dx
  and the crash conditions
    (14) -dW-hat > 0.5 omega_*i tau_A                                 a crash now
    (15) -c_rho rho_hat < -dW-hat < 0.5 omega_*i tau_A  and  omega_*i < c_* gamma_rho,  gamma_rho = C_rho rho_hat^(4/7) S^(-1/7) s1^(6/7) / tau_A
  with c_rho = 1, c_* = 3, C_rho = 1.1 (the paper's and NTCC's). Since dW-hat goes as 1/s1, (15) is s1 > s_crit = max(s1 dW-hat / (c_rho rho_hat), [omega_*i tau_A / (c_* C_rho rho_hat^(4/7) S^(-1/7))]^(7/6)),
  the form of Chapman's review, and that is what TORAX's simple trigger is given each step; (14) sets it to zero.
Left out, on purpose: the fast-ion terms (eq. 13 and dW_fast), until the twin's fast-ion content agrees with what EFIT allows (H83: it is 2 to 4 times too high, and at that level
the fast-ion fluid drive flips the result, docs/lit/40, 1.6); rotation, which MAST's periods depend on and Porcelli lacks. The large aspect ratio expansions are used at eps1 about 0.3,
outside their stated range: untested at MAST's aspect ratio anywhere we found. No coefficient is set on MAST.
"""
import numpy as np

MU0, QE, MP = 4e-7 * np.pi, 1.602176634e-19, 1.67262192e-27
C_RHO, C_STAR, C_RHO_GAMMA = 1.0, 3.0, 1.1


def critical_shear(reactor):
    """The critical shear at q = 1 for TORAX's simple trigger from Porcelli's thermal criteria, and the terms, or None where there is no q = 1 surface."""
    st = reactor.state
    geo, cp = st.geometry, st.core_profiles
    q = np.asarray(cp.q_face)
    rf = np.asarray(geo.rho_face_norm)
    below = np.flatnonzero(q <= 1.0)
    if len(below) == 0 or below[-1] == 0:
        return None
    j = int(below[-1])
    rho_b = float(np.asarray(geo.rho_b)) if np.ndim(np.asarray(geo.rho_b)) == 0 else float(np.asarray(geo.rho_b).ravel()[0])
    R = float(geo.R_major)
    B = abs(float(geo.B_0))
    a = float(geo.a_minor)
    r1 = rf[j] * rho_b                                   # the toroidal-flux radius at q = 1, close to the average radius r1-bar (docs/lit/40, 1.5)
    abar = rho_b
    eps1 = r1 / R
    kappa1 = float(np.asarray(geo.elongation_face)[j])
    s1 = max(float(np.asarray(cp.s_face)[j]), 1e-3)
    # pressures on the cell grid
    rho = np.asarray(geo.rho_norm)
    Te, Ti = np.asarray(cp.T_e.value) * 1e3 * QE, np.asarray(cp.T_i.value) * 1e3 * QE        # J
    ne, ni = np.asarray(cp.n_e.value), np.asarray(cp.n_i.value)
    p = ne * Te + ni * Ti
    pi_ = ni * Ti
    inside = rho <= rf[j]
    if inside.sum() < 2:
        return None
    vpr = np.asarray(geo.vpr)
    p_avg = float(np.sum(p[inside] * vpr[inside]) / np.sum(vpr[inside]))
    p_r1 = float(np.interp(rf[j], rho, p))
    B_p1 = eps1 * B / max(q[j], 1e-3)
    beta_p1 = 2 * MU0 / B_p1 ** 2 * (p_avg - p_r1)
    beta_pc = 0.3 * (1 - 5 * r1 / (3 * abar))
    x = rf[: j + 1] / rf[j]
    li1 = 2 * np.trapezoid(x ** 3 / np.maximum(q[: j + 1], 1e-3) ** 2, x)        # cylinder, inside q = 1
    xi = rho[inside] / rf[j]
    c_p = 2.5 * np.trapezoid(xi ** 1.5 * pi_[inside] / max(pi_[0], 1e-30), xi) if len(xi) > 1 else 1.0
    beta_i0 = 2 * MU0 * pi_[0] / B ** 2
    D_bussac = -9 * np.pi * (li1 - 0.5) * eps1 ** 2 * (beta_p1 ** 2 - beta_pc ** 2)         # s1 dW-hat, each term
    D_el = -18 * np.pi * (li1 - 0.5) ** 3 * ((kappa1 - 1) / 2) ** 2
    D_ko = 0.6 * c_p * np.sqrt(eps1) * beta_i0
    D = D_bussac + D_el + D_ko
    # frequencies and rates at q = 1, the paper's definitions (docs/lit/40, 1.2); NTCC simplifies to central T_i0 and n_i0
    m_i = float(np.ravel(np.asarray(cp.A_i))[0]) * MP
    Ti1 = float(np.interp(rf[j], rho, Ti))
    ni1 = float(np.interp(rf[j], rho, ni))
    v_A = B / np.sqrt(MU0 * max(ni1 * m_i, 1e-12))
    tau_A = np.sqrt(3) * R / v_A
    dpi = np.gradient(pi_, rho) / rho_b                  # d/dr on the same toroidal-flux radius as r1
    r_p = float(np.interp(rf[j], rho, pi_)) / max(abs(float(np.interp(rf[j], rho, dpi))), 1e-12)
    omega_si = Ti1 / (QE * B * r1 * max(r_p, 1e-3))
    rho_i = np.sqrt(m_i * Ti1) / (QE * B)
    rho_hat = rho_i / r1
    sigma = np.asarray(getattr(cp, "sigma", np.full_like(rho, np.nan)))
    eta = 1.0 / max(float(np.interp(rf[j], rho, sigma)) if np.isfinite(sigma).all() else 1e7, 1e3)
    S = MU0 * r1 ** 2 / eta / tau_A
    dW = D / s1
    out = dict(s1=s1, r1=r1, eps1=eps1, li1=li1, beta_p1=beta_p1, D=D, D_bussac=D_bussac, D_el=D_el, D_ko=D_ko, tau_A=tau_A, S=S, rho_hat=rho_hat, w_tau=omega_si * tau_A,
               r_p=r_p, Ti1_keV=Ti1 / QE / 1e3, Ti0_keV=Ti[0] / QE / 1e3)
    if -dW > 0.5 * omega_si * tau_A:                     # (14): the ideal internal kink, unstable beyond the diamagnetic threshold
        out["s_crit"] = 0.0
        out["branch"] = "ideal"
        return out
    s_dw = D / (C_RHO * rho_hat) if D > 0 else 0.0
    s_star = (omega_si * tau_A / (C_STAR * C_RHO_GAMMA * rho_hat ** (4 / 7) * S ** (-1 / 7))) ** (7 / 6)
    out["s_crit"] = float(max(s_dw, s_star))
    out["branch"] = "resistive"
    out["s_dw"], out["s_star"] = s_dw, s_star
    return out
