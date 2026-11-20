"""Atomic rates for the beam model (beams.py, H84): charge exchange, ionization, beam stopping. Published fits only, no number of any machine's; sources, checks and the
disagreements between sources in docs/lit/41.

sigma_cx(E):          H+ + H(1s) charge exchange, total over final states (a fast ion is lost whichever state the electron lands in), E the relative energy in eV per amu
                      (a deuteron at the same speed as a proton): Janev, Reiter and Samm 2003 (Jul-4105) eq. 44, table 9, n = 1, stated accuracy 10 to 20% from 0.1 eV to 2 MeV;
                      the same numbers as FIDASIM's p_cx_1_janev. Riviere 1971 (Freeman and Jones 1974) is 16 to 38% higher below 33 keV/amu: sigma_cx_riviere.
ionization_rate_e(T): electron impact ionization of H(1s), m^3/s, T_e in eV: Voronov 1997, rate = A (1 + P U^0.5) / (X + U) U^K exp(-U), U = 13.6 / T_e,
                      A 2.91e-8 cm^3/s, P 0, X 0.232, K 0.39, valid 1 eV to 20 keV.
stopping(E, n_e, T_e, Z_eff): a hydrogen-isotope beam's effective (multistep) stopping cross-section in a deuterium plasma with carbon, m^2: Suzuki et al. 1998, eqs. 24 and 28 and
                      the low-energy tables (3a row D, 3b row C for Z_eff 1 to 5), as in STELLOPT's suzuki_sigma.f90 (MIT) and ASCOT5's suzuki.c (LGPL-3.0), with the single-impurity
                      form sigma_H [1 + (Z_eff - 1) S_C]. E in keV per amu, valid 10 to 100 keV/amu; n_e 1e18 to 1e21 m^-3 and E / 100 < T_e < E / 2 (T_e in keV), clamped as BEAMS3D does.
Checks against docs/lit/41's hand arithmetic (65 keV D into D, n_e 3e19 m^-3, T_e 1 keV, Z_eff 1.5): stopping 5.41e-20, 7.99e-20, 9.61e-20 m^2 for the full, half and third energy
atoms; sigma_cx 2.43e-20 m^2 at 32.5 keV/amu.
"""
import numpy as np

# Suzuki 1998, 10 <= E < 100 keV/amu, plasma ion D (table 3a): A1 to A10
_A_D = (-67.9, -1.22, 0.0814, 0.0139, 0.454, 0.465, -0.273, 0.0751, -0.0630, -5.08e-4)
# carbon, Z_eff 1 to 5 (table 3b, row C "a"): B1 to B12
_B_C = (0.161, 0.598e-1, -0.336e-2, -0.426e-2, -0.157, -0.396e-1, 0.460e-2, 0.219e-2, 0.391e-1, 0.711e-2, -0.144e-2, -0.385e-3)
# Janev-Smith charge exchange, n = 1: A1 to A6
_CX = (3.2345, 2.3588e2, 2.3713, 3.8371e-2, 3.8068e-6, 1.1832e-10)


def sigma_cx(E_eV_amu):
    """Charge exchange of a proton (or deuteron, at the same speed) with H(1s), total over final states, m^2 (Janev-Smith)."""
    E = np.maximum(np.asarray(E_eV_amu, float), 1e-3) * 1e-3          # keV/amu
    A1, A2, A3, A4, A5, A6 = _CX
    return A1 * np.log(A2 / E + A3) / (1 + A4 * E + A5 * E ** 3.5 + A6 * E ** 5.4) * 1e-20


def sigma_cx_riviere(E_eV_amu):
    """Riviere 1971 / Freeman and Jones 1974, m^2, for comparison."""
    E = np.maximum(np.asarray(E_eV_amu, float), 1.0)
    s = 0.6937e-14 * (1 - 0.155 * np.log10(E)) ** 2 / (1 + 0.1112e-14 * E ** 3.3)
    return np.maximum(s, 0.0) * 1e-4


def ionization_rate_e(T_eV):
    """Electron impact ionization of H(1s), m^3/s (Voronov 1997)."""
    U = 13.6 / np.clip(np.asarray(T_eV, float), 1.0, 2e4)
    return 2.91e-8 / (0.232 + U) * U ** 0.39 * np.exp(-U) * 1e-6


def stopping(E_keV_amu, n_e, T_e_keV, Z_eff):
    """Suzuki 1998 effective beam stopping cross-section, m^2, for a beam at E keV/amu in a D plasma with carbon."""
    E = np.clip(np.asarray(E_keV_amu, float), 10.0, 99.999)
    ne = np.clip(np.asarray(n_e, float), 1e18, 1e21)
    Te = np.clip(np.asarray(T_e_keV, float), E / 100.0, E / 2.0)
    Z = np.clip(np.asarray(Z_eff, float), 1.0, 5.0)
    lE, N, U = np.log(E), ne / 1e19, np.log(Te)
    A1, A2, A3, A4, A5, A6, A7, A8, A9, A10 = _A_D
    sH = (A1 / E) * (1 + A2 * lE + A3 * lE ** 2) * (1 + (1 - np.exp(-A4 * N)) ** A5 * (A6 + A7 * lE + A8 * lE ** 2)) * (1 + A9 * U + A10 * U ** 2) * 1e-20
    B = _B_C
    lN = np.log(N)
    S = (B[0] + B[1] * U + B[2] * lN + B[3] * lN * U + B[4] * lE + B[5] * lE * U + B[6] * lE * lN + B[7] * lE * lN * U
         + B[8] * lE ** 2 + B[9] * lE ** 2 * U + B[10] * lE ** 2 * lN + B[11] * lE ** 2 * lN * U)
    return sH * (1 + (Z - 1) * S)


if __name__ == "__main__":
    for k, E in ((1, 32.5), (2, 16.25), (3, 10.833)):
        print(f"E {E:6.2f} keV/amu: stopping {stopping(E, 3e19, 1.0, 1.5):.3e} m^2 (lit 41: {(5.41e-20, 7.99e-20, 9.61e-20)[k - 1]:.2e}); "
              f"cx Janev-Smith {sigma_cx(E * 1e3):.3e}, Riviere {sigma_cx_riviere(E * 1e3):.3e} m^2")
    print(f"cx at 0.1, 1, 100 keV/amu: {sigma_cx(100.0):.3e} {sigma_cx(1e3):.3e} {sigma_cx(1e5):.3e} m^2 (lit 41: 2.50e-19 1.70e-19 9.98e-22)")
    print(f"Voronov at 10 eV and 1 keV: {ionization_rate_e(10.0):.3e} {ionization_rate_e(1e3):.3e} m^3/s (lit 41: 5.29e-15 2.19e-14)")
