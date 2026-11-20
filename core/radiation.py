"""What MAST's bolometers would read from the twin (H86): the total radiated power, as the store's prad_pol is a weighted sum over the poloidal fan's
chords that sees the core, the edge, the X-points and the divertor legs (H85: the outer chords, which graze the edge and the X-points, read the most).

The budget, each term from physics or published multi-machine numbers, none set on MAST:
  core       TORAX's own line radiation and bremsstrahlung inside the separatrix (carbon out of coronal balance where the run has noncoronal on)
  sol_c      carbon in the scrape-off layer and divertor: TORAX's extended Lengyel model (edge_radiation.py, docs/lit/37) with the heat-flux width from
             Eich et al. 2013 regression 14, 0.63 B_pol,omp^-1.19 mm (multi-machine H-mode, MAST among them) in place of the model's own drift width,
             which lit 37 found 2.3 to 3.3 times below MAST's measured widths; the separatrix density from sol.py where the run has it, else 0.3 of
             the line average (Porter et al. 1999, the multi-machine edge database); carbon from the twin's Z_eff, fully stripped
  h_main     the Lyman-alpha of the main chamber's atoms: KN1D's own emissivity (Johnson-Hinnov rates) across its slab, spread over the separatrix
             area like its ionization; only where the run carries neutrals (fuelling kn1d_gauge)
  h_div      the Lyman-alpha of the fuel recycled at the targets: the targets' ion flux from the two-point model (sol.py; at 0.3 of the line average
             where the run has no SOL model) times the Lyman-alpha energy per ionization KN1D finds in the main chamber (10.5 to 12.4 eV on MAST's frames, H86)
Not counted: molecular bands, charge-exchange neutrals reaching the foils, and carbon sources beyond the core's concentration. Each term is recorded
apart (rad_core, rad_sol_c, rad_h_main, rad_h_div) and the reading as rad_bolo = combine(terms): the fan sees the core, the scrape-off layer's carbon,
hydrogen's lines at ADAS's energy per ionization (H_ALL_LINES times KN1D's Lyman-alpha) and the inner legs' tenth of the divertor's recycling (DIV_IN_VIEW).
whole_shot.py scores rad_bolo against prad_pol where a run has the terms, recombining them with the same function.
"""
import math

import numpy as np

QE, MU0 = 1.602176634e-19, 4e-7 * np.pi
E_LY_DEFAULT_EV = 11.0           # Lyman-alpha energy per ionization when no KN1D result is at hand (KN1D on MAST frames: 10.5 to 12.4 eV)
N_SEP_FRACTION = 0.3            # n_sep / n_line where the run has no SOL model: Porter et al. 1999, the multi-machine edge database (edge_radiation.py)


class Bolometer:
    def __init__(self, interval=0.01, c_carbon=None):
        self.interval, self.c_carbon = interval, c_carbon
        self.t_last = -1e9
        self.last = dict(rad_core=0.0, rad_sol_c=0.0, rad_h_main=0.0, rad_h_div=0.0, rad_bolo=0.0)
        self.failed = 0

    def __call__(self, r, sol=None, neutrals=None):
        import sol as sol_lib
        t = float(r.t)
        ppo = r.ppo
        core = abs(float(getattr(ppo, "P_radiation_e", 0.0))) + abs(float(getattr(ppo, "P_bremsstrahlung_e", 0.0)))
        self.last["rad_core"] = core
        if t - self.t_last >= self.interval - 1e-9:
            self.t_last = t
            geo, cp = r.state.geometry, r.state.core_profiles
            g = (sol.geometry(r) if sol is not None else sol_lib.SOL().geometry(r))
            Ip = abs(float(r.carry.get("Ip_now", 0.0) or 0.0))         # the coupled step's plasma current (A)
            if Ip <= 0:
                Ip = abs(float(r.get_param("profile_conditions.Ip")))
            kappa, delta = float(np.asarray(geo.elongation_face)[-1]), float(np.asarray(geo.delta_face)[-1])
            L_pol = 2 * np.pi * g["a"] * math.sqrt(0.5 * (1 + kappa ** 2))
            B_avg = MU0 * Ip / max(L_pol, 1e-3)
            ratio = float(np.clip(g["B_pol"] / max(B_avg, 1e-6), 0.3, 2.0))
            lam_mm = sol_lib.lambda_q_eich14(g["B_pol"]) * 1e3
            n_sep = None
            if sol is not None and sol.last:
                n_sep = float(sol.last["sol_n_sep"])
            n_line = float(getattr(ppo, "n_e_line_avg", 0.0))
            Z_eff = float(np.mean(np.asarray(cp.Z_eff)))
            P_SOL = float(getattr(ppo, "P_SOL_total", 0.0))
            sol_c = 0.0
            if P_SOL > 1e4 and Ip > 1e4:
                try:
                    import edge_radiation as er
                    out = er.edge_radiation(P_SOL=P_SOL, R0=g["R0"], a=g["a"], kappa=kappa, delta=delta, B0=abs(float(geo.B_0)), Ip=Ip, q95=float(getattr(ppo, "q95", 6.0)),
                                            n_line=n_line if n_sep is None else None, n_sep=n_sep, c_carbon=self.c_carbon, Z_eff=None if self.c_carbon is not None else max(Z_eff, 1.0001),
                                            bpol_omp_over_avg=ratio, lambda_q_mm=lam_mm)
                    sol_c = float(out.P_rad) if np.isfinite(out.P_rad) else 0.0
                except Exception as err:          # the Lengyel solve can fail on an odd state: the term stays at its last value, and the count says how often
                    self.failed += 1
                    sol_c = self.last["rad_sol_c"]
                    if self.failed <= 3:
                        print(f"radiation: SOL carbon failed at {t:.3f}: {repr(err)[:120]}", flush=True)
            self.last["rad_sol_c"] = sol_c
            h_main, e_ly = 0.0, E_LY_DEFAULT_EV
            if neutrals is not None and neutrals.last.get("kn1d_ly") is not None:
                h_main = float(neutrals.last["kn1d_ly"]) * g["A_sep"]
                if neutrals.last.get("kn1d_ion_all", 0.0) > 0:
                    e_ly = float(neutrals.last["kn1d_ly"]) / float(neutrals.last["kn1d_ion_all"]) / QE
            self.last["rad_h_main"] = h_main
            if sol is not None and sol.last:
                G_t = float(sol.last["sol_G_target"])
            else:                                  # the targets' flux from the same two-point model at the published separatrix density, as a reading only
                n_u = N_SEP_FRACTION * n_line
                lam = lam_mm * 1e-3
                q_par = sol_lib.SOL().f_div * max(P_SOL, 1e4) / (2 * np.pi * g["R_omp"] * lam * g["B_pol"] / g["B_omp"])
                _, T_t, n_t = sol_lib.two_point(q_par, math.pi * float(getattr(ppo, "q95", 6.0)) * g["R0"], max(n_u, 1e17))
                G_t = n_t * sol_lib.c_s(T_t) * (2 * np.pi * g["R_omp"] * lam * g["B_pol"] / g["B_omp"]) / sol_lib.SOL().f_div
                self.last["rad_T_t"], self.last["rad_n_t"] = T_t, n_t
            h_div = G_t * e_ly * QE
            self.last["rad_G_target"] = G_t
            self.last["rad_h_div"] = h_div
            self.last["rad_e_ly"] = e_ly
        self.last["rad_bolo"] = combine(self.last)
        r.carry.update(self.last)
        return self.last


H_ALL_LINES = 16.0 / 11.0       # hydrogen's light over Lyman-alpha alone: ADAS 2012 gives 16 eV radiated per ionization at 10 eV and 1e19 m^-3 (docs/lit/44, computed there), KN1D's Lyman-alpha 10.5 to 12.4 eV
DIV_IN_VIEW = 0.1               # the share of the targets' recycling the fan sees: its steepest chord crosses R 1.0 m at Z 1.05 m, below MAST's outer strike points; the inner legs, in view,
                                # take 1 part in 10 of the power in double null (Ahn et al. 2001, quoted by Paul et al. 2005), and their recycling is taken in the same share


def combine(t):
    """The reading from its terms (a run's record or the module's last): what the poloidal fan sees. Used by the run and by whole_shot.py alike."""
    g = lambda k: float(t.get(k, 0.0) or 0.0) if np.isfinite(float(t.get(k, 0.0) or 0.0)) else 0.0
    return g("rad_core") + g("rad_sol_c") + H_ALL_LINES * (g("rad_h_main") + DIV_IN_VIEW * g("rad_h_div"))
