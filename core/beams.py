"""Neutral beams as a model, not an input (H40, H43): TORAX takes heating as a prescribed shape; MAST's backtest showed that is too crude. Its beams heat through fast ions that take tens of ms to slow down, its low-density shots lose a good part of the beam straight through the plasma,
and its stored energy and nearly all its neutrons come from those fast ions. All of that follows from the plasma and the injected power, so none of it needs steering.

Deposition: a tangential beam (tangency radius R_tan) in the midplane, attenuated as dI/ds = -I n_e sigma along its path; what is left goes through to the far wall.
Slowing down (Stix 1972): each fast ion loses energy as dE/dt = -(2 E / tau_se) (1 + (E_c / E)^(3/2)), tau_se = 6.28e14 A_b T_e^(3/2) / (Z^2 n_e lnL) (T_e in eV, n_e in m^-3), E_c = 14.8 A_b T_e (sum n_i Z_i^2 / (n_e A_i))^(2/3). From that come the share of the energy going to ions, the thermalization time, the energy held by
fast ions per unit of power (tau_eff, used as the time constant of a reservoir dW_f/dt = P_dep - W_f / tau_eff, exact at steady state) and the neutrons per fast ion. Pitch-angle scattering and the beam's radial redistribution are left out. Orbits enter once, at birth (spread): a pencil beam through the axis, flux-surface averaged, is singular there, and real ions are not. spread="larmor" is one Larmor radius; spread="orbit" is the drift-orbit width of a passing ion, off its flux surface by q times its Larmor radius in cos(theta), so rms q rho_L / sqrt(2). With f_lost="orbit", what the orbit's Gaussian puts beyond the edge is lost: no fitted number, and it grows at low current as q does.
Current (drive_current): the fast ions' parallel momentum, slowing down and scattering in pitch, with no fit. Their speed falls as dv/dt = -(v / tau_se)(1 + (v_c / v)^3) and the mean pitch as d ln xi / dt = -beta (v_c / v)^3 / tau_se, beta = Z_eff / (A_b sum n_i Z_i^2 / (n_e A_i)), which gives in steady state j_f = e S xi_0 v_0 tau_se I, I = integral of (u^3 / (u^3 + y^3)) (u^3 (1 + y^3) / (u^3 + y^3))^(beta / 3) du from 0 to 1, y = v_c / v_0 (Cordey 1976, Gaffey 1976, Mikkelsen and Singer 1983). The electrons answer with -j_f / Z_eff times (1 - G), G their trapped share's correction (Start and Cordey 1980, low collisionality; held to its range): (1.55 + 0.85 / Z_eff) eps^0.5 - (0.2 + 1.55 / Z_eff) eps. xi_0 = R_tan / R; the current takes the momentum's own time, tau_se I, to build.
No machine's numbers live here: energy, tangency radius, cross section and prompt loss come from the scenario. MAST's are mast_anchor.BEAMS.
Neutrons: beam-target D(d,n)He3, with the Bosch and Hale 1992 cross section (coefficients from memory, checked against 16 mb at 50 keV centre of mass): Y = n_D (tau_se / 2) integral of sigma(E/2) v(E) / (E (1 + (E_c / E)^1.5)) dE. Measured neutron rates are a test this model has to pass without any fitting of its own.
"""
import numpy as np
from scipy.special import erfc

import ntm

QE, MP = 1.602176634e-19, 1.67262192e-27


def sigma_dd_n(E_cm_keV):
    """D(d,n)He3 in m^2, Bosch and Hale 1992, E in keV centre of mass."""
    E = np.maximum(E_cm_keV, 0.5)
    S = (5.3701e4 + E * (3.3027e2 + E * (-1.2706e-1 + E * (2.9327e-5 + E * -2.5151e-9))))
    return S / (E * np.exp(31.3970 / np.sqrt(E))) * 1e-31       # mb to m^2


class Beams:
    def __init__(self, reactor, power, E_keV, R_tan, sigma, f_lost, A_b=2.0, A_i=2.0, n_path=2001, spread="larmor", drive_current=False, history=None, neutron_model="births",
                 stopping="fixed", cx=None, coulomb="fixed", orbit_average=False, orbit_width="q_rho"):
        """f_lost: the share lost promptly, a number, a function of the plasma current in A, or "orbit".
        history: the injected power before the start, dict(t=s, P_MW=MW), measured. With it the fast ions and their current start from that history, each cell's reservoir filled with the start's
        deposition and slowing down (H83): a beam switched on 25 ms before the start has not filled a reservoir that takes 20 to 60 ms. Without it they start at steady state, as before.
        stopping: "fixed", the one cross-section sigma, as before; "suzuki", Suzuki et al. 1998's beam stopping from the plasma along the path (beam_atomic.py, H84).
        cx: None, as before; "halo", charge exchange of the fast ions on the beam's own halo (H84): the beam neutrals that charge-exchange with thermal ions leave thermal neutrals, which live
        until electrons ionize them; the fast ions lose themselves on those as n_halo sigma_cx(E) v, all the way down in energy. Flux-surface averaged, the halo's own spreading and the wall's neutrals
        left out (so a lower bound on the loss). No number of a machine's: atomic rates from beam_atomic.py.
        coulomb: "fixed", ln Lambda 17 as before; "nrl", the electrons' ln Lambda from the plasma, 24 - ln(n_e^(1/2) / T_e) (NRL formulary, T_e above 10 eV; n_e in cm^-3, T_e in eV), about 15.5 in MAST's core.
        orbit_average: the fast ions slow down on what their drift orbit crosses, not on their birth surface alone: the drag rate 1 / tau_se, E_c, the target density and the halo
        are averaged over the orbit, rho0 + D cos(theta) with theta uniform (a passing ion's time), D = q rho_L (widths(); its rms D / 2^(1/2) is the Gaussian the deposition and
        pressure already use), reflected at the axis and held inside the edge (an orbit crossing it is the prompt loss f_lost counts) (H84). On MAST D is 0.18 to 0.38 in rho at 65 keV.
        orbit_width: "q_rho", the drift orbit's half-width q rho_L, a large aspect ratio form, as before; "pphi", from conservation of the canonical toroidal momentum,
        m R v_phi + e psi / (2 pi), with the plasma's own flux (TORAX's psi, total): a passing ion's flux changes by (2 pi m / e)(R_out v_par,out - R_in v_par,in) from the outboard
        to the inboard midplane, v_par from mu conservation with B ~ 1 / R, and an ion that cannot reach the inboard side (trapped) by (2 pi m / e) R_out v_par,out from each leg to
        its tip; the pitch at birth xi = R_tan / R_out. On MAST (measured profiles) 0.11 to 0.20 in rho for the passing core ions, about half q rho_L, and 0.31 to 0.43 for the
        trapped ones outside rho 0.5, close to it. Used everywhere the orbit width is (deposition spread, pressure, prompt loss, the orbit average)."""
        self.r, self.power, self.E0, self.R_tan, self.sigma, self.f_lost, self.A_b, self.A_i = reactor, power, E_keV, R_tan, sigma, f_lost, A_b, A_i
        self.stopping, self.cx, self.coulomb, self.orbit_average, self.orbit_width = stopping, cx, coulomb, orbit_average, orbit_width
        self.drive_current, self.J = drive_current, None
        self.neutron_model = neutron_model     # "births": the steady-state yield of the ions born now, as before; "population": the yield of the fast ions present, W / tau_eff per cell (H84), so it rises after the beams switch on as the population does
        self.history = None
        if history is not None:
            th, Ph = np.asarray(history["t"], float), np.clip(np.nan_to_num(np.asarray(history["P_MW"], float)), 0.0, None)
            keep = th <= reactor.t + 1e-9
            self.history = (th[keep], Ph[keep])
        self.n_path, self.do_spread = n_path, spread
        self.x = ntm.TURB_RHO
        self.W = None
        self.t_last = reactor.t
        self.last = {}

    def path(self):
        """Along the beam, in the midplane: step lengths and the rho at each."""
        geo = self.r.state.geometry
        rho, R_in, R_out = np.asarray(geo.rho_norm), np.asarray(geo.R_in), np.asarray(geo.R_out)
        R_edge = R_out[-1]
        s = np.linspace(-np.sqrt(max(R_edge ** 2 - self.R_tan ** 2, 1e-6)), np.sqrt(max(R_edge ** 2 - self.R_tan ** 2, 1e-6)), self.n_path)
        R = np.sqrt(self.R_tan ** 2 + s ** 2)
        q = np.interp(R, R_out, rho, left=np.nan, right=1.0)     # outboard of the axis
        inboard = R < R_out[0]
        q[inboard] = np.interp(R[inboard], R_in[::-1], rho[::-1], left=1.0, right=rho[0])
        return s[1] - s[0], q

    def deposit(self):
        """Share of the power absorbed per cell, and the share lost straight through. Also kept: self.halo_share, the share of the injected particles that charge-exchange with
        thermal ions per cell (each leaves a thermal neutral), when cx is "halo"."""
        cp = self.r.state.core_profiles
        rho, ne = np.asarray(self.r.state.geometry.rho_norm), np.asarray(cp.n_e.value)
        ds, q = self.path()
        ne_s = np.interp(q, rho, ne)
        if self.stopping == "suzuki" or self.cx == "halo":
            import beam_atomic
            Te_s = np.interp(q, rho, np.maximum(np.asarray(cp.T_e.value), 1e-3))
            Z_s = np.interp(q, rho, np.maximum(np.asarray(cp.Z_eff), 1.0))
            ni_s = np.interp(q, rho, np.asarray(cp.n_i.value))
        sig = beam_atomic.stopping(self.E0 / self.A_b, ne_s, Te_s, Z_s) if self.stopping == "suzuki" else self.sigma
        tau = np.cumsum(ne_s * sig * ds)
        absorbed = np.exp(-np.concatenate([[0.0], tau[:-1]])) - np.exp(-tau)
        cell = np.clip(np.searchsorted(rho, q), 0, len(rho) - 1)
        share = np.zeros_like(rho)
        np.add.at(share, cell, absorbed)
        if self.cx == "halo":
            f_cx = np.clip(ni_s * beam_atomic.sigma_cx(self.E0 * 1e3 / self.A_b) / np.maximum(ne_s * sig, 1e-40), 0.0, 1.0)
            self.halo_share = np.zeros_like(rho)
            np.add.at(self.halo_share, cell, absorbed * f_cx)
        return share, float(np.exp(-tau[-1]))

    def halo(self, share_inj):
        """The halo's neutral density per cell, m^-3, flux-surface averaged: its source (injected particles per second times the share that charge-exchanged there), spread by the thermal
        neutrals' random walk, over the cell's volume, times their life against electron impact ionization. The walk: steps of v_th / nu_cx between charge exchanges with the thermal ions,
        for a life of 1 / nu_ion, so a Gaussian of width L = v_th / (nu_cx nu_ion)^(1/2) (about 0.25 m at 1 keV and 3e19 m^-3) in the minor radius."""
        geo, cp = self.r.state.geometry, self.r.state.core_profiles
        import beam_atomic
        rho = np.asarray(geo.rho_norm)
        dv = np.asarray(geo.vpr) * float(np.asarray(geo.drho_norm).ravel()[0])
        N_dot = max(float(self.power(self.r.t)), 0.0) * 1e6 / (self.E0 * 1e3 * QE)
        ne, Te = np.asarray(cp.n_e.value), np.maximum(np.asarray(cp.T_e.value), 1e-3)
        ni = np.asarray(cp.n_i.value)
        Ti = np.maximum(np.asarray(cp.T_i.value), 1e-3) if getattr(cp, "T_i", None) is not None else Te
        nu_ion = np.maximum(ne * beam_atomic.ionization_rate_e(Te * 1e3), 1e-6)
        v_th = np.sqrt(2 * Ti * 1e3 * QE / (self.A_i * MP))
        nu_cx = np.maximum(ni * beam_atomic.sigma_cx(Ti * 1e3 / self.A_i) * np.sqrt(2.0) * v_th, 1e-6)
        R_in, R_out = np.asarray(geo.R_in), np.asarray(geo.R_out)
        a = 0.5 * (float(R_out[-1]) - float(R_in[-1]))
        width = np.maximum(v_th / np.sqrt(nu_cx * nu_ion) / a, 0.02)
        src = self.blur(share_inj * N_dot, rho, dv, width)
        return src / nu_ion / np.maximum(dv, 1e-9)

    def widths(self, rho):
        """In rho: the full-energy ion's Larmor radius, and its drift orbit's rms width."""
        geo = self.r.state.geometry
        B = abs(float(geo.B_0)) * float(geo.R_major) / max(float(np.asarray(geo.R_out)[0]), 1e-3)
        a = 0.5 * (float(np.asarray(geo.R_out)[-1]) - float(np.asarray(geo.R_in)[-1]))
        larmor = np.sqrt(2 * self.A_b * MP * self.E0 * 1e3 * QE) / (QE * B) / a * np.ones_like(rho)
        if self.orbit_width == "pphi":
            return larmor, self.half_width_pphi(rho) / np.sqrt(2.0)
        q = np.abs(np.asarray(self.r.state.core_profiles.q_face))
        return larmor, larmor * 0.5 * (q[1:] + q[:-1]) / np.sqrt(2.0)

    def half_width_pphi(self, rho):
        """The full-energy ion's orbit half-width in rho from its canonical toroidal momentum (see orbit_width)."""
        geo = self.r.state.geometry
        R_in, R_out = np.asarray(geo.R_in), np.asarray(geo.R_out)
        psi = np.asarray(self.r.state.core_profiles.psi.value)
        dpsi = np.maximum(np.abs(np.gradient(psi, rho)), 1e-6)
        v0 = np.sqrt(2 * self.E0 * 1e3 * QE / (self.A_b * MP))
        xi = np.minimum(self.R_tan / R_out, 1.0)
        reach = (1 - xi ** 2) * R_out / R_in                  # v_perp^2 / v^2 at the inboard midplane
        m_e = self.A_b * MP / QE
        passing = np.pi * m_e * v0 * (xi * R_out - np.sqrt(np.clip(1 - reach, 0, None)) * R_in) / dpsi
        trapped = 2 * np.pi * m_e * v0 * xi * R_out / dpsi
        return np.clip(np.where(reach > 1, trapped, passing), 1e-3, 1.0)

    @staticmethod
    def blur(per_cell, rho, dv, width):
        """A Gaussian of this width weighted by volume, the total kept."""
        K = np.exp(-0.5 * ((rho[:, None] - rho[None, :]) / width[None, :]) ** 2) * dv[:, None]
        return (K / K.sum(axis=0, keepdims=True)) @ per_cell

    def spread(self, per_cell, rho, dv):
        """What lands inside, the total kept, and the share of the total the orbit's Gaussian puts beyond the edge."""
        larmor, orbit = self.widths(rho)
        outside = 0.5 * erfc((1.0 - rho) / (np.sqrt(2.0) * orbit))
        return self.blur(per_cell, rho, dv, orbit if self.do_spread == "orbit" else larmor), float((outside * per_cell).sum() / max(per_cell.sum(), 1e-30))

    def pressure(self):
        """The fast ions' pressure per cell in Pa, or None before the first step: two thirds of their energy density, taken as isotropic though a tangential beam is not, averaged over their orbits, since that is where they exert it."""
        if self.W is None:
            return None
        geo = self.r.state.geometry
        rho, dv = np.asarray(geo.rho_norm), np.asarray(geo.vpr) * float(np.asarray(geo.drho_norm).ravel()[0])
        return (2.0 / 3.0) * self.blur(self.W, rho, dv, self.widths(rho)[1]) / dv

    def orbit_kernel(self, rho, n_theta=32):
        """K[i, j]: the share of its time an ion born in cell i spends in cell j, its orbit rho_i + D_i cos(theta), theta uniform, reflected at the axis, held at the edge."""
        D = np.sqrt(2.0) * np.maximum(self.widths(rho)[1], 1e-3)
        th = (np.arange(n_theta) + 0.5) * np.pi / n_theta            # cos(theta) over half a turn covers the orbit's radii once
        x = np.clip(np.abs(rho[:, None] + D[:, None] * np.cos(th)[None, :]), rho[0], rho[-1])
        j = np.clip(np.searchsorted(rho, x) - 1, 0, len(rho) - 2)
        f = (x - rho[j]) / (rho[j + 1] - rho[j])
        K = np.zeros((len(rho), len(rho)))
        rows = np.repeat(np.arange(len(rho)), n_theta)
        np.add.at(K, (rows, j.ravel()), (1 - f).ravel())
        np.add.at(K, (rows, j.ravel() + 1), f.ravel())
        return K / K.sum(axis=1, keepdims=True)

    def lnL(self, ne, Te):
        """The Coulomb logarithm of the fast ions on the electrons (ne in m^-3, Te in keV)."""
        if self.coulomb != "nrl":
            return 17.0
        return 24.0 - np.log(np.sqrt(np.maximum(ne, 1e15) * 1e-6) / np.maximum(Te * 1e3, 10.0))

    def slowing(self, n_halo=None):
        """Per cell: tau_se (s), E_c (keV), ion share, tau_eff (s), neutrons per fast ion. With a halo (m^-3 per cell) the fast ions also charge-exchange on it as they slow down:
        each energy is reached by the share S(E) still there, and self.delivered keeps the share of their energy that reaches the plasma rather than leaving as fast neutrals."""
        cp = self.r.state.core_profiles
        Te, ne = np.maximum(np.asarray(cp.T_e.value), 1e-3), np.asarray(cp.n_e.value)
        ni = np.asarray(cp.n_i.value)
        tau_se = 6.28e14 * self.A_b * (Te * 1e3) ** 1.5 / (ne * self.lnL(ne, Te))
        E_c = 14.8 * self.A_b * Te * (ni / (ne * self.A_i)) ** (2 / 3)
        if self.orbit_average:
            rho = np.asarray(self.r.state.geometry.rho_norm)
            K = self.orbit_kernel(rho)                           # row: an ion born in that cell, the share of its time over each cell
            drag = K * (1.0 / tau_se)[None, :]
            self.orbit_heat = drag / drag.sum(axis=1, keepdims=True)       # where an ion born in a cell gives its energy: its time there times the drag there
            self.orbit_time = K
            tau_se, E_c, ni = 1.0 / (K @ (1.0 / tau_se)), K @ E_c, K @ ni
            if n_halo is not None:
                n_halo = K @ np.asarray(n_halo)
        y = np.linspace(0.0, 1.0, 81)[1:]                       # E / E0
        dy = y[1] - y[0]
        x = (E_c / self.E0)[:, None] ** 1.5 / y[None, :] ** 1.5
        # the time spent per unit energy: dt = dE / |dE/dt| = (tau_se / 2) dE / (E (1 + x))
        dt_dy = 0.5 * tau_se[:, None] / (y[None, :] * (1 + x))
        v = np.sqrt(2 * y * self.E0 * 1e3 * QE / (self.A_b * MP))
        if n_halo is None:
            S = np.ones_like(dt_dy)
        else:
            import beam_atomic
            nu = np.asarray(n_halo)[:, None] * beam_atomic.sigma_cx(y * self.E0 * 1e3 / self.A_b)[None, :] * v[None, :]       # per s, at each energy
            lost = nu * dt_dy * dy                               # chance of charge exchange while crossing each energy step
            S = np.exp(-(np.cumsum(lost[:, ::-1], axis=1)[:, ::-1] - 0.5 * lost))       # surviving from E0 down to each energy (midpoint of the step)
        self.delivered = np.sum(S, axis=1) * dy                  # share of the birth energy given to the plasma by drag; the rest leaves as fast neutrals
        ion = np.sum(x / (1 + x) * S, axis=1) * dy / np.maximum(self.delivered, 1e-12)
        tau_eff = np.sum(dt_dy * y[None, :] * S, axis=1) * dy
        Y = ni * np.sum(dt_dy * sigma_dd_n(y * self.E0 / 2)[None, :] * v[None, :] * S, axis=1) * dy
        return tau_se, E_c, ion, tau_eff, Y

    def current(self, P_dep, dt):
        """The driven toroidal current per cell, A, net of the electrons, kept in a reservoir with the momentum's time. Returns the current density against TURB_RHO and the total."""
        geo, cp = self.r.state.geometry, self.r.state.core_profiles
        rho, spr = np.asarray(geo.rho_norm), np.asarray(geo.spr)
        drho = float(np.asarray(geo.drho_norm).ravel()[0])
        R_in, R_out = np.asarray(geo.R_in), np.asarray(geo.R_out)
        R, eps = 0.5 * (R_in + R_out), (R_out - R_in) / (R_out + R_in)
        Te, ne, ni = np.maximum(np.asarray(cp.T_e.value), 1e-3), np.asarray(cp.n_e.value), np.asarray(cp.n_i.value)
        Z = np.maximum(np.asarray(cp.Z_eff), 1.0)
        tau_se = 6.28e14 * self.A_b * (Te * 1e3) ** 1.5 / (ne * self.lnL(ne, Te))
        E_c = 14.8 * self.A_b * Te * (ni / (ne * self.A_i)) ** (2 / 3)
        y3 = (E_c / self.E0) ** 1.5
        beta = Z / (self.A_b * (ni / (ne * self.A_i) + np.maximum(Z - ni / ne, 0.0) / np.asarray(cp.A_impurity)))
        u = np.linspace(0.0, 1.0, 81)[1:]
        share = u[None, :] ** 3 / (u[None, :] ** 3 + y3[:, None])
        I = np.sum(share * (share * (1.0 + y3[:, None])) ** (beta[:, None] / 3.0), axis=1) * (u[1] - u[0])
        v0 = np.sqrt(2 * self.E0 * 1e3 * QE / (self.A_b * MP))
        xi0 = np.minimum(self.R_tan / R, 1.0)
        J_ss = QE * (P_dep / (self.E0 * 1e3 * QE)) * xi0 * v0 * tau_se * I / (2 * np.pi * R)     # A per cell
        tau_J = np.maximum(tau_se * I, 1e-6)
        if self.J is None:
            P_inj = max(float(self.power(self.r.t)) * 1e6, 1.0)
            self.J = J_ss if self.history is None else self.filled(J_ss / (P_inj * tau_J), tau_J, J_ss)      # steady at the start, or filled from the measured history
        elif dt > 0:
            a = np.exp(-dt / tau_J)
            self.J = self.J * a + J_ss * (1 - a)
        G = np.clip((1.55 + 0.85 / Z) * np.sqrt(eps) - (0.2 + 1.55 / Z) * eps, 0.0, 1.0)
        net = self.J * (1.0 - (1.0 - G) / Z)
        return np.interp(self.x, rho, net / (spr * drho)), float(net.sum())

    def before_step(self):
        """Heat and particles for the next step, from what the fast ions do now."""
        self.prepare()
        return self.finish(self.halo_own)

    def prepare(self):
        """The first half of a step: deposition, and this beam's own halo (kept as self.halo_own). Mixed calls it on every energy component before any finishes, so all of them see the whole halo."""
        t = self.r.t
        self.dt_step = t - self.t_last
        self.t_last = t
        self.P_step = float(self.power(t)) * 1e6
        self.deposited = self.deposit()
        self.halo_own = self.halo(self.halo_share) if self.cx == "halo" else None

    def finish(self, n_halo):
        """The second half: slowing down on the halo given, the reservoir, heat, particles and neutrons."""
        r = self.r
        dt, P = self.dt_step, self.P_step
        rho, dv = np.asarray(r.state.geometry.rho_norm), np.asarray(r.state.geometry.vpr) * float(np.asarray(r.state.geometry.drho_norm).ravel()[0])
        share, through = self.deposited
        _, _, ion, tau_eff, Y = self.slowing(n_halo)
        spreading, outside = self.spread(share, rho, dv) if self.do_spread else (share, 0.0)
        lost = outside if self.f_lost == "orbit" else self.f_lost(abs(r.get_param("profile_conditions.Ip"))) if callable(self.f_lost) else self.f_lost
        P_dep = P * (1 - lost) * spreading
        if self.W is None:
            self.W = P_dep * tau_eff if self.history is None else self.filled(P_dep / max(P, 1.0), tau_eff, P_dep * tau_eff)     # steady state at the start, or filled from the measured history
        elif dt > 0:
            a = np.exp(-dt / tau_eff)
            self.W = self.W * a + P_dep * tau_eff * (1 - a)
        out = self.W / np.maximum(tau_eff, 1e-6)                # the birth energy flowing through the reservoir, J per s per cell; at steady state P_dep
        Q = out * self.delivered                                 # to the plasma; the rest (with a halo) leaves as fast neutrals
        Q_i, Q_e = Q * ion, Q * (1 - ion)
        rate = out / (self.E0 * 1e3 * QE)                       # fast ions finishing per second, which fuels (one charge-exchanged leaves a thermal ion behind in its place)
        if self.orbit_average:                                   # given along the orbit, not on the birth surface alone
            Q_i, Q_e, rate = self.orbit_heat.T @ Q_i, self.orbit_heat.T @ Q_e, self.orbit_time.T @ rate
        at = lambda per_cell: np.interp(self.x, rho, per_cell / dv)
        self.last = dict(through=through, W_fast=float(self.W.sum()), P_abs=float(P_dep.sum()), P_e=float(Q_e.sum()), P_i=float(Q_i.sum()), neutrons=float((P_dep / (self.E0 * 1e3 * QE) * Y).sum()) if self.neutron_model == "births" else float((out / (self.E0 * 1e3 * QE) * Y).sum()), lost=float(lost),
                         P_cx=float((out * (1 - self.delivered)).sum()), n_halo=0.0 if n_halo is None else float(np.sum(n_halo * dv) / np.sum(dv)))
        if self.drive_current:
            ntm.BEAM_CURRENT[:], self.last["I_driven"] = self.current(P_dep, dt)
        return (at(Q_e), at(Q_i)), at(rate), self.last

    def filled(self, per_W, tau, steady):
        """A reservoir per cell, d X/dt = per_W P(t) - X / tau, run over the measured injected power before the start (per_W: X per watt injected at steady state, divided by tau). Falls back to steady."""
        th, Ph = self.history
        if len(th) < 2:
            return steady
        X = np.zeros_like(np.asarray(steady, float))
        for i in range(len(th) - 1):
            dt = th[i + 1] - th[i]
            if dt <= 0:
                continue
            a = np.exp(-dt / np.maximum(tau, 1e-6))
            X = X * a + per_W * Ph[i] * 1e6 * tau * (1 - a)
        return X

    def state(self):
        return dict(W=None if self.W is None else self.W.copy(), J=None if self.J is None else self.J.copy(), t_last=self.t_last)

    def set_state(self, x):
        self.W, self.t_last = None if x["W"] is None else x["W"].copy(), x["t_last"]
        self.J = None if x.get("J") is None else x["J"].copy()


def make(reactor, power, species=None, **kw):
    """Beams, or Mixed when the beam line's energy components are given (species: power shares of E0, E0 / 2, E0 / 3)."""
    return Mixed(reactor, power, fractions=species, **kw) if species else Beams(reactor, power, **kw)


class Mixed:
    """A positive-ion beam: its full, half and third energy components, each a Beams with its share of the injected power, their outputs added. fractions are the power shares of
    E0, E0 / 2 and E0 / 3 leaving the injector, a beam line's hardware (docs/lit/41). The fast ions of every component see the halo all components make. Same interface as Beams."""
    def __init__(self, reactor, power, E_keV, fractions, history=None, **kw):
        self._power = power
        self.parts = []
        for k, f in zip((1, 2, 3), fractions):
            if f <= 0:
                continue
            h = None if history is None else dict(t=history["t"], P_MW=np.asarray(history["P_MW"], float) * f)
            self.parts.append((float(f), Beams(reactor, (lambda t, f=float(f): f * float(self._power(t))), E_keV / k, history=h, **kw)))
        self.r, self.E0, self.last = reactor, E_keV, {}
        self.drive_current = any(b.drive_current for _, b in self.parts)

    @property
    def power(self):
        return self._power

    @power.setter
    def power(self, p):                      # the components read it through their own lambdas, so a scaled programme (sensors_past.py) reaches all of them
        self._power = p

    @property
    def W(self):
        ws = [b.W for _, b in self.parts]
        return None if any(w is None for w in ws) else sum(ws)

    def pressure(self):
        ps = [b.pressure() for _, b in self.parts]
        return None if any(p is None for p in ps) else sum(ps)

    def before_step(self):
        for _, b in self.parts:
            b.prepare()
        halos = [b.halo_own for _, b in self.parts if b.halo_own is not None]
        n_halo = sum(halos) if halos else None
        Q_e = Q_i = rate = 0.0
        lasts, I = [], 0.0
        J = np.zeros_like(ntm.BEAM_CURRENT) if self.drive_current else None
        for _, b in self.parts:
            (qe, qi), rt, last = b.finish(n_halo)
            Q_e, Q_i, rate = Q_e + qe, Q_i + qi, rate + rt
            lasts.append(last)
            if self.drive_current:
                J += ntm.BEAM_CURRENT
                I += last.get("I_driven", 0.0)
        if self.drive_current:
            ntm.BEAM_CURRENT[:] = J
        add = lambda k: float(sum(x.get(k, 0.0) for x in lasts))
        share = lambda k: float(sum(f * x.get(k, 0.0) for (f, _), x in zip(self.parts, lasts)))
        self.last = dict(W_fast=add("W_fast"), P_abs=add("P_abs"), P_e=add("P_e"), P_i=add("P_i"), neutrons=add("neutrons"), P_cx=add("P_cx"), through=share("through"), lost=share("lost"),
                         n_halo=0.0 if n_halo is None else lasts[0].get("n_halo", 0.0))
        if self.drive_current:
            self.last["I_driven"] = I
        return (Q_e, Q_i), rate, self.last

    def state(self):
        return [b.state() for _, b in self.parts]

    def set_state(self, x):
        for (_, b), s in zip(self.parts, x):
            b.set_state(s)
