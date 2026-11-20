"""One coupled plasma on any machine (H49): TORAX, FreeGSNKE, EPED-NN, beams, the ELM cycle, sawteeth, islands and a beta limit, each reading the others through the plasma's state, with no number fitted to a reactor. What a machine brings is hardware and its programme (machines.py). Every coefficient here is published and multi-machine, and docs/lit/28 to 30 say where it comes from and how far it can be trusted; where a published rule fails on some machine (the L-H threshold on spherical tokamaks is 3 to 8 times low) it is left failing, and measured, not tuned.

Each step, before TORAX moves:
  power      P_in = P_SOL + P_rad from TORAX, which holds the beams, heating and fusion; P_net = P_in - dW/dt; tau_E = W / P_in
  L-H        P_net against Martin 2008 with 2/M, a low-density branch at 0.35 n_GW, and (3 / A)^2.2 below A = 3, which carries the measured thresholds of MAST, NSTX and Pegasus (lh_general.py). Back to L below 0.8 of that, TORAX's own default; JET, AUG and NSTX show little hysteresis when dW/dt is kept, so this is mostly numerical. Geometry is whatever the equilibrium says now
  transport  where a physics surrogate is valid (R/a above about 2.7: docs/lit/30) it decides alone. Below that no network in TORAX is valid, so Bohm-gyroBohm gives the shape, a measured start the profile of the multipliers, and the level is held to the confinement law of the state, absolute, as METIS does everywhere: d ln g / dt = ln(W_th / (tau_law P_net)) / tau_law. Thermal energy alone: ITER89-P was fitted on total energy in plasmas where fast ions hold a small share, and at low density MAST's fast ions can hold more than the law itself, which then winds the thermal transport up without end. Ions take g whole and electrons g^0.5: at low aspect ratio ions run near neoclassical and electrons stay anomalous (NSTX, MAST). The split's size is a free choice
  edge       the pedestal top follows from the confinement law of the state (ITER89-P in L, IPB98(y,2) in H, absolute) and a stiff core, T_top ~ W_law / (n V), capped in H by EPED-NN's limit on the actual shape and beta; where a surrogate is valid EPED-NN alone sets it in H. All moves take 2 tau_law (de Vries 2018: 1.5 to 3 tau_E on eight machines; on the MAST fit shots the measured top follows this to 10 to 20% from 100 ms on)
  density    feedback on the line average, to the machine's density programme, or to a held Greenwald fraction if it has none (every integrated simulator closes density on a programme: docs/lit/30). A machine running a set gas programme, as MAST, gets on top of that the rise in particle confinement in H, taken as the rise in energy confinement, tau_98 / tau_89. On MAST the line density over the programme's ratio is 0.96, 1.10, 1.12, 1.04 at 50 to 200 ms. Particle transport is tied to the heat channels, D = eta chi_e chi_i / (chi_e + chi_i) (Garzotti 2003, as TORAX's Bohm-gyroBohm), so the core answers the edge at the plasma's own particle time; a real shot's measured profile fixes only V/D
  ELMs       in H: size from the seven-tokamak fit of Perillo 2026, frequency from the power share they carry (0.3; Herrmann 2002, Huijsmans 2016: 0.2 to 0.4), 4% of the pedestal's particles (Connor 2008)
  sawteeth   TORAX's own trigger and mixing
  islands    modified Rutherford with r Delta' = -m (1 - x / 0.9) / (1 - x), x the share of the beta limit below: -m far from the limit, a pole at it (Brennan 2003), crossing zero at 90% of it (Turco 2013, DIII-D; Snape 2012 on MAST: about 1.6 suffices near the limit, which this gives at x of 0.94), so a 2/1 mode can grow with no seed as the limit nears; Sauter's small-island width; seeded at a crash by 2 w_d times (rho_q1 / rho_q2)^2, as a 1/1 mode's reach falls off towards q = 2, at a collapse by 2 w_d, at an ELM by w_d / 4 (seed size has no first-principles rule, and these are the free choices); rotation from the beam's torque held for tau_E; locks at the wall's rate; a locked island ends the shot at w / a of 0.3
  collapse   kinetic beta_N against 4 li(1) (Strait 1994, Howl 1992), with the limit's published scatter of 18% drawn afresh each time, as nobody knows it better than that. A crossing flattens T and n inside q = 2 and ends H-mode. Flattening alone loses nothing, so 40% of the heat and 25% of the particles go with it: the published sizes (40% of heat, IRE simulation of Hayashi; density down up to 40%, TST-2). MAST's own measured sizes (heat to 0.33, density to 0.68) are not used
Usage: from natural.py, `run --models beams hmode elms sawtooth modes collapse free --overrides '{"general": true}'`, or `coupled.build_shot`.
"""
import os

import numpy as np

import lh_general
import machines

QE, MU0, MP = 1.602176634e-19, 4e-7 * np.pi, 1.67262192e-27
ELM_POWER_SHARE = 0.3                  # share of the exhaust power carried by ELMs
ELM_PARTICLES = 0.04                   # share of the pedestal's particles lost at an ELM
LIMIT_SCATTER = 0.18                   # relative error of the no-wall limit as a rule (Piccione 2020)
LOCK_ENDS_AT = 0.3                     # w / a at which a locked island ends the shot (de Vries 2016)
SEED_CRASH, SEED_ELM = 2.0, 0.25       # seed island in units of w_d: about 5 cm and 0.6 cm where w_d is 2.5 cm (measured seeds 2 to 8 cm, a simulated ELM seed 0.6 cm: docs/lit/29, section 5)
COLLAPSE_HEAT, COLLAPSE_PARTICLES = 0.4, 0.25     # share lost at a collapse
POLE_ZERO, POLE_CAP = 0.9, 0.97        # r Delta' crosses zero at this share of the no-wall limit; and it is taken no closer to the pole than this, where it reads about +5, since nothing here knows the limit to better than several percent
HYSTERESIS = 0.8
ELECTRON_SHARE = 0.5                   # how much of the correction to the transport level, in log, the electrons take where the law holds it: a free choice
SURROGATE_ASPECT = 2.7                 # below this R/a QLKNN leaves its training range inside r/a 0.85 (docs/lit/30, section 5)


def tau_89(Ip_MA, R, a, kappa, n20, B, M, P_MW):
    """ITER89-P, L-mode, s."""
    return 0.048 * Ip_MA ** 0.85 * R ** 1.2 * a ** 0.3 * kappa ** 0.5 * n20 ** 0.1 * B ** 0.2 * M ** 0.5 * P_MW ** -0.5


def tau_98(Ip_MA, R, a, kappa, n20, B, M, P_MW):
    """IPB98(y,2), ELMy H-mode, s."""
    return 0.0562 * Ip_MA ** 0.93 * B ** 0.15 * (10 * n20) ** 0.41 * P_MW ** -0.69 * R ** 1.97 * kappa ** 0.78 * (a / R) ** 0.58 * M ** 0.19


def rotation_from_beam(P_abs, E_keV, R_tan, A_b, tau, n_i_mean, V, R, A_i):
    """The plasma's rotation, Hz, from the beam's torque held for tau: T = (P / E) sqrt(2 m E) R_tan, I = N m R^2. Zero with no beam: the intrinsic part is not modelled."""
    E = E_keV * 1e3 * QE
    torque = P_abs / E * np.sqrt(2 * A_b * MP * E) * R_tan
    return torque * tau / max(n_i_mean * V * A_i * MP * R ** 2, 1e-30) / (2 * np.pi)


KINK_DQ = 0.12          # the critical q_min - 1 of MAST's ideal n = 1 internal kink (Chapman et al. 2010, section 3.3)

class Coupled:
    SNAPSHOT_SHARED = ("oracle", "density_programme")      # measured traces and the machine's programme: read, never written, during a run
    """The hook that ties the models together, one per reactor. `programme` holds the machine's plan: Ip(t) in A and, if it has one, density(t) in any unit."""

    def __init__(self, h, density=None, H0=False, seed=0, models=(), tau_w=None, beam=None, last=None, feedback=True):
        self.h, self.r, self.density_programme = h, h.reactor, density
        self.feedback, self.after_collapse = feedback, 1.0      # a machine with density feedback refills after a collapse; one running a set gas programme stays down
        self.models = set(models)
        self.H, self.H0 = bool(H0), bool(H0)
        self.rng = np.random.default_rng(seed)
        self.fT, self.fn = 1.0, 1.0
        self.t_last = self.r.t
        self.s0 = None
        self.tau_E, self.P_net = None, None
        self.W_prev = None
        self.t_hold = -np.inf                  # no H-mode before this: a collapse leaves the edge gone for some time
        self.x_crit = None
        self.h_log, self.seeds, self.collapses = [], [], []
        self.tau_w = tau_w
        self.beam_settings = beam or {}
        self.eped_model = None
        self.status = {}
        self.rot_start, self.f_rot_source = None, "beam torque"
        self.table_start, self.g = None, 1.0
        self.pole = True
        self.seeding = True                    # crashes and ELMs seed islands; off when a test sets its own seeds
        self.oracle = {}
        self.fn_gain = 1.0
        self.trim = 1.0
        self.settle = 0.0
        self.edge_law = None                   # "cap": the edge from the confinement law and a stiff core, capped by EPED-NN, on any machine; "law": with no cap
        self.P_extra = None                    # power that TORAX's own numbers miss, when radiation comes in as a net source
        self.ped_width = False                 # place the pedestal top by EPED1's width law, not at a set radius
        self.relative = False                  # a measured start: the laws and EPED-NN give changes from the start, not absolute values, so what this plasma has that they miss stays
        self.law0, self.T_lim_ref = None, None
        self.h_rise = True                     # a set gas programme: particle confinement rises with energy confinement in H
        self.follow = None                     # dict(tau, shape): the particle source that holds the line density on its target, as every integrated simulator's adaptive source does (docs/lit/33); None leaves it to the edge alone

    # ---------- what the plasma is now
    def now(self):
        r = self.r
        st, ppo = r.state, r.ppo
        geo, cp = st.geometry, st.core_profiles
        beam = self.h.plant.beam
        W_fast = (beam.last or {}).get("W_fast", 0.0) if beam is not None else 0.0
        W_th = max(float(ppo.W_thermal_total), 1.0)
        P_in = max(float(ppo.P_SOL_total) - float(ppo.P_radiation_e) + (self.P_extra(r.t) if self.P_extra else 0.0), 1e3)
        rho = np.asarray(geo.rho_norm)
        top = float(r.get_param("pedestal.rho_norm_ped_top"))
        i_top = int(np.argmin(np.abs(rho - top)))
        n_e, T_e = np.asarray(cp.n_e.value), np.asarray(cp.T_e.value)
        A_i = float(np.ravel(np.asarray(cp.A_i))[0])
        a_, kappa_ = float(geo.a_minor), float(np.asarray(geo.elongation_face)[-1])
        V_ = float(np.asarray(geo.volume_face)[-1])
        li1 = float(ppo.li3) * (2 * np.pi * a_) ** 2 * 0.5 * (1 + kappa_ ** 2) * float(geo.R_major) / (2 * V_)      # TORAX gives li(3); limits are stated in li(1), the mean poloidal field squared over its value at the edge
        return dict(st=st, ppo=ppo, geo=geo, cp=cp, W_th=W_th, W_fast=W_fast, P_in=P_in, Ip=abs(float(np.asarray(cp.Ip_profile_face)[-1])) / 1e6, R=float(geo.R_major), a=float(geo.a_minor), B=abs(float(geo.B_0)), kappa=float(np.asarray(geo.elongation_face)[-1]),
                    V=float(np.asarray(geo.volume_face)[-1]), n20=max(float(ppo.n_e_line_avg) / 1e20, 1e-3), Z_eff=float(np.mean(np.asarray(cp.Z_eff))), M=A_i, n_top=float(n_e[i_top]), T_top=float(T_e[i_top]), i_top=i_top, rho=rho, beta_N=float(ppo.beta_N) * (1.0 + W_fast / W_th), li=li1)

    def confinement_ratio(self, x):
        """IPB98(y,2) over ITER89-P for this plasma: how far H-mode lifts confinement. Particle confinement is taken to follow energy confinement."""
        args = (x["Ip"], x["R"], x["a"], x["kappa"], x["n20"], x["B"], x["M"], x["P_in"] / 1e6)
        return tau_98(*args) / tau_89(*args)

    def law(self, x, H, P):
        """Confinement time from the law of this state, s, at this power, W: IPB98(y,2) counts thermal energy, ITER89-P all of it."""
        return (tau_98 if H else tau_89)(x["Ip"], x["R"], x["a"], x["kappa"], x["n20"], x["B"], x["M"], max(P, 1e3) / 1e6)

    def no_wall_limit(self, x):
        """beta_N at the no-wall limit: 4 li(1) as before, or with limit "piccione" and R/a below 2 the closed form Piccione 2020 fitted to NSTX DCON runs (+-18%; docs/lit/29 and 40),
        in li(1), the aspect ratio and the pressure peaking F_p. Its aspect-ratio term goes negative near R/a 3, so above R/a 2 the conventional 4 li stays."""
        A = x["R"] / max(x["a"], 1e-3)
        if getattr(self, "limit", "4li") != "piccione" or A >= 2.0:
            return 4.0 * x["li"]
        cp, geo = x["cp"], x["geo"]
        p = np.asarray(cp.n_e.value) * np.asarray(cp.T_e.value) + np.asarray(cp.n_i.value) * np.asarray(cp.T_i.value)
        vpr = np.asarray(geo.vpr)
        Fp = float(p[0] / max(np.sum(p * vpr) / np.sum(vpr), 1e-12))
        li = x["li"]
        b_li = 4.91 * np.exp(-((li - 1.17) / 1.14) ** 2) + 0.21 * np.exp(-((li - 0.27) / 0.39) ** 2)
        b_A = -4.14 * A ** 3 + 13.47 * A ** 2 - 14.95 * A + 10.0
        b_Fp = 2.56 * np.exp(-((Fp - 4.06) / 2.59) ** 2) + 3.10 * np.exp(-((Fp - 0.43) / 4.29) ** 2)
        a0, aw = -0.12, ((3.0 * 0.3764, b_li), (1.2 * 0.3246, b_Fp), (1.0 * 0.2990, b_A))
        return float(((sum(w for w, _ in aw) - a0) / sum(w / max(b, 1e-3) ** 3 for w, b in aw)) ** (1 / 3))

    def edge_rule(self):
        """In L-mode no transport barrier outside the pedestal top (ntm.LMODE_EDGE); in H-mode TORAX's barrier. Only where the run asks for it (lmode_edge, H83)."""
        import ntm
        if getattr(self, "lmode_edge", False):
            ntm.LMODE_EDGE[0] = 0.0 if self.H else 1.0

    # ---------- each step
    def __call__(self, r):
        t = r.t
        dt = t - self.t_last
        self.t_last = t
        x = self.now()
        if getattr(self, "noncoronal", False):        # the impurities' residence times from the plasma now (ntm.set_residence, H84)
            import ntm
            ntm.set_residence(r)
        W = x["W_th"] + x["W_fast"]
        if self.s0 is None:
            self.s0 = dict(W=W, W_th=x["W_th"], n20=x["n20"], V=x["V"], n_G=x["Ip"] / (np.pi * x["a"] ** 2), F=self.confinement_ratio(x), density=self.density_programme(t) if self.density_programme else None, T_top=x["T_top"], n_top=x["n_top"])
            self.normalize = x["R"] / x["a"] < SURROGATE_ASPECT
            self.steps_top = 0
            self.s0["n_line"] = x["n20"]
            if getattr(self, "density_absolute", False):       # a programme in absolute units: the target is the programme itself, not its change from the start
                self.s0["density"] = x["n20"] * 1e20
            self.tau_E, self.P_net, self.W_prev = W / x["P_in"], x["P_in"], W
            self.t_hold = t + self.settle * self.tau_E     # a measured start relaxes for a moment, and that dW/dt is not power crossing the edge
        rec = r.records[-1] if r.records else {}
        # power, with the short filter every code needs on this test
        if dt > 0:
            a = 1 - np.exp(-dt / max(self.tau_E, 5 * dt))
            if W / x["P_in"] < 10.0 * self.tau_E:        # a transient is not confinement: in the step the pedestal jumps at the start of coupling, ITER's P_SOL reads -60 MW, P_in hits its floor, and that one step took tau_E from 2.7 to 12000 s, where its own filter then held it and P_net for the rest of the run
                self.tau_E += a * (W / x["P_in"] - self.tau_E)
            self.P_net += a * (x["P_in"] - (W - self.W_prev) / dt - self.P_net)
            self.W_prev = W
        # L-H
        P_LH = float(lh_general.P_LH(x["n20"], x["B"], lh_general.surface_area(x["R"], x["a"], x["kappa"]), x["R"], x["a"], x["Ip"], x["M"])) * 1e6
        if getattr(self, "lh_sigma", 0.0) > 0:          # the threshold drawn once per run from the published fit's scatter, as the beta limit is (H83): RMSE 0.307 in ln P for Martin 2008 (docs/lit/28)
            if getattr(self, "lh_factor", None) is None:
                self.lh_factor = float(np.exp(self.lh_sigma * np.random.default_rng(self.lh_seed).normal()))
            P_LH *= self.lh_factor
        if "hmode" in self.models and dt > 0:
            H = self.P_net > (HYSTERESIS * P_LH if self.H else P_LH) and t >= self.t_hold
            if self.oracle.get("H") is not None:       # the error budget: what was measured, in place of the model
                H = self.oracle["H"](t)
            if H != self.H:
                self.h_log.append((t, bool(H)))
                self.H = H
                self.edge_rule()
        tau_law = self.law(x, self.H, self.P_net)
        W_now = x["W_th"]
        W_law = tau_law * max(self.P_net, 0.1 * x["P_in"])
        ramp = 2.0 * tau_law
        if self.relative and self.law0 is None:
            self.law0 = W_law
        W_goal = W_law * (self.s0["W_th"] / max(self.law0, 1.0)) if self.relative else W_law      # the start's own confinement against the law stays, as a measured start gives it
        if self.normalize and self.table_start is not None and dt > 0:
            self.g = float(np.clip(self.g * np.exp(dt / tau_law * np.log(max(W_now, 1.0) / max(W_goal, 1.0))), 0.02, 50.0))
            table = self.table_start.copy()
            table[1] *= self.g
            table[0] *= self.g ** ELECTRON_SHARE
            self.h.table0 = table
        # the edge
        F = self.confinement_ratio(x)
        F_rel = 1.0 if self.feedback or not self.h_rise else (F if self.H else 1.0) / (self.s0["F"] if self.H0 else 1.0)      # a feedback loop holds the line density whatever the confinement; a set gas programme gets the rise
        programme = self.density_programme(t) / self.s0["density"] if self.density_programme else (x["Ip"] / (np.pi * x["a"] ** 2)) / self.s0["n_G"]
        fn_target = programme * F_rel * self.after_collapse
        line_target = self.s0["n_line"] * fn_target
        if self.oracle.get("line") is not None:
            line_target = self.s0["n_line"] * self.oracle["line"](t)
        fT_target = (W_law / (self.s0["W_th"] if self.H else self.s0["W"])) / max(fn_target * x["V"] / self.s0["V"], 1e-3)
        if self.relative and self.H == self.H0:
            fT_target = (W_law / self.law0) / max(fn_target * x["V"] / self.s0["V"], 1e-3)
        if self.edge_law in ("loop", "loop_cap"):      # no assumption about the shape: the edge moves until the thermal energy sits on the law
            fT_target = self.fT * max(W_goal, 1.0) / max(W_now, 1.0)
        lim = None
        if self.H and "eped" not in self.models:
            lim = self.pedestal_limit(x)
            if lim is not None and lim["p"] > 0:
                n_next = self.s0["n_top"] * (self.fn if self.follow and self.follow.get("n_top_ratio") else fn_target)
                T_lim = lim["p"] / (2 * QE * 1e3 * max(n_next, 1e17))     # n_i about n_e at the top
                if self.relative:
                    if self.T_lim_ref is None:
                        self.T_lim_ref = self.s0["T_top"] / max(T_lim, 1e-3) if self.H0 else 1.0
                    T_lim *= self.T_lim_ref
                f_lim = T_lim / max(self.s0["T_top"], 1e-3)
                if self.edge_law in ("law", "loop"):   # the law alone, with no cap
                    pass
                elif self.edge_law in ("cap", "loop_cap") or self.normalize:
                    fT_target = min(fT_target, f_lim)
                else:
                    fT_target = f_lim                  # where a surrogate holds the core, EPED-NN alone sets the height
                if not self.normalize and lim.get("width") is not None and dt > 0:   # and where the top sits
                    pn = np.asarray(x["ppo"].psi_norm)
                    new_top = float(np.clip(np.interp(1.0 - 2.0 * lim["width"], pn, np.asarray(x["geo"].rho_face_norm)), 0.7, 0.97))
                    if abs(new_top - float(r.get_param("pedestal.rho_norm_ped_top"))) > 0.005:
                        r.set_param("pedestal.rho_norm_ped_top", new_top)       # transport then runs up to here; TORAX's outer patch is off while a pedestal is set
        if dt > 0:
            a = 1 - np.exp(-dt / max(ramp, dt))
            self.fT += a * (float(np.clip(fT_target, 0.05, 40.0)) - self.fT)
            if getattr(self, "free_edge", False):      # fuelling by neutrals (H85): no density pin at the pedestal top (below, ped_extra), so the edge density comes from sources and transport; fn keeps its value for the record
                self.fn = float(np.clip(x["n_top"] / max(self.s0["n_top"], 1e15), 0.05, 40.0))
                if getattr(self, "sol", None) is not None:    # the separatrix from the scrape-off layer's own balance (sol.py, H86): its density, and with sol_T its temperature, are the core's boundary
                    if self.sol.n_u is None:
                        self.sol.n_u = float(r.get_param("profile_conditions.n_e_right_bc"))     # from the measured start, so the boundary moves, not jumps
                    st = self.sol(r)
                    r.set_param("profile_conditions.n_e_right_bc", float(st["sol_n_sep"]))
                    if getattr(self, "sol_T", False):
                        r.set_param("profile_conditions.T_e_right_bc", float(st["sol_T_sep"]) / 1e3)
                        r.set_param("profile_conditions.T_i_right_bc", float(st["sol_T_sep"]) / 1e3)
                    r.carry.update(st)
                if getattr(self, "sep_follow", False):  # tying the separatrix to the edge removes the boundary's drain, and the density runs away (H85 smoke test); off by default, the start's value held
                    if getattr(self, "n_sep0", None) is None:
                        self.n_sep0 = r.get_param("profile_conditions.n_e_right_bc")
                    r.set_param("profile_conditions.n_e_right_bc", self.n_sep0 * self.fn)
            elif self.follow:
                tau = self.follow["tau"] if "tau" in self.follow else self.follow["tau_share"] * tau_law
                err = float(np.log(line_target / x["n20"]))
                rate = err / max(tau, dt)
                if self.follow.get("integral"):         # integral action (H87): proportional control alone sits short of its target by about tau / tau_p, which the published particle
                    t_i = float(self.follow["integral"])    # transport's short particle confinement (10 to 20 ms) makes 25 to 50%; the integral supplies the steady outflux
                    self.follow_int = float(np.clip(getattr(self, "follow_int", 0.0) + err * dt / t_i, 0.0 if self.follow.get("no_sink") else -5.0, 5.0))
                    rate += self.follow_int / max(tau, dt)
                if self.follow.get("no_sink"):          # fuel only adds (H87): a gas system has no sink, and a sink shaped like the profile emptied the edge cells of an H-mode pedestal,
                    rate = max(rate, 0.0)               # whose particle diffusivity cannot refill them, until TORAX stopped (22786, 18464 in v17); density falls by transport alone
                if self.follow.get("n_top_ratio"):      # the pedestal from a published rule on the line density (TRANSP's n_ped = 0.71 n_line, docs/lit/28), reached over the pedestal's own time; if the core peaks more than that rule assumes, the edge comes down slowly to keep the line on its programme
                    self.trim = float(np.clip(self.trim * np.exp(dt / tau_law * np.log(line_target / x["n20"])), 0.5, 1.5))
                    self.fn += a * (float(np.clip(self.trim * self.follow["n_top_ratio"] * line_target * 1e20 / max(self.s0["n_top"], 1e15), 0.05, 40.0)) - self.fn)
                else:
                    self.fn = float(np.clip(line_target / self.s0["n_line"], 0.05, 40.0))      # the edge goes with the level; the source fills what is behind it
                self.particle_source(x, rate)
            else:
                self.fn = float(np.clip(self.fn * np.exp(self.fn_gain * dt / tau_law * np.log(line_target / x["n20"])), 0.05, 40.0))
        import ntm
        if getattr(self, "free_edge", False) and r.records and r.records[-1].get("elm") and getattr(r, "elms", None) is not None:
            # an ELM's particle loss with the density pin off (H86): the cycle's density share (ntm.ELMs.drop_n, or the law's) acts on the pin, and with no pin
            # the twin's ELMs took no particles out, so H-mode filled without limit in the H85 oracle runs. The same share now comes off n_e outside the pedestal top.
            drop_n = float(r.elm_drop[1]) if r.elms.law is not None else float(r.elms.drop_n)
            n_elms = int(round(float(r.records[-1]["elm"])))
            top = float(r.get_param("pedestal.rho_norm_ped_top"))
            r.edit_profile("n_e", (1.0 - drop_n) ** max(n_elms, 1), (top, 1.0))
            self.elm_particles = getattr(self, "elm_particles", 0) + max(n_elms, 1)
        if getattr(self, "neutrals", None) is not None:          # KN1D's ionization inside the separatrix, on top of the plant's sources (neutrals.py)
            self.neutrals(r)
        if getattr(self, "bolometer", None) is not None and dt > 0:     # the bolometers' total as the twin would give it (radiation.py, H86): a reading, the plasma is untouched
            self.bolometer(r, sol=getattr(self, "sol", None), neutrals=getattr(self, "neutrals", None))
        if getattr(self, "particle_law", False):
            ntm.PARTICLE_LAW[4] = float(self.H)                 # the pedestal's ratio and no turbulent pinch in H-mode
            r.carry.update(D_ratio=float(ntm.PARTICLE_LAW[1]), c_q=float(ntm.PARTICLE_LAW[3]))
        ped_ti = 1.0
        if self.oracle.get("top") is not None:
            self.fT, self.fn, ped_ti = self.oracle["top"](t)
        r.ped_extra = np.array([self.fT, 0.0 if getattr(self, "free_edge", False) else self.fn])     # a zero density target switches TORAX's density pin off (its adaptive source acts only where the target is non-zero)
        ntm.PEDESTAL_TI[0] = ped_ti
        if r.elms is None or r.elm_period is None:
            ntm.PEDESTAL[:] = r.ped_scale * r.ped_extra
        # the q = 2 surface, for the collapse and the seeds
        q = np.asarray(x["cp"].q_face)
        rf = np.asarray(x["geo"].rho_face_norm)
        cross = np.flatnonzero((q[:-1] < 2.0) & (q[1:] >= 2.0))
        rho_q2 = float(rf[cross[-1]]) if len(cross) else None
        # a crash or an ELM just now seeds an island
        if "modes" in self.models and self.h.modes and dt > 0:
            self.islands(r, rec, x, t, rho_q2)
        # the beta limit
        ratio = x["beta_N"] / max(self.no_wall_limit(x), 1e-3)
        if getattr(self, "porcelli", False) and "sawtooth" in self.models:       # Porcelli's thermal criteria set the trigger's critical shear (porcelli.py, H84)
            import porcelli
            pc = porcelli.critical_shear(r)
            if pc is not None:
                r.set_param("mhd.sawtooth.trigger_model.s_critical", float(pc["s_crit"]))
                r.carry["s_crit"] = float(pc["s_crit"])
        if "modes" in self.models and self.h.modes and self.pole:
            share = min(ratio / max(self.x_crit or 1.0, 1e-3), POLE_CAP)
            r.c.rdelta0 = -float(self.h.modes[0].m) * (1.0 - share / POLE_ZERO) / (1.0 - share)
        if "collapse" in self.models and dt > 0:
            if self.x_crit is None:
                self.x_crit = float(max(1.0 + LIMIT_SCATTER * self.rng.normal(), 0.5))
            if ratio > self.x_crit and rho_q2 is not None and t >= self.t_hold:
                self.collapse(r, t, rho_q2, ratio)
        below = np.flatnonzero(q <= 1.0)
        r.carry.update(H=float(self.H), fT=self.fT, fn=self.fn, P_net=self.P_net, P_LH_twin=P_LH, tau_E=self.tau_E, beta_kin=x["beta_N"], beta_over_limit=ratio, x_crit=self.x_crit or 0.0, g_law=self.g, T_top=x["T_top"], n_top=x["n_top"], line_target=line_target * 1e20, q0=float(q[0]), q_min=float(q.min()), rho_q1=float(rf[below[-1]]) if len(below) else 0.0, s_q1=float(np.asarray(x["cp"].s_face)[below[-1]]) if len(below) else 0.0, f0=self.f_now if hasattr(self, "f_now") else 0.0, Ip_now=x["Ip"] * 1e6)
        # the n = 1 internal mode as a reading (H86): ideal n = 1 internal kinks are unstable for q_min - 1 up to 0.12 in MAST's beam-heated plasmas (Chapman et al.,
        # Nucl. Fusion 50 (2010) 045007, CASTOR on MAST equilibria; seen only with beams), and below q = 1 the n = 1 activity of sawteeth and fishbones. A reading of the
        # twin's own q profile, as a magnetic probe would see a mode; what the mode does (rotation braking, fast-ion losses) is not modelled
        r.carry["kink_n1"] = float(float(q.min()) <= 1.0 + KINK_DQ and float(r.carry.get("beam_P_abs", 0.0) or 0.0) > 1e5)
        self.status = dict(P_LH=P_LH, P_net=self.P_net, H=self.H, ratio=ratio, x_crit=self.x_crit, collapses=len(self.collapses), seeds=len(self.seeds), tau_E=self.tau_E, eped=lim)

    def particle_source(self, x, rate):
        """Particles at rate (per s, as a share of the plasma's) added to what the plant already put on the table this step. shape "profile": in proportion to the density there, so the level moves and the shape stays with transport. "wall": from the edge, falling off over `length` of rho, as neutrals would."""
        import ntm
        rho = x["rho"]
        n_e = np.asarray(x["cp"].n_e.value)
        if self.follow.get("cap") is not None:
            rate = min(rate, self.follow["cap"])
        if self.follow.get("shape", "profile") == "profile":
            S = rate * n_e
        else:
            rate = max(rate, 0.0)              # a wall gives particles; it takes them only through transport, as the edge falls
            dv = np.asarray(x["geo"].vpr) * float(np.asarray(x["geo"].drho_norm).ravel()[0])
            shape = np.exp(-(1.0 - rho) / self.follow.get("length", 0.1))
            S = rate * float(np.sum(n_e * dv)) * shape / float(np.sum(shape * dv))
        ntm.EXTRA_PARTICLES[:] += np.interp(ntm.TURB_RHO, rho, S)

    def pedestal_limit(self, x):
        """EPED-NN's pressure for this plasma: the height, its width, and how far outside its training the plasma is."""
        import epednn
        if self.eped_model is None:
            self.eped_model = epednn.Model("EPED1NNensemble")
        geo = x["geo"]
        lower = float(np.asarray(getattr(geo, "delta_lower_face", geo.delta_face))[-1])
        upper = float(np.asarray(getattr(geo, "delta_upper_face", geo.delta_face))[-1])
        kappa = x["kappa"]
        free = self.h.free
        if not self.normalize and free is not None and getattr(free, "res", None) is not None:      # the network takes the shape of the separatrix, as FUSE gives it. TORAX's last surface stands at 99% of the flux: on SPARC kappa 1.86 and delta 0.43 there against 1.95 and 0.52, worth 20% in pressure
            import freeboundary
            shape = freeboundary.shape_of(free.res)
            kappa, lower, upper = shape["kappa"], shape["delta_lower"], shape["delta"]
        res = self.eped_model(a=x["a"], betan=x["beta_N"], bt=x["B"], delta=epednn.effective_delta(lower, upper), ip=max(x["Ip"], 0.05), kappa=kappa, m=x["M"], neped=max(x["n_top"] / 1e19, 0.1), r=x["R"], zeffped=x["Z_eff"], nesep_ratio=0.25, tesep=75.0)
        if self.ped_width and res.get("width") is None and res["p"] > 0:      # the ensemble gives the height alone: EPED1's width law, 0.076 beta_p,ped^0.5 in psi_N, is what EPED itself assumes
            B_p = MU0 * x["Ip"] * 1e6 / (2 * np.pi * x["a"] * np.sqrt(0.5 * (1 + kappa ** 2)))
            res["width"] = 0.5 * 0.076 * np.sqrt(2 * MU0 * res["p"] / max(B_p ** 2, 1e-6))
        if self.follow and self.follow.get("eped_scale"):      # a bracket: EPED's published scatter is about 20%
            res["p"] = res["p"] * self.follow["eped_scale"]
        return res

    # ---------- the ELM cycle: asked by the reactor for the next ELM
    def elm(self, r, t):
        if "elms" not in self.models or not self.H:
            return None
        x = self.now()
        W_plasma = x["W_th"] + x["W_fast"]
        share = 0.068 * max(x["T_top"], 0.02) ** 0.03 * max(x["n_top"] / 1e19, 0.05) ** -0.4 * x["kappa"] ** -0.4 * x["R"] ** 0.4
        dW = share * W_plasma
        P_SOL = max(float(x["ppo"].P_SOL_total), 1e4)
        interval = dW / (ELM_POWER_SHARE * P_SOL) * max(1.0 + 0.2 * float(self.rng.normal()), 0.2)
        W_ped = 3e3 * QE * x["n_top"] * x["T_top"] * x["V"]
        total = min(dW / max(W_ped, 1.0), 0.6)
        drop_n = min(ELM_PARTICLES, 0.5 * total)
        return max(interval, 2e-4), total - drop_n, drop_n

    # ---------- islands: the plasma's rotation and the seeds
    def islands(self, r, rec, x, t, rho_q2):
        md = self.h.modes[0]
        beam = self.h.plant.beam
        if md.rotation is not None:
            P_abs = (beam.last or {}).get("P_abs", 0.0) if beam is not None else 0.0
            if P_abs > 0:          # with no beam, the rotation stays as the scenario gives it: intrinsic rotation is not modelled
                n_mean = float(np.mean(np.asarray(x["cp"].n_i.value)))
                f = rotation_from_beam(P_abs, self.beam_settings.get("E_keV", 65.0), self.beam_settings.get("R_tan", x["R"]), self.beam_settings.get("A_b", 2.0), self.tau_E, n_mean, x["V"], x["R"], x["M"])
                self.f_now = f
                md.rotation.f0 = max(f, 1.5 * md.rotation.f_lock)
            md.rotation.tau_M0 = max(self.tau_E, 1e-4)
            md.rotation.quench_rate = 1e6 if r.widths[0] > LOCK_ENDS_AT * x["a"] else 0.0
        loc = r.locs[0]
        if loc is None:
            return
        import ntm
        w_d = ntm.small_island_width(loc)
        reach = (min(rec.get("rho_q1", 0.0), rho_q2) / rho_q2) ** 2 if rho_q2 else 0.0
        factor = SEED_CRASH * reach if rec.get("sawtooth") else SEED_ELM if rec.get("elm") else 0.0
        if self.seeding and factor and factor * w_d > r.widths[0]:
            self.h.apply(dict(cmd="island", m=md.m, n=md.n, w_cm=100 * factor * w_d))
            self.seeds.append(dict(t=t, w_cm=100 * factor * w_d, by="crash" if rec.get("sawtooth") else "ELM"))

    # ---------- the collapse
    def collapse(self, r, t, rho_q2, ratio):
        import ntm
        r.flatten_zone((0.0, rho_q2), density=True)
        n_left, T_left = 1.0 - COLLAPSE_PARTICLES, (1.0 - COLLAPSE_HEAT) / (1.0 - COLLAPSE_PARTICLES)
        for name, factor in (("n_e", n_left), ("n_i", n_left), ("T_e", T_left), ("T_i", T_left)):
            r.edit_profile(name, factor)
        if not self.feedback:
            self.after_collapse *= n_left
        self.collapses.append(dict(t=t, ratio=ratio, limit=self.x_crit, rho_q2=rho_q2, W_before=float(r.ppo.W_thermal_total)))
        r.carry["collapse"] = 1.0
        if self.H:
            self.h_log.append((t, False))
        self.H = False
        self.edge_rule()
        self.t_hold = t + 3.0 * self.tau_E
        self.x_crit = None
        if "modes" in self.models and self.h.modes and r.locs[0] is not None:
            w = SEED_CRASH * ntm.small_island_width(r.locs[0])
            if w > r.widths[0]:
                md = self.h.modes[0]
                self.h.apply(dict(cmd="island", m=md.m, n=md.n, w_cm=100 * w))
                self.seeds.append(dict(t=t, w_cm=100 * w, by="collapse"))


def attach(h, models=("hmode", "elms", "sawtooth", "modes", "collapse"), seed=0, H0=True, n_line=None, machine=None, follow=None):
    """The coupled plasma on a harness that is already running, on any machine: the harness's couple command. n_line: the machine's density programme, m^-3, a number or {t: value}; without one the run holds the Greenwald fraction it has now."""
    density = None
    if n_line is not None:
        if isinstance(n_line, dict):
            times = np.array(sorted(float(k) for k in n_line))
            values = np.array([n_line[k] for k in sorted(n_line, key=float)], float)
            density = lambda t: float(np.interp(t, times, values))
        else:
            density = lambda t: float(n_line)
    c = Coupled(h, density=density, H0=H0, seed=seed, models=models, beam=h.s["beams"] if isinstance(h.s.get("beams"), dict) else {}, tau_w=(machines.MACHINES.get(machine) or {}).get("tau_wall_s"))
    if h.reactor.state.geometry.a_minor and float(h.reactor.state.geometry.R_major) / float(h.reactor.state.geometry.a_minor) < SURROGATE_ASPECT:
        c.table_start = h.table0.copy()
    c.density_absolute = density is not None
    c.follow = follow
    if follow:
        c.ped_width = bool(follow.get("ped_width"))
    r = h.reactor
    if r.elms is not None and "elms" in models:
        r.elms.law = c.elm
    r.before_step.append(c)
    h.natural = c
    return c


def run_machine(path, until=None):
    """A machine with no shots, from a scenario file, against what is published for it."""
    import json
    import time

    import twin
    t0 = time.perf_counter()
    s = json.load(open(path))
    h = twin.Harness(s, keep_history=False)
    r = h.reactor
    rows = []
    t_print = 0.0
    until = until if until is not None else s.get("t_final", 1e9)
    while r.t < until and r.step() is not None:
        rec = r.records[-1]
        if r.t >= t_print:
            t_print = r.t + max(until / 20, 0.5)
            st = (getattr(h, "natural", None).status if getattr(h, "natural", None) else {}) or {}
            print(f"t {r.t:7.2f}  P_fus {rec.get('P_fusion', 0) / 1e6:7.1f} MW  Q {rec.get('Q_fusion', 0):5.2f}  W {rec.get('W_thermal', 0) / 1e6:6.1f} MJ  H98 {rec.get('H98', 0):4.2f}  beta_N {rec.get('beta_N', 0):4.2f}  n_line {rec.get('n_line', 0) / 1e20:5.2f}e20  T_e0 {rec.get('T_e0', 0):5.1f}  T_i0 {rec.get('T_i0', 0):5.1f}  q95 {rec.get('q95', 0):4.2f}  H {st.get('H', '-')}  P/P_LH {(st.get('P_net', 0) / max(st.get('P_LH', 1), 1)):5.2f}  beta/4li {st.get('ratio', 0) or 0:4.2f}  T_ped {r.get_param('pedestal.T_e_ped') * r.ped_extra[0]:5.2f}  elms {len(r.elm_times)}  wall {time.perf_counter() - t0:6.0f} s", flush=True)
        rows.append(dict(t=r.t, P_fusion=rec.get("P_fusion"), Q=rec.get("Q_fusion"), W=rec.get("W_thermal"), H98=rec.get("H98"), beta_N=rec.get("beta_N"), n_line=rec.get("n_line"), T_e0=rec.get("T_e0"), T_i0=rec.get("T_i0"), q95=rec.get("q95"), f_GW=rec.get("f_GW"), P_SOL=rec.get("P_SOL"), li3=rec.get("li3"), f_bs=rec.get("f_bs")))
    out = os.path.join(os.path.abspath(os.environ.get("XTORAX_HOME") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "runs", "twin", "machines")
    os.makedirs(out, exist_ok=True)
    json.dump(dict(scenario=path, sim_error=str(r.sim_error), rows=rows[::max(1, len(rows) // 400)], free=h.free.summary() if h.free is not None else None, wall_s=time.perf_counter() - t0), open(os.path.join(out, os.path.basename(path)), "w"), default=lambda v: np.asarray(v).tolist())
    print("ended", r.sim_error, f"{time.perf_counter() - t0:.0f} s", flush=True)
    if h.free is not None:
        print("shape", {k: round(v, 3) for k, v in h.free.summary()["shape"].items()}, "solves", h.free.summary()["solves"], "refused", h.free.summary()["rejected"], flush=True)
        import pickle
        pickle.dump(h.free.log, open(os.path.join(out, os.path.basename(path) + ".solves.pkl"), "wb"))
        h.free.close()


def oracle(shot, truth, k, names, natural, catalog):
    """The error budget (H52): what was measured put in place of one model at a time, to see how far each error falls. Not a mode for use: nothing here is known ahead of a real shot."""
    out = {}
    if "H" in names:
        windows = catalog["phases"]["hmode"]["windows"]
        out["H"] = lambda t: any(w["t0"] <= t <= w["t1"] for w in windows)
    if "line" in names:
        raw = np.load(os.path.join(natural.OUT, "scalars.npz"))
        tt, nel = raw[f"{shot}_t"], np.nan_to_num(raw[f"{shot}_nel"])
        n0 = max(float(np.interp(truth.t[k], tt, nel)), 1.0)
        out["line"] = lambda t: max(float(np.interp(t, tt, nel)), 0.1 * n0) / n0
    if "line_ts" in names:              # the same, from Thomson's fitted profile averaged over rho, the line density whole_shot.py scores (H86): the interferometer is broken or empty on 3 of the 25 shots
        tt = np.asarray(truth.t, float)
        nm = np.nanmean(np.asarray(truth.n_e, float), axis=1)
        good = np.isfinite(nm) & (nm > 0)
        tt, nm = tt[good], nm[good]
        n0 = max(float(np.interp(truth.t[k], tt, nm)), 1.0)
        out["line"] = lambda t: max(float(np.interp(t, tt, nm)), 0.1 * n0) / n0
    if "top" in names:
        rho = truth.rho
        at = lambda X, i: float(np.interp(0.91, rho, X[i]))
        idx = lambda t: int(np.clip(np.searchsorted(truth.t, t), 0, len(truth.t) - 1))
        mean = lambda X, a, b: float(np.mean([at(X, i) for i in range(max(a, 0), min(b, len(truth.t) - 1) + 1)]))
        T0, n0 = mean(truth.T_e, k - 4, k), mean(truth.n_e, k - 4, k)
        Ti0 = mean(truth.T_i, k - 4, k) if truth.T_i is not None else None

        def top(t):
            i = idx(t)
            fT = float(np.clip(mean(truth.T_e, i - 1, i + 1) / max(T0, 1e-3), 0.05, 40.0))
            fn = float(np.clip(mean(truth.n_e, i - 1, i + 1) / max(n0, 1e15), 0.05, 40.0))
            ti = float(np.clip(mean(truth.T_i, i - 1, i + 1) / max(Ti0, 1e-3) / fT, 0.1, 10.0)) if Ti0 is not None else 1.0
            return fT, fn, ti
        out["top"] = top
    return out


def beam_options(o, shot):
    """The run's beam options; species "measured" is the shot's own full, half and third energy split by power from FAIR-MAST level 1 (beam_species.py,
    runs/twin/beam_species.json; H85): machine hardware data, like the injected power."""
    import json
    import mast
    opts = dict(o.get("beam_options") or {})
    if opts.get("species") == "measured":
        path = os.path.join(mast.ROOT, "runs", "twin", "beam_species.json")
        table = json.load(open(path)) if os.path.exists(path) else {}
        mix = (table.get(str(shot)) or {}).get("mix")
        if mix:
            opts["species"] = [float(v) for v in mix]
        else:                             # no split measured for this shot: the split most shots share (SS at 89:8.5:2.3, H85)
            opts["species"] = [0.89, 0.08, 0.03]
    return opts


def build_shot(shot, models, seed, overrides=None, s_critical=0.25, mixing=1.3, k=None, keep=True):
    """A real shot's start and programme on the coupled plasma. Nothing here is fitted, on any shot. The hardware comes from machines.py."""
    import time

    import mast
    import mast_anchor
    import natural
    import ntm
    import twin
    o = overrides or {}
    machine = machines.MACHINES["MAST"]
    path = os.path.join(mast.ROOT, natural.SHOTS, f"{shot}.npz")
    truth, g, info = mast.load_shot(path)
    e = dict(np.load(os.path.join(natural.EXTRAS, f"{shot}.npz")))
    mast.START_LINE = True
    mast.EDGE_COVER = 0.9 if o.get("edge_start") else None     # the start's fits only from slices whose Thomson points reach rho 0.9 (H83)
    mast.EDGE_CAP = bool(o.get("edge_cap", False))            # and no temperature rising past a slice's last Thomson point (H86)
    if k is None and mast.EDGE_COVER is not None:      # and the first cut that has one among its trailing slices
        cov, n_tr = mast.edge_cover(g), mast.FIXED["trailing_slices"]
        k = next((i for i in range(n_tr - 1, len(truth.t)) if np.any(cov[i - n_tr + 1:i + 1] >= mast.EDGE_COVER)), n_tr - 1)
    k = mast.FIXED["trailing_slices"] - 1 if k is None else k
    t_cut = float(truth.t[k])
    x = natural.catalog_shot(shot)
    h0 = any(w["t0"] <= t_cut <= w["t1"] for w in x["phases"]["hmode"]["windows"])          # the state at the start is measured, as the profiles are
    holder = {}
    Ip_prog = np.interp(truth.t, e["t_prog"], np.nan_to_num(e["prog_i_plasma"]))
    if o.get("Ip_factor"):                                # past the shots (H67): the machine asked for more or less current from the cut on
        ramp = o["Ip_factor"]
        Ip_prog = Ip_prog * (1.0 + (ramp["f"] - 1.0) * np.clip((truth.t - t_cut - ramp["start"]) / ramp["ramp"], 0.0, 1.0))
    scale = float(np.mean(truth.Ip[max(k - 1, 0):k + 1])) / max(float(np.interp(t_cut, e["t_prog"], np.nan_to_num(e["prog_i_plasma"]))), 1e4)
    tied = o.get("tied", True)
    s = dict(name=f"MAST {shot}", mast=dict(shot=shot, model=o.get("base", "bgb"), past=0.0, natural=True, cut=t_cut, shots=natural.SHOTS, Ip=(Ip_prog * scale).tolist(), anchor_rates=o.get("anchor_rates"), anchor_bounds=dict(o.get("anchor_bounds") or {}, tied=tied), lmode_edge=bool(o.get("lmode_edge")) and not h0, noncoronal=bool(o.get("noncoronal"))), dt=1e-3, edge=False, rotation=False, impurities={"C": 0.0167, "Ne": 0.0, "Ar": 0.0}, physics=o.get("physics", {"a_bs": 1.5, "a_cd": 0.5}),
             gyrotrons={"units": 4, "unit_MW": 0.25, "mirror_rate": 2.0}, sensors={"sigma_w_cm": 0.0, "sigma_rho_cm": 0.0, "latency_steps": 0, "seed": 1}, decision={"kind": "none"}, events=[], modes=[], beams=dict(drive_current=o.get("beam_current", True), neutron_model=o.get("neutron_model", "births"), **({"history": dict(t=e["t_sum"], P_MW=np.nan_to_num(e["power_nbi"]) / 1e6)} if o.get("beam_history") and "power_nbi" in e else {}), **beam_options(o, shot)) if "beams" in models else False)
    s["elms"] = dict(law=lambda r, t: holder["c"].elm(r, t), seed=shot % 1000 + seed) if "elms" in models else False
    if "sawtooth" in models:
        s["sawtooth"] = dict(trigger_model=dict(model_name="simple", minimum_radius=0.15, s_critical=s_critical), redistribution_model=dict(model_name="simple", flattening_factor=1.001, mixing_radius_multiplier=mixing), crash_step_duration=1e-4)
    if "modes" in models:
        tau_w = machine["tau_wall_s"]
        tau_A = np.sqrt(MU0 * 2 * MP * float(np.mean(truth.n_e[k]))) * float(g["R0"][k]) / abs(float(g["B0"][k]))
        s["rotation"] = dict(f0=1e4, tau_M0=0.02, tau_A0=float(tau_A), tau_w=tau_w, w_vac=0.00625 * float(g["a"][k]), dt=1e-5, quench_rate=0.0, f_lock=1.0 / tau_w, **({"C_M": o["C_M"]} if "C_M" in o else {}))
        s["events"] = [dict(type="island", t=1e3, m=2, n=1, w_cm=1.0)]             # never reached: names the mode
    t0 = time.perf_counter()
    h = twin.Harness(s, keep_history=keep)
    r = h.reactor
    pr = np.nan_to_num(e["prog_n_e_line"]) if "prog_n_e_line" in e else None
    density = None if pr is None or np.interp(t_cut, e["t_prog"], pr) <= 0 else (lambda t: max(float(np.interp(t, e["t_prog"], pr)), 1e-3))
    c = Coupled(h, density=density, H0=h0, seed=100003 * seed + shot, models=models, tau_w=machine["tau_wall_s"], beam=mast_anchor.BEAMS, feedback=o.get("density_feedback", machine.get("density_feedback", True)))
    if o.get("normalize", True):
        c.table_start = h.table0.copy()
    c.pole = o.get("pole", True)
    c.oracle = oracle(shot, truth, k, o.get("oracle", ()), natural, x)
    c.fn_gain = o.get("fn_gain", 1.0)
    c.follow = o.get("follow")
    c.free_edge = o.get("fuelling") == "kn1d_gauge"
    if c.free_edge:                                            # fuelling by neutrals from the measured wall pressure, an oracle (neutrals.py, H85); the density is no longer steered
        import neutrals
        c.sol = None
        if o.get("sol") == "balance":                          # the scrape-off layer's particle and power balance (sol.py, H86)
            import sol as sol_lib
            c.sol = sol_lib.SOL(D_perp=o.get("sol_D", 1.0), L_par=o.get("sol_L"))
            c.sol_T = bool(o.get("sol_T", False))
        c.neutrals = neutrals.KN1DSource(shot, interval=o.get("kn1d_interval", 0.01), sol=c.sol)
        c.follow = None
    if o.get("bolometer"):                                     # the bolometers' total from a radiation budget (radiation.py, H86), recorded as rad_bolo and its terms
        import radiation
        c.bolometer = radiation.Bolometer(interval=o.get("bolometer_interval", 0.01))
    if o.get("particle_law") == "published":                   # published particle transport (ntm.PARTICLE_LAW, docs/lit/42, H85), drawn once per run from the published ranges, as the beta limit is
        import ntm
        rng_p = np.random.default_rng(104729 * seed + shot)
        ntm.PARTICLE_LAW[:] = (1.0, rng_p.uniform(0.4, 0.67), rng_p.uniform(0.05, 0.1), rng_p.uniform(0.0, 0.75), float(h0))
        c.particle_law = True
    c.h_rise = o.get("h_rise", True)
    c.relative = o.get("relative", False)
    c.lmode_edge = bool(o.get("lmode_edge"))
    c.noncoronal = bool(o.get("noncoronal"))
    c.limit = o.get("limit", "4li")
    c.porcelli = bool(o.get("porcelli"))
    c.lh_sigma, c.lh_seed = float(o.get("lh_scatter", 0.0)), 7919 * seed + shot
    c.edge_rule()
    holder["c"] = c
    r.before_step.append(c)
    r.retries = 3
    free = None
    if "free" in models:
        free = natural.FreeBoundary(h, g, shot, k, interval=0.01, Ip_prog=Ip_prog * scale)
        r.before_step.append(free)
    return dict(h=h, nat=c, free=free, truth=truth, info=info, k=k, t_cut=t_cut, h0=h0, rate0=None, t0=t0)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    run_machine(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else None)
