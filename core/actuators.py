"""Actuators and instabilities of the harness (H40, H41): named commands beyond the gyrotrons, each saying what level of physics it is.

Levels, on every entry of SPEC, which is also what the front end builds its controls from:
  torax        TORAX's own model does the work; the command only changes its inputs
  reduced      a published reduced equation, integrated here
  prescribed   an effect on the profiles with assumed size and timing: a way to ask "what if", not a prediction
  margin       a 0D model on top of TORAX's scalars, with no effect on the profiles until it ends the shot
Plant holds what they need between steps; Harness calls before_step() each step and snapshots state() with a branch.

Sources of the numbers, from search summaries unless a paper is named as read:
  ITER's disruption mitigation is shattered pellet injection (Lehnen, Workshop on Theory and Simulations of Disruptions, July 2021, read directly): 27 injectors (12 + 6 + 6 equatorial, 3 upper), pellets 28.5 mm by 57 mm, 1.6e24 neon or 1.9e24 hydrogen atoms; for thermal load mitigation
  more than 8e21 neon alone or more than 1e21 with hydrogen (3D fluid simulations; no radiated fraction on the slide); for current quench loads more than 2e21; for runaway impact 4e21 to 5e22. Speed a few hundred m/s (US ITER, ORNL; search summary). A 15 MA current quench should take 50 to 150 ms (search summary).
  Titanium is a laser blow-off tracer (HL-2A: Al, Ni, Ti, impurity confinement 30 to 60 ms; search summary), not an ITER actuator.
  Pellet ablation: the neutral gas shielding scaling of Parks and Turnbull (Phys. Fluids 1978), good to about a factor of 2; the coefficient 1.12e16 agrees with JOREK's code (models/pellet_module.f90: 4.12e16 in SI for deuterium, which is this in cgs), not yet checked against the paper itself. For neon and argon only the mass term changes here; real shielding differs.
  Current quench and runaway figures in disruption_card are textbook estimates: L / R with Spitzer resistivity, the Connor-Hastie field, and the avalanche gain exp(2.5 I[MA]) (Rosenbluth and Putvinski 1997, from memory).
"""
import dataclasses
import json

import jax.numpy as jnp
import numpy as np

import impurities as imp
import ntm

QE = ntm.QE
SOLID = dict(H=4.5e28, D=6.0e28, T=6.0e28, He=3.0e28, Ne=4.3e28, Ar=2.1e28, N=4.0e28, Kr=1.8e28, Xe=1.5e28, Li=4.6e28, Be=12.3e28, C=11.0e28, Ti=5.7e28, W=6.3e28)     # atoms per m^3 in the solid (noble gases and hydrogen frozen)
MASS = dict(H=1.0, D=2.0, T=3.0, **{k: v[1] for k, v in imp.ATOMS.items()})
ELECTRONS = dict(H=1.0, D=1.0, T=1.0, Ne=5.0, Ar=6.0)      # per atom in the cold cloud, roughly; only used for how fast a cloud cools its surroundings


def p(default, lo=None, hi=None, unit="", doc="", options=None):
    return {k: v for k, v in dict(default=default, min=lo, max=hi, unit=unit, doc=doc, options=options).items() if v not in (None, "")}


SPEC = dict(
    pellet=dict(kind="actuator", level="reduced", title="Fuelling pellet", doc="A frozen pellet, ablated along its path by the neutral gas shielding scaling. Adds cold particles and conserves energy. From the inboard side it reaches deeper. A pellet ending at the pedestal can set off an ELM (pacing).",
                params=dict(species=p("D", options=["D", "Ne", "Ar", "N", "Kr", "Xe", "Li", "C"]),
                            d_mm=p(None, 0.3, 10, "mm", "the injector's size: a cylinder with length equal to diameter, as injectors are given. With neither size, the machine's own injector (machines.py: MAST 1.1 mm, ITER 5 mm), else a 3 mm radius"),
                            r_mm=p(None, 0.2, 10, "mm", "or the radius of a sphere with the same atoms; it wins over d_mm"), v=p(300.0, 50, 1500, "m/s", "speed"), side=p("hfs", options=["hfs", "lfs"], doc="launched from the inboard (hfs) or outboard (lfs) side"), drift=p(0.1, 0, 0.3, "rho", "shift of the deposit towards the outboard side, by the ablation cloud's drift"))),
    spi=dict(kind="actuator", level="reduced", title="Shattered pellet (ITER's disruption mitigation gun)", doc="ITER's gun: 27 injectors, each pellet a 28.5 by 57 mm cylinder of frozen neon, hydrogen or a mix, shattered into a plume of shards that ablate as they cross the plasma. The impurity radiates the thermal energy away. Ends the shot in a mitigated quench if the plasma collapses.",
             params=dict(species=p("Ne", options=["Ne", "Ar", "D"]), atoms=p(5e22, 1e20, 2e24, "atoms", "of the species: ITER's table asks for more than 8e21 neon, at most 5e22 for runaway impact; a full neon pellet is 1.6e24"), hydrogen=p(0.0, 0.0, 2e24, "atoms", "deuterium in the same pellet, up to 1.9e24: it cools the plasma by dilution, and lowers the neon needed"), shards=p(200, 1, 3000, "", "number of shards, set by how the pellet shatters"), v=p(300.0, 100, 800, "m/s", "shard speed"), flight=p(0.02, 0, 0.1, "s", "from firing to reaching the plasma edge; assumed"), mix=p(100.0, 1, 1000, "", "transport multiplier once the cold front reaches the q = 2 surface: the MHD that ends a thermal quench, which a 1D code cannot produce; assumed"), mix_at=p(0.5, 0.05, 0.95, "", "the temperature at q = 2 over its starting value that sets off that mixing; assumed"))),
    gas=dict(kind="actuator", level="torax", title="Gas valve", doc="A puff at the edge until changed. Deuterium fuels; neon, argon, nitrogen or krypton cool the edge and core by radiation. Arrives after the valve's delay.",
             params=dict(species=p("D", options=["D", "Ne", "Ar", "N", "Kr", "Xe"]), rate=p(1e21, 0, 1e23, "atoms/s", "flow"), delay=p(0.1, 0, 2, "s", "from valve to plasma; assumed"), duration=p(0.0, 0, 1000, "s", "0: until changed"))),
    laser_blowoff=dict(kind="actuator", level="reduced", title="Laser blow-off tracer", doc="A burst of metal atoms at the edge, for measuring impurity transport. Titanium, aluminium, nickel, iron and molybdenum are followed as tracers with no effect on the plasma: TORAX has no radiation data for them.",
                       params=dict(species=p("Ti", options=["Ti", "Al", "Ni", "Fe", "Mo", "Ne", "Ar", "W"]), atoms=p(1e19, 1e16, 1e22, "atoms"))),
    nbi=dict(kind="actuator", level="prescribed", title="Neutral beams", doc="Beam heating and fuelling as a Gaussian, set until changed, on top of the scenario's heating. No beam physics: deposition is an input.",
             params=dict(P=p(0.0, 0, 60, "MW"), E_keV=p(1000.0, 30, 1500, "keV", "beam energy, for the fuelling rate"), rho0=p(0.2, 0, 0.8, "rho"), width=p(0.25, 0.05, 0.5, "rho"), f_e=p(0.7, 0, 1, "", "share to electrons"))),
    icrh=dict(kind="actuator", level="prescribed", title="Ion cyclotron heating", doc="Heating as a Gaussian, set until changed, on top of the scenario's heating.",
              params=dict(P=p(0.0, 0, 40, "MW"), rho0=p(0.1, 0, 0.8, "rho"), width=p(0.2, 0.05, 0.5, "rho"), f_i=p(0.6, 0, 1, "", "share to ions"))),
    ecrh=dict(kind="actuator", level="prescribed", title="Central electron heating", doc="Gyrotrons used for heating instead of current drive: takes their power away from island control. Central electron heating is how machines flush tungsten.",
              params=dict(P=p(0.0, 0, 20, "MW"), rho0=p(0.15, 0, 0.8, "rho"), width=p(0.1, 0.03, 0.4, "rho"))),
    rmp=dict(kind="actuator", level="prescribed", title="Resonant magnetic perturbation coils", doc="Level 0 to 1. Pumps density out of the pedestal (up to 20%), shrinks ELMs to none above level 0.7, brakes rotation and adds error field, so it can lock islands.",
             params=dict(level=p(0.0, 0, 1, "", "coil current over its maximum"))),
    ip=dict(kind="actuator", level="torax", title="Central solenoid: plasma current", doc="Ramp the plasma current to a target at a limited rate. TORAX's current diffusion follows.",
            params=dict(I_MA=p(10.5, 2, 17, "MA"), rate=p(0.5, 0.01, 5, "MA/s"))),
    density_control=dict(kind="loop", level="torax", title="Density feedback (gas)", doc="A slow loop on the gas valve to hold the line-averaged density. Off with target 0.",
                         params=dict(target=p(0.0, 0, 2, "1e20 m^-3"), gain=p(2e22, 0, 1e24, "atoms/s per 1e20"), rate_max=p(5e22, 0, 1e24, "atoms/s"))),
    radiation_control=dict(kind="loop", level="torax", title="Radiation feedback (seeding)", doc="A slow loop on a seeding valve to hold the exhaust power at a target. Off with target 0.",
                           params=dict(target_MW=p(0.0, 0, 200, "MW", "wanted P_SOL"), species=p("Ne", options=["Ne", "Ar", "N", "Kr"]), gain=p(1e19, 0, 1e21, "atoms/s per MW"), rate_max=p(1e22, 0, 1e24, "atoms/s"))),
    mitigation=dict(kind="loop", level="margin", title="Disruption mitigation trigger", doc="Arm the shattered pellet gun: fires when any island locks, or when an island passes a width, after a decision time. Off with armed 0.",
                    params=dict(armed=p(0, 0, 1), width_cm=p(0.0, 0, 40, "cm", "0: only on a lock"), delay=p(0.005, 0, 0.2, "s", "decision time"), species=p("Ne", options=["Ne", "Ar", "D"]), atoms=p(5e22, 1e20, 2e24, "atoms"), hydrogen=p(0.0, 0.0, 2e24, "atoms"))),
    fishbone=dict(kind="instability", level="prescribed", title="Fishbones", doc="Fast-ion driven bursts near q = 1: each burst expels fast ions from the core, so core alpha heating moves outward or is lost, and a share of bursts seed an island. Burst timing and size are inputs.",
                  params=dict(period=p(0.5, 0.01, 10, "s"), expelled=p(0.1, 0, 1, "", "share of core alpha heat moved, averaged over a period"), lost=p(0.2, 0, 1, "", "share of that lost to the wall"), rho=p(0.3, 0.05, 0.8, "rho", "edge of the core zone"), seed_prob=p(0.0, 0, 1, "", "chance a burst seeds a 2/1 island"), seed_cm=p(3.0, 0.5, 15, "cm"), duration=p(20.0, 0, 1000, "s"))),
    alfven=dict(kind="instability", level="prescribed", title="Alfven eigenmodes", doc="Fast-ion redistribution by a sea of Alfven modes: alpha heating spreads outward from a zone, some of it lost. Size is an input.",
                params=dict(redistributed=p(0.2, 0, 1, "", "share of the zone's alpha heat moved"), lost=p(0.1, 0, 1, "", "share of that lost"), rho=p(0.5, 0.1, 0.9, "rho", "edge of the zone"), duration=p(20.0, 0, 1000, "s"))),
    wall_flake=dict(kind="instability", level="reduced", title="Wall flake (a 'UFO')", doc="A flake of wall material falls in: a burst of impurity at the edge, then transport and the neoclassical pinch carry it. Tungsten can accumulate in the core and radiate it down.",
                    params=dict(species=p("W", options=["W", "C", "Ne", "Ar"]), atoms=p(1e19, 1e17, 1e22, "atoms"))),
    marfe=dict(kind="instability", level="prescribed", title="Edge radiation collapse (MARFE)", doc="A radiating zone at the edge takes a share of the exhaust power, and the pedestal falls with it. Near the density limit this is how disruptions start. Timing and size are inputs.",
               params=dict(share=p(0.5, 0, 1.5, "", "of P_SOL radiated in the zone"), pedestal=p(0.6, 0.1, 1, "", "pedestal scale it leaves"), duration=p(5.0, 0, 1000, "s"))),
    back_transition=dict(kind="instability", level="prescribed", title="H to L back transition", doc="The pedestal falls over a ramp, as when exhaust power drops below the threshold. Also happens by itself with limits on.",
                         params=dict(T=p(0.4, 0.1, 1, "", "pedestal temperature scale"), n=p(0.7, 0.1, 1, "", "pedestal density scale"), ramp=p(0.5, 0, 10, "s"), duration=p(0.0, 0, 1000, "s", "0: stays until changed"))),
    transport_barrier=dict(kind="instability", level="prescribed", title="Improved core confinement", doc="Turbulence reduced over a wide zone, as a barrier would. This is not a real narrow barrier, which TORAX's stiff transport model cannot hold: measured on the ITER hybrid, a factor 0.5 over rho 0.2 to 0.8 gives 7% more fusion, but 0.2 over rho 0.3 to 0.5 lowers the stored energy, since the model next door throws out what the zone holds in, and a fixed ceiling on the transport runs the core past 200 keV.",
                           params=dict(factor=p(0.5, 0.05, 1), rho0=p(0.2, 0, 0.9, "rho"), rho1=p(0.8, 0.05, 1, "rho"), duration=p(10.0, 0, 1000, "s"))),
    rwm=dict(kind="instability", level="margin", title="Resistive wall mode", doc="Grows on the wall's time above the no-wall beta limit, unless the plasma rotates fast enough or coils hold it. An amplitude only: ends the shot when large, with no effect on the profiles before.",
             params=dict(seed=p(1e-3, 1e-5, 1, "", "relative amplitude"), beta_N_limit=p(2.8, 1, 5), tau_wall=p(0.3, 0.01, 2, "s"), gain=p(0.0, 0, 100, "1/s", "coil feedback"))),
    vde=dict(kind="instability", level="margin", title="Vertical displacement", doc="The plasma drifts vertically at a growth rate, against a position loop with a limited speed. Past the wall limit the shot ends. TORAX's geometry is fixed, so nothing else moves until then.",
             params=dict(kick_cm=p(5.0, 0, 100, "cm"), growth=p(5.0, 0.1, 500, "1/s", "assumed"), gain=p(20.0, 0, 1000, "1/s", "position loop"), v_max=p(1.0, 0, 20, "m/s", "coil limit"), z_max_cm=p(50.0, 5, 200, "cm"))),
    limits=dict(kind="loop", level="margin", title="Operating limits", doc="Watch the Greenwald density, the exhaust power against the L-H threshold and the radiated power. On: past the Greenwald limit a MARFE starts; below the threshold the plasma falls back to L mode; radiating more than the heating ends the shot.",
                params=dict(on=p(0, 0, 1), greenwald=p(1.2, 0.5, 2, "", "line density over the limit"), hl_hold=p(1.0, 0, 20, "s", "time below the threshold before falling back"))),
)


def gaussian(x, x0, w):
    return np.exp(-0.5 * ((x - x0) / w) ** 2)


def spread(rho, atoms, w):
    kernel = gaussian(rho[:, None], rho[None, :], w)
    out = kernel @ atoms / kernel.sum(0).mean()
    return out * atoms.sum() / max(out.sum(), 1e-30)


def ablation_rate(ne, Te_keV, r_m, mass):
    """Atoms per second from a sphere of radius r_m. Neutral gas shielding (Parks and Turnbull 1978); ne in m^-3."""
    return 1.12e16 * (np.maximum(ne, 1e12) * 1e-6) ** 0.333 * (np.maximum(Te_keV, 1e-3) * 1e3) ** 1.64 * (r_m * 100) ** 1.33 * mass ** (-0.333)


def cooling_table(species, ne_tau=1e16):
    """Line radiation L_Z(T_e) in W m^3 on a grid of log T in keV, for a fast quench: Mavrin's non-coronal fits (2017, in TORAX) where they exist, at n_e tau = 1e16 m^-3 s (1e20 m^-3 for 0.1 ms, how long an ion has before heat and impurities move on; assumed), since ions have no time to reach coronal balance; coronal elsewhere."""
    from torax._src.physics.radiation import mavrin_noncoronal_cooling_rate as noncoronal
    from torax._src.physics.radiation import radiation
    T = np.geomspace(1e-3, 100.0, 200)
    L = np.array(radiation.calculate_cooling_rate(jnp.asarray(T), species))
    if species in noncoronal.COEFFS:
        fit = (T >= noncoronal.MIN_TEMPERATURES[species]) & (T <= noncoronal.MAX_TEMPERATURES[species])
        L[fit] = np.asarray(radiation.calculate_mavrin_cooling_rate(jnp.asarray(T[fit]), species, radiation.MavrinModelType.NONCORONAL, ne_tau))
    return np.log(T), L


def quench_0d(W, V, n_e, n_z, tables, tau=3e-3, power=0.5):
    """From the plasma at the start of a thermal quench, well mixed: dW/dt = -P_rad - W / tau(T), until 1% of W is left. P_rad = n_e sum(n_z L_z(T)) V, with T from W. tau is the conduction time set by the stochastic field: tau at the start (assumed: ITER's quench
    is expected to take 1 to 3 ms) and growing as T^-power as the plasma cools, since heat crosses a stochastic field at the thermal speed (Rechester and Rosenbluth 1978). Returns radiated and conducted energy, in J, and the time, in s.
    A 1D solver cannot follow this: densities reach 30 times normal and the field is 3D."""
    E_rad, E_cond, t, W0 = 0.0, 0.0, 0.0, W
    T0 = W / (3 * n_e * V * 1e3 * QE)
    while W > 0.01 * W0 and t < 0.2:
        T = max(W / (3 * n_e * V * 1e3 * QE), 1e-3)
        P = n_e * V * sum(nz * np.interp(np.log(T), tables[s][0], tables[s][1]) for s, nz in n_z.items())
        C = W / (tau * (T0 / T) ** power)
        h = min(0.02 * W / max(P + C, 1.0), 1e-4)
        E_rad += h * P
        E_cond += h * C
        W -= h * (P + C)
        t += h
    return E_rad, E_cond, t


def fly(x, r, v, dt, chord, rho, ne, Te, mass, solid, dv, count=1.0, electrons=1.0, h=2e-5, radiator=None, n_z=None):
    """Shards moving along a chord: x (m from the plasma's edge), radius r (m), count shards per group, for dt. Returns new x, r and the atoms released per cell of rho, per group. What they release dilutes and cools the plasma around them
    as they go (energy kept, spread over a few cells, as a flux surface fills in much less than a step), and a radiator cools them further, over microseconds at these densities: that is what limits a large pellet, since a cloud stops ablating once it has
    cooled its surroundings. radiator: (log T grid, L_Z, share of the species in the atoms); n_z: the species' density there already."""
    released = np.zeros((len(x), len(rho)))
    x, r, ne, Te = x.copy(), r.copy(), ne.copy(), Te.copy()
    nz = None if radiator is None else n_z.copy()
    steps = max(int(np.ceil(dt / h)), 1)
    h = dt / steps
    kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    for _ in range(steps):
        inside = (x >= 0) & (x <= chord["x"][-1]) & (r > 0)
        if inside.any():
            q = np.interp(x[inside], chord["x"], chord["rho"])
            G = ablation_rate(np.interp(q, rho, ne), np.interp(q, rho, Te), r[inside], mass)
            new_r = np.maximum(r[inside] ** 3 - 3 * G * h / (4 * np.pi * solid), 0.0) ** (1 / 3)
            atoms = 4 / 3 * np.pi * solid * (r[inside] ** 3 - new_r ** 3)
            r[inside] = new_r
            cell = np.clip(np.searchsorted(rho, q), 0, len(rho) - 1)
            released[np.where(inside)[0], cell] += atoms
            added = np.zeros_like(rho)
            np.add.at(added, cell, atoms * count * electrons)
            dn = np.convolve(added, kernel, mode="same") / dv
            Te *= ne / (ne + dn)
            ne += dn
            if radiator is not None:
                nz += radiator[2] * dn / electrons
                touched = nz > 0
                L = np.interp(np.log(np.maximum(Te[touched], 1e-3)), radiator[0], radiator[1])
                Te[touched] /= 1 + h * (2 / 3) * nz[touched] * L / (1e3 * QE * Te[touched])
        x += v * h
    return x, r, released


def machine_of(h):
    """The machine file a harness's plasma belongs to: a MAST shot, a scenario started from a machine file, or a plasma with a machine's major and minor radius (TORAX's ITER hybrid). None if no file fits."""
    import machines
    s = getattr(h, "s", None) or {}
    if s.get("mast"):
        return "MAST"
    name = (s.get("machine_start") or {}).get("machine")
    if name in machines.MACHINES:
        return name
    geo = h.reactor.state.geometry
    R0, a = float(np.asarray(geo.R_major)), float(np.asarray(geo.a_minor))
    for name, m in machines.MACHINES.items():
        if abs(R0 - m["R0"]) < 0.05 * m["R0"] and abs(a - m["a"]) < 0.1 * m["a"]:
            return name
    return None


def spec_for(h):
    """SPEC with the pellet's default size set to the machine's own injector, for a front end that shows defaults."""
    import copy
    import machines
    out = copy.deepcopy(SPEC)
    hw = machines.MACHINES.get(machine_of(h) or "", {}).get("pellets") if h is not None else None
    if hw:
        out["pellet"]["params"]["d_mm"]["default"] = hw["default_d_mm"]
    else:
        out["pellet"]["params"]["r_mm"]["default"] = 3.0
    return out


def chord_of(geo, side):
    """Along the midplane: distance from the edge against rho_norm. Outboard: from R_out(1) inward; inboard: from R_in(1) outward."""
    rho, R_in, R_out = np.asarray(geo.rho_norm), np.asarray(geo.R_in), np.asarray(geo.R_out)
    if side == "lfs":
        out = (R_out[-1] - R_out)[::-1]
        back = (R_out[-1] - R_in)
    else:
        out = (R_in - R_in[-1])[::-1]
        back = (R_out - R_in[-1])
    return dict(x=np.concatenate([out, back]), rho=np.concatenate([rho[::-1], rho]))


@dataclasses.dataclass
class Effect:
    name: str
    until: float
    c: dict
    extra: dict = dataclasses.field(default_factory=dict)


class Plant:
    def __init__(self, h, background=None):
        self.h, self.r = h, h.reactor
        self.imp = imp.Impurities(self.r, background) if background is not None else None
        self.effects, self.shards, self.tables = [], [], {}
        self.beam = None
        self.heat = {}             # name: (electrons, ions) on ntm.TURB_RHO, W/m^3
        self.parts = {}            # name: electron source on ntm.TURB_RHO, per m^3 per s
        self.fuel_atoms, self.fuel_extra, self.tau_p, self.eta_gas = 0.0, 0.0, h.s.get("tau_p", 3.0), h.s.get("eta_gas", 0.1)
        self.settings = dict(density_control=dict(target=0.0), radiation_control=dict(target_MW=0.0), mitigation=dict(armed=0), limits=dict(on=0), rmp=dict(level=0.0), ip=None)
        self.below_since = None
        self.gas_rate = 0.0
        self.Ip0 = self.r.get_param("profile_conditions.Ip")
        self.rng = np.random.default_rng(h.s.get("plant_seed", 0))
        self.fired = False
        self.fast_until, self.fast_dt = -1.0, 2e-4
        self.event_log = []
        self.base_elm = dataclasses.replace(self.r.elms) if self.r.elms else None
        self.saved_units = None
        self.card = None
        self.z_cm = 0.0
        self.A_rwm = 0.0
        self.tq_start = None

    # commands
    def log(self, what, **kw):
        self.event_log.append(dict(t=self.r.t, what=what, **kw))

    def command(self, c):
        name = c["cmd"]
        spec = SPEC[name]
        v = {k: c.get(k, q.get("default")) for k, q in spec["params"].items()}
        out = getattr(self, "do_" + name)(v, c)
        self.log(name, **{k: x for k, x in v.items()})
        return dict(ok=True, cmd=name, t=self.r.t, level=spec["level"], **(out or {}))

    def _need(self, species):
        if self.imp is None:
            raise ValueError('this scenario has no live impurities: add "impurities" to it, for example {"Ne": 0.01, "W": 1e-5, "Ar": 0.0}')
        if species not in self.imp.species:
            raise ValueError(f"{species} is not in the scenario's impurities or the tracers: {sorted(self.imp.species)}")

    def add_electrons(self, dn):
        """Cold electrons, on the cell grid: density up, energy kept. Ions cool with them when they are fuel."""
        r = self.r
        cp = r.state.core_profiles
        n0 = np.asarray(cp.n_e.value)
        f = n0 / (n0 + dn)
        new = dict(n_e=dataclasses.replace(cp.n_e, value=jnp.asarray(n0 + dn, dtype=cp.n_e.value.dtype)),
                   T_e=dataclasses.replace(cp.T_e, value=jnp.asarray(np.asarray(cp.T_e.value) * f, dtype=cp.T_e.value.dtype)),
                   T_i=dataclasses.replace(cp.T_i, value=jnp.asarray(np.asarray(cp.T_i.value) * f, dtype=cp.T_i.value.dtype)))
        r.state = dataclasses.replace(r.state, core_profiles=dataclasses.replace(cp, **new))

    def release(self, species, fraction, atoms_cells):
        """Atoms per cell, of a pellet with this share of species and the rest deuterium, into the plasma."""
        dv = self.imp.cell_volumes() if self.imp else np.asarray(self.r.state.geometry.vpr) * float(np.asarray(self.r.state.geometry.drho_norm).ravel()[0])
        if species == "D":
            self.add_electrons(atoms_cells / dv)
            return
        self._need(species)
        dn = fraction * atoms_cells / dv
        self.imp.excess[species] = self.imp.excess[species] + dn
        self.imp.changed = True
        self.add_electrons(dn * self.imp.charge(species) + (1 - fraction) * atoms_cells / dv)

    def do_pellet(self, v, c):
        r, geo, cp = self.r, self.r.state.geometry, self.r.state.core_profiles
        rho = np.asarray(geo.rho_norm)
        chord = chord_of(geo, v["side"])
        rm, size_from = self.pellet_radius(v)
        sp = v["species"]
        x, rr, rel = fly(np.zeros(1), np.array([rm]), v["v"], chord["x"][-1] / v["v"] * 1.01, chord, rho, np.asarray(cp.n_e.value), np.asarray(cp.T_e.value), MASS[sp], SOLID[sp], self._dv(), 1.0, ELECTRONS.get(sp, 8.0))
        atoms = rel[0]
        shift = -v["drift"] if v["side"] == "hfs" else v["drift"]        # the cloud drifts to the outboard side: deeper from the inboard side, shallower from the outboard side
        moved = np.interp(rho, rho + shift, atoms, left=0.0, right=0.0)
        moved *= atoms.sum() / max(moved.sum(), 1e-30)
        moved = spread(rho, moved, 0.03)                  # the cloud expands along and across the field before it settles
        total = float(moved.sum())
        if sp == "D":                                     # TORAX pins the pedestal top, and a spike just inside it would drain into the pin within a step, taking the core's density with it (measured). So what lands in or near the pedestal counts through the pedestal's particle balance
            outside = rho > r.get_param("pedestal.rho_norm_ped_top") - 0.15
            self.fuel_atoms += float(moved[outside].sum())
            moved = moved * ~outside
        self.release(sp, 1.0, moved)
        peak = float((moved * rho).sum() / max(moved.sum(), 1e-30))
        if self.r.elms and total > 0 and float(moved.sum()) < 0.5 * total:      # most of it landed at the pedestal: it can set off an ELM (pacing)
            r.elm_period = 0.0
        pellet_atoms = 4.0 / 3.0 * np.pi * rm ** 3 * SOLID[sp]
        electrons = float(np.sum(np.asarray(cp.n_e.value) * self._dv()))
        out = dict(atoms=total, atoms_in_pedestal_balance=total - float(moved.sum()), mean_rho_inside=peak, burnt_through=bool(rr[0] <= 0),
                   radius_mm=1e3 * rm, size_from=size_from, pellet_atoms=pellet_atoms, plasma_electrons=electrons, share_of_plasma=pellet_atoms / max(electrons, 1.0))
        if pellet_atoms > electrons:
            out["warning"] = f"this pellet holds {pellet_atoms / max(electrons, 1.0):.1f} times the plasma's electrons"
        return out

    def pellet_radius(self, v):
        """The radius of a sphere with the pellet's atoms, in m, and where its size came from: r_mm, else d_mm (a cylinder with length equal to diameter, of the same
        volume), else the machine's own injector (machines.py), else 3 mm."""
        import machines
        given = lambda x: x is not None and np.isfinite(float(x)) and float(x) > 0
        if given(v.get("r_mm")):
            return float(v["r_mm"]) / 1e3, "r_mm"
        if given(v.get("d_mm")):
            return (3.0 / 16.0) ** (1.0 / 3.0) * float(v["d_mm"]) / 1e3, "d_mm"
        name = machine_of(self.h)
        hw = machines.MACHINES.get(name or "", {}).get("pellets")
        if hw:
            return (3.0 / 16.0) ** (1.0 / 3.0) * hw["default_d_mm"] / 1e3, f"{name}'s injector, {hw['default_d_mm']} mm"
        return 3e-3, "no injector in the machine file: a 3 mm radius"

    def do_spi(self, v, c):
        sp = v["species"]
        total = v["atoms"] + v["hydrogen"]
        f = v["atoms"] / total if sp != "D" else 0.0
        solid = f * SOLID[sp] + (1 - f) * SOLID["D"]
        n = int(v["shards"])
        groups = 7
        rs = (3 * total / (4 * np.pi * solid * n)) ** (1 / 3)
        speed = v["v"] * np.linspace(0.7, 1.3, groups)
        self.shards.append(dict(t_last=self.r.t, species=sp, fraction=f, x=-speed * v["flight"], r=np.full(groups, rs), v=speed, count=n / groups, mass=f * MASS[sp] + (1 - f) * MASS["D"], solid=f * SOLID[sp] + (1 - f) * SOLID["D"], chord_side="lfs"))
        if self.tq_start is None:
            self.tq_start = dict(t=self.r.t, W=float(self.r.ppo.W_thermal_total), E_rad=0.0, mix=v["mix"], mix_at=v["mix_at"], mixing=False, pin_off=False)
        self.fast_until = max(self.fast_until, self.r.t + v["flight"] + 0.2)
        return dict(atoms_total=total, shard_radius_mm=1e3 * rs)

    def _radiator(self, s):
        if s["species"] == "D" or s["fraction"] <= 0:
            return None
        table = self.tables.setdefault(s["species"], cooling_table(s["species"]))
        return table[0], table[1], s["fraction"]

    def do_gas(self, v, c):
        sp = v["species"]
        if sp != "D":
            self._need(sp)
        self.h.script = sorted(self.h.script + [dict(cmd="_valve", species=sp, rate=v["rate"], t=self.r.t + v["delay"])] + ([dict(cmd="_valve", species=sp, rate=0.0, t=self.r.t + v["delay"] + v["duration"])] if v["duration"] > 0 else []), key=lambda x: x["t"])

    def valve(self, c):
        sp, rate = c["species"], c["rate"]
        rho = np.asarray(self.r.state.geometry.rho_norm)
        shape = np.exp((rho - 1.0) / 0.03)
        if sp == "D":
            self.gas_rate = rate
        else:
            self.imp.steady[sp] = rate * shape / max((shape * self._dv()).sum(), 1e-30)
            self.imp.changed = True

    def _dv(self):
        return np.asarray(self.r.state.geometry.vpr) * float(np.asarray(self.r.state.geometry.drho_norm).ravel()[0])

    def do_laser_blowoff(self, v, c):
        self._need(v["species"])
        rho = np.asarray(self.r.state.geometry.rho_norm)
        self.imp.deposit(v["species"], v["atoms"], gaussian(rho, 0.97, 0.02))

    def _gaussian_heat(self, P_MW, x0, w, f_e):
        """A heat source of this total power, split between electrons and ions, on the table's knots, normalized over the cells."""
        rho = np.asarray(self.r.state.geometry.rho_norm)
        total = (gaussian(rho, x0, w) * self._dv()).sum()
        shape = gaussian(ntm.TURB_RHO, x0, w) * P_MW * 1e6 / max(total, 1e-30)
        return f_e * shape, (1 - f_e) * shape

    def do_nbi(self, v, c):
        self.heat["nbi"] = self._gaussian_heat(v["P"], v["rho0"], v["width"], v["f_e"])
        rho = np.asarray(self.r.state.geometry.rho_norm)
        rate = v["P"] * 1e6 / (v["E_keV"] * 1e3 * QE)
        self.parts["nbi"] = rate * gaussian(ntm.TURB_RHO, v["rho0"], v["width"]) / max((gaussian(rho, v["rho0"], v["width"]) * self._dv()).sum(), 1e-30)

    def do_icrh(self, v, c):
        self.heat["icrh"] = self._gaussian_heat(v["P"], v["rho0"], v["width"], 1 - v["f_i"])

    def do_ecrh(self, v, c):
        g = self.h.chain
        if self.saved_units is None:
            self.saved_units = g.units
        g.units = max(self.saved_units - int(np.ceil(v["P"] / g.unit_MW)), 0)
        self.heat["ecrh"] = self._gaussian_heat(v["P"], v["rho0"], v["width"], 1.0)
        return dict(units_left_for_islands=g.units)

    def do_rmp(self, v, c):
        self.settings["rmp"] = dict(v)
        self.apply_rmp()

    def apply_rmp(self):
        lv = self.settings["rmp"]["level"]
        r = self.r
        if r.elms and self.base_elm:
            f = max(1 - lv / 0.7, 0.0)
            r.elms.drop_T, r.elms.drop_n = self.base_elm.drop_T * f, self.base_elm.drop_n * f
        for md in self.h.modes:
            rot = md.rotation
            if rot is not None:
                if "f0" not in md.__dict__.setdefault("_base", {}):
                    md._base.update(f0=rot.f0, w_vac=rot.w_vac)
                rot.f0, rot.w_vac = md._base["f0"] * (1 - 0.3 * lv), md._base["w_vac"] * (1 + lv)

    def do_ip(self, v, c):
        self.settings["ip"] = dict(target=v["I_MA"] * 1e6, rate=v["rate"] * 1e6, now=self.r.get_param("profile_conditions.Ip"))

    def do_density_control(self, v, c):
        self.settings["density_control"] = dict(v, on_rate=0.0)
        if v["target"] == 0:
            self.parts.pop("density_control", None)

    def do_radiation_control(self, v, c):
        self.settings["radiation_control"] = dict(v, on_rate=0.0)
        self._need(v["species"])
        if v["target_MW"] == 0:
            self.imp.steady[v["species"]] = np.zeros_like(self.imp.steady[v["species"]])

    def do_mitigation(self, v, c):
        self.settings["mitigation"] = dict(v)
        self.fired = False

    def do_fishbone(self, v, c):
        self.effects.append(Effect("fishbone", self.r.t + v["duration"], v, dict(next=self.r.t + v["period"])))

    def do_alfven(self, v, c):
        self.effects.append(Effect("alfven", self.r.t + v["duration"], v))

    def do_wall_flake(self, v, c):
        self._need(v["species"])
        rho = np.asarray(self.r.state.geometry.rho_norm)
        self.imp.deposit(v["species"], v["atoms"], gaussian(rho, 0.95, 0.03))

    def do_marfe(self, v, c):
        self.effects.append(Effect("marfe", self.r.t + v["duration"], v, dict(ped=self.r.ped_scale.copy())))
        self.r.ped_scale = self.r.ped_scale * np.array([v["pedestal"], 1.0])

    def do_back_transition(self, v, c):
        self.effects.append(Effect("back_transition", self.r.t + (v["duration"] if v["duration"] > 0 else 1e12), v, dict(t0=self.r.t, ped=self.r.ped_scale.copy())))

    def do_transport_barrier(self, v, c):
        self.h.apply(dict(cmd="turbulence", factor=v["factor"], rho=[v["rho0"], v["rho1"]], duration=v["duration"], channel="heat"))

    def do_rwm(self, v, c):
        self.A_rwm = max(self.A_rwm, v["seed"])
        self.settings["rwm"] = dict(v)

    def do_vde(self, v, c):
        self.z_cm += v["kick_cm"]
        self.settings["vde"] = dict(v)

    def do_limits(self, v, c):
        self.settings["limits"] = dict(v)

    # each step
    def before_step(self):
        r, h = self.r, self.h
        t = r.t
        dt = r.records[-1]["t"] - r.records[-2]["t"] if len(r.records) > 1 else h.dt
        rec = r.records[-1]
        heat_e, heat_i, parts = np.zeros_like(ntm.TURB_RHO), np.zeros_like(ntm.TURB_RHO), np.zeros_like(ntm.TURB_RHO)
        self._shards(dt)
        if self.imp is not None:
            self.imp.advance()
        self._loops(rec, dt)
        if self.tq_start is not None:                    # a floor of 15 eV: TORAX stops below 5 eV, and the cold edge layer holds next to no energy
            cp = r.state.core_profiles
            new = {T: dataclasses.replace(getattr(cp, T), value=jnp.maximum(getattr(cp, T).value, 0.015)) for T in ("T_e", "T_i")}
            r.state = dataclasses.replace(r.state, core_profiles=dataclasses.replace(cp, **new))
        for e in list(self.effects):
            if t >= e.until:
                self._end(e)
                self.effects.remove(e)
                continue
            q = self._effect(e, t, dt)
            if q is not None:
                heat_e, heat_i = heat_e + q[0], heat_i + q[1]
        if self.beam is not None:
            (b_e, b_i), fuel, last = self.beam.before_step()
            self.heat["beam"], self.parts["beam"] = (b_e, b_i), fuel
            r.carry.update(beam_W_fast=last["W_fast"], beam_neutrons=last["neutrons"], beam_through=last["through"], beam_P_abs=last["P_abs"])
        for a, b in self.heat.values():
            heat_e, heat_i = heat_e + a, heat_i + b
        for x in self.parts.values():
            parts = parts + x
        ntm.EXTRA_HEAT[0], ntm.EXTRA_HEAT[1], ntm.EXTRA_PARTICLES[:] = heat_e, heat_i, parts
        r.max_dt = self.fast_dt if t < self.fast_until else np.inf

    def _shards(self, dt):
        r = self.r
        self._quench(dt)
        if not self.shards:
            return
        st = r.state
        geo, cp = st.geometry, st.core_profiles
        rho = np.asarray(geo.rho_norm)
        for s in self.shards:
            chord = chord_of(geo, s["chord_side"])
            elapsed = r.t - s["t_last"]
            s["t_last"] = r.t
            if elapsed <= 0:
                continue
            s["x"], s["r"], rel = fly(s["x"], s["r"], s["v"], elapsed, chord, rho, np.asarray(cp.n_e.value), np.asarray(cp.T_e.value), s["mass"], s["solid"], self._dv(), s["count"], s["fraction"] * ELECTRONS.get(s["species"], 8.0) + 1 - s["fraction"], radiator=self._radiator(s), n_z=self.imp.excess[s["species"]] if s["species"] != "D" else None)
            atoms = spread(rho, rel.sum(0) * s["count"], 0.02)
            if atoms.sum() > 0:
                self.release(s["species"], s["fraction"], atoms)
                s.setdefault("released", 0.0)
                s["released"] += float(atoms.sum())
        self.shards = [s for s in self.shards if np.any((s["x"] < chord_of(geo, s["chord_side"])["x"][-1]) & (s["r"] > 0))]

    def _quench(self, dt):
        """A mitigated thermal quench, step by step: radiation at the edge, a cold front moving in; TORAX's pedestal pin must go, since nothing holds a pedestal in a quench. When the front reaches the q = 2 surface, transport across the plasma goes up by mix: 1.5D modelling of material
        injection does this because the mixing itself is 3D MHD (the same assumption as in ASTRA-STRAHL studies of ASDEX Upgrade, from memory)."""
        q, r = self.tq_start, self.r
        if q is None or r.disrupted:
            return
        q["E_rad"] += dt * r.records[-1].get("P_rad", 0.0)
        inside = any(np.any(s["x"] >= 0) for s in self.shards)
        if inside and not q["pin_off"]:
            q["pin_off"] = True
            r.set_param("pedestal.set_pedestal", 0.0)
        rho_q2 = r.records[-1].get("rho_2_1")
        rho = np.asarray(r.state.geometry.rho_norm)
        Te = np.asarray(r.state.core_profiles.T_e.value)
        if rho_q2 is not None and np.isfinite(rho_q2):
            T_here = float(np.interp(rho_q2, rho, Te))
            q.setdefault("T_ref", T_here)
            if not q["mixing"] and T_here < q["mix_at"] * q["T_ref"]:
                q["mixing"] = True
                self.h.bursts.append(dict(rows=[0, 1, 2], rho=(-1.0, 2.0), factor=q["mix"], until=r.t + 1.0))
                self.log("cold front at q = 2: mixing starts", T_keV=T_here)
        if float(r.ppo.W_thermal_total) < 0.1 * q["W"]:
            self.end_shot("mitigated")

    def end_shot(self, kind):
        r = self.r
        r.records[-1].update(disrupted=1.0, disrupted_kind=kind)
        r.disrupted = True
        r.finished = True
        self.card = self.disruption_card(kind)
        self.log("end", kind=kind)

    def _loops(self, rec, dt):
        r = self.r
        s = self.settings
        d = s["density_control"]
        feedback = 0.0
        if d.get("target"):
            feedback = float(np.clip(d["gain"] * (d["target"] - rec.get("n_line", 0.0) / 1e20), 0.0, d["rate_max"]))
        total = self.gas_rate + feedback
        r.carry["gas_rate"] = total
        # the pedestal's particle balance, since TORAX takes its height as an input: extra electrons in the plasma N, fed by the gas at a fuelling efficiency, lost on tau_p. Assumed: tau_p 3 s, eta_gas 0.1.
        electrons = float((np.asarray(r.state.core_profiles.n_e.value) * self._dv()).sum())
        self.fuel_extra += dt * self.eta_gas * total + self.fuel_atoms - dt * self.fuel_extra / self.tau_p
        self.fuel_atoms = 0.0
        n_factor = (1 + self.fuel_extra / electrons) * (1 - 0.2 * s["rmp"]["level"])
        r.ped_extra = np.array([(r.get_param("profile_conditions.Ip") / self.Ip0) ** 2 / n_factor, n_factor])      # the pedestal pressure follows the current squared, as in EPED; density comes from the balance, temperature from pressure over density
        c = s["radiation_control"]
        if c.get("target_MW"):
            rate = float(np.clip(c["gain"] * (rec.get("P_SOL", 0.0) / 1e6 - c["target_MW"]), 0.0, c["rate_max"]))
            shape = np.exp((np.asarray(r.state.geometry.rho_norm) - 1.0) / 0.03)
            self.imp.steady[c["species"]] = rate * shape / max((shape * self._dv()).sum(), 1e-30)
            r.carry["seed_rate"] = rate
        ip = s.get("ip")
        if ip:
            step = np.clip(ip["target"] - ip["now"], -ip["rate"] * dt, ip["rate"] * dt)
            if abs(step) > 0:
                ip["now"] += float(step)
                r.set_param("profile_conditions.Ip", ip["now"])
        m = s["mitigation"]
        if m.get("armed") and not self.fired:
            widths = 100 * max(r.widths) if r.widths else 0.0
            locked = any(rec.get(f"locked_{md.m}_{md.n}") for md in self.h.modes)
            if locked or (m["width_cm"] > 0 and widths >= m["width_cm"]):
                self.fired = True
                self.h.script = sorted(self.h.script + [dict(cmd="spi", species=m["species"], atoms=m["atoms"], hydrogen=m["hydrogen"], t=r.t + m["delay"])], key=lambda x: x["t"])
                self.log("mitigation fired", locked=bool(locked), width_cm=widths)
        lim = s["limits"]
        if lim.get("on"):
            if rec.get("f_GW", 0.0) > lim["greenwald"] and not any(e.name == "marfe" for e in self.effects):
                self.h.apply(dict(cmd="marfe"))
                self.log("Greenwald limit", f_GW=rec.get("f_GW"))
            P_SOL, P_LH = rec.get("P_SOL", 1.0), rec.get("P_LH", 0.0)
            if P_SOL < P_LH:
                self.below_since = self.below_since if self.below_since is not None else r.t
                if r.t - self.below_since > lim["hl_hold"] and not any(e.name == "back_transition" for e in self.effects):
                    self.h.apply(dict(cmd="back_transition"))
                    self.log("below the L-H threshold", P_SOL_MW=P_SOL / 1e6, P_LH_MW=P_LH / 1e6)
            else:
                self.below_since = None
            P_in = rec.get("P_alpha", 0.0) + r.get_param("sources.generic_heat.P_total") * float(ntm.HEAT[0]) + rec.get("P_ohmic", 0.0)
            if rec.get("P_rad", 0.0) > 1.2 * max(P_in, 1.0) and rec.get("W_thermal", 0.0) < 0.5 * float(r.records[0].get("W_thermal", 1e30)):
                self.end_shot("radiative collapse")
        if "rwm" in s and self.A_rwm > 0:
            v = s["rwm"]
            gamma = (rec.get("beta_N", 0.0) / v["beta_N_limit"] - 1) / v["tau_wall"] - v["gain"]
            self.A_rwm = float(self.A_rwm * np.exp(gamma * dt))
            r.carry["A_rwm"] = self.A_rwm
            if self.A_rwm > 1.0:
                self.end_shot("resistive wall mode")
        if "vde" in s and self.z_cm != 0:
            v = s["vde"]
            n = max(int(dt / 1e-3), 1)
            z = self.z_cm / 100
            for _ in range(n):
                z += dt / n * (v["growth"] * z - np.clip(v["gain"] * z, -v["v_max"], v["v_max"]))
            self.z_cm = float(z * 100)
            r.carry["z_cm"] = self.z_cm
            if abs(self.z_cm) > v["z_max_cm"]:
                self.end_shot("vertical displacement")

    def _alpha(self):
        """The alpha heat density against the table's knots, electrons and ions."""
        st = self.r.state
        rho = np.asarray(st.geometry.rho_norm)
        e, i = st.core_sources.T_e.get("fusion"), st.core_sources.T_i.get("fusion")
        return (np.interp(ntm.TURB_RHO, rho, np.asarray(e)) if e is not None else np.zeros_like(ntm.TURB_RHO),
                np.interp(ntm.TURB_RHO, rho, np.asarray(i)) if i is not None else np.zeros_like(ntm.TURB_RHO))

    def _redistribute(self, share, lost, edge):
        """Take a share of the alpha heat inside edge and put it, less what is lost, just outside."""
        a_e, a_i = self._alpha()
        x = ntm.TURB_RHO
        inside = 0.5 * (1 - np.tanh((x - edge) / 0.03))
        rho, dv = np.asarray(self.r.state.geometry.rho_norm), self._dv()
        vol = np.interp(x, rho, np.asarray(self.r.state.geometry.vpr))
        spot = gaussian(x, min(edge + 0.1, 0.95), 0.05)
        out = []
        for a in (a_e, a_i):
            take = share * a * inside
            power = np.trapezoid(take * vol, x)
            give = (1 - lost) * power * spot / max(np.trapezoid(spot * vol, x), 1e-30)
            out.append(give - take)
        return out

    def _effect(self, e, t, dt):
        c = e.c
        if e.name == "fishbone":
            while t >= e.extra["next"]:
                e.extra["next"] += c["period"]
                self.log("fishbone burst")
                if self.rng.random() < c["seed_prob"]:
                    self.h.apply(dict(cmd="island", m=2, n=1, w_cm=c["seed_cm"]))
            return self._redistribute(c["expelled"], c["lost"], c["rho"])
        if e.name == "alfven":
            return self._redistribute(c["redistributed"], c["lost"], c["rho"])
        if e.name == "marfe":
            x = ntm.TURB_RHO
            rho = np.asarray(self.r.state.geometry.rho_norm)
            vol = np.interp(x, rho, np.asarray(self.r.state.geometry.vpr))
            zone = 0.5 * (1 + np.tanh((x - 0.9) / 0.02))
            sink = c["share"] * self.r.records[-1].get("P_SOL", 0.0) * zone / max(np.trapezoid(zone * vol, x), 1e-30)
            cp = self.r.state.core_profiles
            energy = 1.5 * np.interp(x, rho, np.asarray(cp.n_e.value) * np.asarray(cp.T_e.value)) * 1e3 * QE
            return -np.minimum(sink, 0.2 * energy / self.h.dt), np.zeros_like(sink)
        if e.name == "back_transition":
            f = min((t - e.extra["t0"]) / max(c["ramp"], 1e-6), 1.0)
            self.r.ped_scale = e.extra["ped"] * (1 + f * (np.array([c["T"], c["n"]]) - 1))
        return None

    def _end(self, e):
        if e.name in ("marfe", "back_transition"):
            self.r.ped_scale = e.extra["ped"].copy()

    def disruption_card(self, kind):
        """What a disruption would mean, from where the plasma stands: 0D estimates, not a quench simulation."""
        r = self.r
        rec = r.records[-1]
        geo = r.state.geometry
        R0, a, kappa = float(geo.R_major), float(geo.a_minor), float(np.asarray(geo.elongation_face)[-1]) if hasattr(geo, "elongation_face") else 1.7
        I_p = abs(r.get_param("profile_conditions.Ip"))
        li = rec.get("li3", 0.8)
        L = ntm.MU0 * R0 * (np.log(8 * R0 / (a * np.sqrt(kappa))) - 2 + li / 2)
        Z = rec.get("Z_eff_avg", 1.5)
        n = np.asarray(r.state.core_profiles.n_e.value).mean()
        cards = {}
        for T_eV in (10.0, 20.0, 40.0):
            eta = 1.65e-9 * Z * 15.0 / (T_eV / 1e3) ** 1.5
            R_p = eta * 2 * np.pi * R0 / (np.pi * a * a * kappa)
            E = I_p * R_p / (2 * np.pi * R0)
            E_c = n / 1e20 * 0.08
            cards[f"T_{int(T_eV)}eV"] = dict(tau_CQ_ms=1e3 * L / R_p, E_over_Ec=E / E_c)
        W0 = self.tq_start["W"] if self.tq_start else float(r.ppo.W_thermal_total)
        E_rad = self.tq_start.get("E_rad", 0.0) if self.tq_start else 0.0
        # the rest of the quench, from what is left: every species mixed through the plasma, by the 0D model
        W_now = float(r.ppo.W_thermal_total)
        dv = self._dv()
        V = float(dv.sum())
        n_avg = float((np.asarray(r.state.core_profiles.n_e.value) * dv).sum() / V)
        n_z = {}
        for sp, ratio in (self.imp.background.items() if self.imp else ()):
            self.tables.setdefault(sp, cooling_table(sp))
            n_z[sp] = ratio * n_avg + self.imp.atoms(sp) / V
        if not n_z:
            self.tables.setdefault("Ne", cooling_table("Ne"))
            n_z = dict(Ne=0.01 * n_avg)
        more_rad, more_cond, t_tq = quench_0d(W_now, V, n_avg, n_z, self.tables)
        E_rad += more_rad
        atoms = {s: self.imp.atoms(s) for s in self.imp.species if np.any(self.imp.excess[s] > 0)} if self.imp else {}
        return dict(kind=kind, t=r.t, W_at_end_MJ=W_now / 1e6, W_thermal_MJ=W0 / 1e6, W_magnetic_MJ=0.5 * L * I_p ** 2 / 1e6, I_p_MA=I_p / 1e6, E_radiated_MJ=E_rad / 1e6, radiated_share=(E_rad / W0 if W0 > 0 else None), E_conducted_MJ=max(W0 - E_rad, 0.0) / 1e6, quench_0d_ms=1e3 * t_tq, torax_stop=str(getattr(r.sim_error, "name", r.sim_error)), quench=cards, atoms_in_plasma=atoms,
                quench_ms=1e3 * (r.t - self.tq_start["t"]) if self.tq_start else None, avalanche_gain_exp=2.5 * I_p / 1e6, iter_cq_target_ms=[50, 150], note="estimates from the state at the end: L/R with Spitzer resistivity, Connor-Hastie field, gain exp(2.5 I[MA]); no quench is simulated")

    # branches
    def state(self):
        import copy
        return dict(imp=self.imp.state() if self.imp else None, effects=copy.deepcopy(self.effects), shards=copy.deepcopy(self.shards), heat={k: (a.copy(), b.copy()) for k, (a, b) in self.heat.items()}, parts={k: v.copy() for k, v in self.parts.items()},
                    settings=copy.deepcopy(self.settings), fuel=(self.fuel_atoms, self.fuel_extra, self.gas_rate), beam=self.beam.state() if self.beam else None, below_since=self.below_since, rng=self.rng.bit_generator.state, fired=self.fired, fast_until=self.fast_until, n_log=len(self.event_log), card=self.card, z=self.z_cm, A=self.A_rwm, tq=copy.deepcopy(self.tq_start), units=self.saved_units)

    def set_state(self, x):
        import copy
        if self.imp:
            self.imp.set_state(x["imp"])
        self.effects, self.shards = copy.deepcopy(x["effects"]), copy.deepcopy(x["shards"])
        self.heat, self.parts = {k: (a.copy(), b.copy()) for k, (a, b) in x["heat"].items()}, {k: v.copy() for k, v in x["parts"].items()}
        self.settings = copy.deepcopy(x["settings"])
        self.fuel_atoms, self.fuel_extra, self.gas_rate = x["fuel"]
        if self.beam:
            self.beam.set_state(x["beam"])
        self.below_since, self.fired, self.fast_until, self.card, self.z_cm, self.A_rwm = x["below_since"], x["fired"], x["fast_until"], x["card"], x["z"], x["A"]
        self.rng.bit_generator.state = x["rng"]
        del self.event_log[x["n_log"]:]
        self.tq_start = copy.deepcopy(x["tq"])
        self.saved_units = x["units"]
        if self.settings["rmp"]["level"] > 0 or self.base_elm:
            self.apply_rmp()

    def summary(self):
        out = dict(effects=[e.name for e in self.effects], shards=len(self.shards), settings={k: v for k, v in self.settings.items() if v}, card=self.card, z_cm=self.z_cm, A_rwm=self.A_rwm)
        if self.imp:
            out["impurity_atoms"] = {s: self.imp.atoms(s) for s in self.imp.species if np.any(self.imp.excess[s] > 0)}
        return out


if __name__ == "__main__":
    print(json.dumps({k: {a: v[a] for a in ("kind", "level", "title")} for k, v in SPEC.items()}, indent=1))
