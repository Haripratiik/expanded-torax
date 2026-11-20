"""Impurities in the harness (H41): material put into the plasma during a run, with TORAX's composition, dilution, Z_eff and line radiation following it.

TORAX takes impurity densities as inputs. Here an excess density n_Z(rho, t) per species, above the scenario's constant background, is evolved between TORAX's steps and written into TORAX's runtime parameters (Reactor.set_ratio) as n_Z / n_e:
  d(V' n_Z)/dt = d/drho [ (g1 / V') D_Z dn_Z/drho - g0 V_Z n_Z ] + V' S_Z      (rho = rho_norm, TORAX's form for electrons, implicit in time)
  D_Z = D_e from TORAX's transport model, at least d_floor (0.1 m^2/s), plus d_neo (0.05 m^2/s)
  V_Z = V_e from TORAX + a neoclassical pinch d_neo Z (dln n_i/dr - 0.5 dln T_i/dr) (Hirshman and Sigmar's trace limit with the usual temperature screening factor; collisional, so a rough figure for the hot core)
  edge: n_Z = 0 at rho = 1; what leaves goes to a wall reservoir, from which it returns at the edge (rate 1 / tau_return, recycling species) or is pumped (rate 1 / tau_pump)
Assumed: d_floor, d_neo, tau_return (0.1 s for noble gases, never for metals), tau_pump (2 s). Injected ions are at the local mean charge from TORAX's Mavrin fit when electrons are added.
Radiators are the species TORAX has fits for (He, Li, Be, C, N, O, Ne, Ar, Kr, Xe, W), and must be named in the scenario's "impurities" (zero is fine) because the compiled step fixes the list. Tracers (Ti, Al, Ni, Fe, Mo) are followed for transport studies with no effect on the plasma:
TORAX has no radiation data for them, and laser blow-off puts in too little to matter.
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_banded

from torax._src.torax_pydantic import interpolated_param_1d
from torax._src.torax_pydantic import interpolated_param_2d

import ntm

ATOMS = dict(He=(2, 4.0), Li=(3, 6.9), Be=(4, 9.0), B=(5, 10.8), C=(6, 12.0), N=(7, 14.0), O=(8, 16.0), Ne=(10, 20.2), Al=(13, 27.0), Ar=(18, 40.0), Ti=(22, 47.9), Fe=(26, 55.8), Ni=(28, 58.7), Kr=(36, 83.8), Mo=(42, 96.0), Xe=(54, 131.3), W=(74, 183.8))
RECYCLING = {"He", "Ne", "Ar", "Kr", "Xe", "N"}
TRACERS = {"Ti", "Al", "Ni", "Fe", "Mo"}


class Impurities:
    def __init__(self, reactor, background, d_floor=0.1, d_neo=0.05, tau_return=0.1, tau_pump=2.0):
        self.r, self.background, self.d_floor, self.d_neo, self.tau_return, self.tau_pump = reactor, dict(background), d_floor, d_neo, tau_return, tau_pump
        self.species = list(background) + sorted(TRACERS)
        n = len(reactor.state.geometry.rho_norm)
        self.excess = {s: np.zeros(n) for s in self.species}
        self.source = {s: np.zeros(n) for s in self.species}     # atoms per m^3 per s, for the next step
        self.steady = {s: np.zeros(n) for s in self.species}     # the same, until changed: gas valves
        self.reservoir = {s: 0.0 for s in self.species}
        self.t_last, self.changed = reactor.t, False
        self.knots = ntm.TURB_RHO
        names = tuple(self.background)
        knots = jnp.asarray(self.knots)
        with_edge = reactor.provider.edge is not None and hasattr(reactor.provider.edge, "fixed_impurity_concentrations")

        @jax.jit
        def update(provider, ratios):     # all species at once, compiled: uncompiled it costs about 25 ms per species. The edge model reads the same species, at the plasma's edge
            nodes = lambda x: tuple(x.plasma_composition.impurity.species[n] for n in names) + (tuple(x.edge.fixed_impurity_concentrations[n] for n in names) if with_edge else ())
            core = tuple(interpolated_param_2d.TimeVaryingArrayUpdate(value=ratios[i][None, :], rho_norm=knots, time=jnp.zeros(1)) for i in range(len(names)))
            edge = tuple(interpolated_param_1d.TimeVaryingScalarUpdate(value=ratios[i][-3:-2]) for i in range(len(names))) if with_edge else ()
            return provider.update_provider(nodes, core + edge)
        self.update = update

    def cell_volumes(self):
        geo = self.r.state.geometry
        return np.asarray(geo.vpr) * float(np.asarray(geo.drho_norm).ravel()[0])

    def deposit(self, species, atoms, shape):
        """Atoms added now, spread as shape (any profile on the cell grid, normalized here over the cells' volume)."""
        dv = self.cell_volumes()
        self.excess[species] = self.excess[species] + atoms * shape / max((shape * dv).sum(), 1e-30)
        self.changed = True

    def charge(self, species):
        """The mean charge against radius, for adding electrons. Above T_e's fit range the ion is stripped."""
        if species not in self.background:
            return np.full_like(self.excess[species], float(ATOMS[species][0]))
        from torax._src.physics import charge_states
        Te = np.clip(np.asarray(self.r.state.core_profiles.T_e.value), 0.1, 100.0)
        return np.asarray(charge_states.calculate_average_charge_state_single_species(Te, species))

    def atoms(self, species):
        return float((self.excess[species] * self.cell_volumes()).sum())

    def _solve(self, species, n, S, dt):
        st, geo = self.r.state, self.r.state.geometry
        Z = float(ATOMS[species][0])
        cp, tr = st.core_profiles, st.core_transport
        rf, a = np.asarray(geo.rho_face_norm), float(geo.a_minor)
        drho = float(np.asarray(geo.drho_norm).ravel()[0])
        vpr, g1v, g0 = np.asarray(geo.vpr), np.asarray(geo.g1_over_vpr_face), np.asarray(geo.g0_face)
        rc = np.asarray(geo.rho_norm)
        lnn, lnT = np.log(np.maximum(np.asarray(cp.n_i.value), 1.0)), np.log(np.maximum(np.asarray(cp.T_i.value), 1e-3))
        grad = lambda f: np.concatenate([[0.0], np.diff(f) / (drho * a), [(f[-1] - f[-2]) / (drho * a)]])
        D = np.maximum(np.asarray(tr.d_face_el), self.d_floor) + self.d_neo
        V = np.asarray(tr.v_face_el) + self.d_neo * Z * (grad(lnn) - 0.5 * grad(lnT))
        A = g1v * D / drho                      # flux at a face: -A (n_right - n_left) + B * n_upwind, per unit rho
        B = g0 * V
        Bplus, Bminus = np.maximum(B, 0.0), np.minimum(B, 0.0)
        m = len(n)
        lower, diag, upper = np.zeros(m), np.zeros(m), np.zeros(m)
        for i in range(m):
            left, right = i, i + 1              # faces
            diag[i] = vpr[i] * drho / dt
            # right face
            if i < m - 1:
                diag[i] += A[right] + Bplus[right]
                upper[i] = -A[right] + Bminus[right]
            else:                               # n = 0 at rho = 1, half a cell away
                diag[i] += 2 * A[right] + Bplus[right]
            # left face (zero at the axis)
            if i > 0:
                diag[i] += A[left] - Bminus[left]
                lower[i] = -A[left] - Bplus[left]
        rhs = vpr * drho * (n / dt + S)
        ab = np.zeros((3, m))
        ab[0, 1:] = upper[:-1]
        ab[1] = diag
        ab[2, :-1] = lower[1:]
        new = np.maximum(solve_banded((1, 1), ab, rhs), 0.0)
        out = (2 * A[-1] + Bplus[-1]) * new[-1]   # atoms per s through the last surface
        return new, out

    def advance(self):
        """One TORAX step later: move the excess, then tell TORAX."""
        t = self.r.t
        dt = t - self.t_last
        self.t_last = t
        active = False
        for s in self.species:
            has = np.any(self.excess[s] > 0) or np.any(self.source[s] > 0) or np.any(self.steady[s] > 0) or self.reservoir[s] > 0
            if not has:
                continue
            active = True
            if dt <= 0:
                continue
            S = self.source[s] + self.steady[s]
            res = self.reservoir[s]
            if s in RECYCLING and res > 0:
                back = res * (1 - np.exp(-dt / self.tau_return)) / dt
                shape = np.exp(-((np.asarray(self.r.state.geometry.rho_norm) - 1.0) / 0.03) ** 2)
                S = S + back * shape / max((shape * self.cell_volumes()).sum(), 1e-30)
                res -= back * dt
            self.excess[s], out = self._solve(s, self.excess[s], S, dt)
            res = (res + out * dt) * np.exp(-dt / self.tau_pump) if s in RECYCLING else 0.0
            self.reservoir[s] = res
            self.source[s] = np.zeros_like(S)
        if active or self.changed:
            self.write()
            self.changed = active

    def write(self):
        rc = np.asarray(self.r.state.geometry.rho_norm)
        ne = np.asarray(self.r.state.core_profiles.n_e.value)
        ratios = {s: self.background[s] + self.excess[s] / ne for s in self.background}
        charge = sum(ratios[s] * self.charge(s) for s in self.background)          # keep the ions' share positive: TORAX's quasineutrality needs it
        scale = np.minimum(0.9 / np.maximum(charge, 1e-30), 1.0)
        self.r.provider = self.update(self.r.provider, jnp.asarray(np.stack([np.interp(self.knots, rc, ratios[s] * scale) for s in self.background])))

    def state(self):
        return dict(excess={k: v.copy() for k, v in self.excess.items()}, source={k: v.copy() for k, v in self.source.items()}, steady={k: v.copy() for k, v in self.steady.items()}, reservoir=dict(self.reservoir), t_last=self.t_last, changed=self.changed)

    def set_state(self, x):
        self.excess, self.source, self.steady = {k: v.copy() for k, v in x["excess"].items()}, {k: v.copy() for k, v in x["source"].items()}, {k: v.copy() for k, v in x["steady"].items()}
        self.reservoir, self.t_last, self.changed = dict(x["reservoir"]), x["t_last"], x["changed"]
