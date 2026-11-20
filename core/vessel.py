"""Vessel currents from their circuit equation (H46), for free-boundary solves that are otherwise static. For any machine FreeGSNKE describes:
M_pp dI_p/dt + R_p I_p = -M_pa dI_a/dt - dPhi/dt, with FreeGSNKE's inductance and resistance matrices, the coil currents given, and Phi the plasma's flux through each vessel element from its current density on the grid. Implicit in time.
The plasma and the vessel talk once per step, not within it, so the step has to stay shorter than the vessel's slowest time. FreeGSNKE's evolving solver does this coupling properly at 3 s per 0.5 ms step; this does it cheaply, between static solves. Runs in FreeGSNKE's venv.
"""
import numpy as np


class Vessel:
    def __init__(self, tok, R, Z, factor=1.0):
        from freegsnke.gradshafranov import Greens
        n = tok.n_active_coils
        self.tok, self.active, self.passive = tok, list(tok.coils_list[:n]), list(tok.coils_list[n:])
        M, r = np.asarray(tok.coil_self_ind), np.asarray(tok.coil_resist)
        self.M_pp, self.M_pa, self.R_p = M[n:, n:], M[n:, :n], r[n:] * factor
        RR, ZZ = np.meshgrid(R, Z)                 # [z, R], as the current densities come
        rows = []
        for name in self.passive:
            c = tok.coils_dict[name]
            coords = c["coords"]
            mult = np.resize(c["multiplier"], len(coords[0]))
            rows.append(2 * np.pi * np.sum([m * Greens(a, b, RR, ZZ) for a, b, m in zip(coords[0], coords[1], mult)], axis=0).ravel())
        self.G = np.array(rows)
        self.dA = float((R[1] - R[0]) * (Z[1] - Z[0]))
        self.I = np.zeros(len(self.passive))
        self.I_a = self.Phi = None

    def times(self):
        """The vessel's L/R times, longest first, in seconds."""
        w = np.linalg.eigvals(np.linalg.solve(self.M_pp, np.diag(self.R_p)))
        return np.sort(1.0 / w.real)[::-1]

    def flux(self, j):
        return self.G @ (np.asarray(j, float).ravel() * self.dA)

    def start(self, coils, j):
        """coils: currents by name; one not named counts as steady, so it induces nothing."""
        self.I_a, self.Phi = np.array([coils.get(n, 0.0) for n in self.active], float), self.flux(j)

    def step(self, dt, coils, j):
        a, phi = np.array([coils.get(n, 0.0) for n in self.active], float), self.flux(j)
        self.I = np.linalg.solve(self.M_pp + dt * np.diag(self.R_p), self.M_pp @ self.I - self.M_pa @ (a - self.I_a) - (phi - self.Phi))
        self.I_a, self.Phi = a, phi
        return self.I

    def run_up(self, t_coils, coils, t_eq, j_tor, t_end, step=0.002):
        """From the start of the coil traces to t_end, with no current in the vessel at the start: coils is a dict of traces against t_coils, j_tor the plasma's current density [z, R, time] against t_eq. Before a shot's breakdown there is no plasma, so the traces hold all that drives the vessel."""
        j_tor, t_eq = np.nan_to_num(j_tor).astype(float), np.asarray(t_eq, float)
        at = lambda t: {n: float(np.interp(t, t_coils, v)) for n, v in coils.items()}

        def j_at(t):
            i = int(np.clip(np.searchsorted(t_eq, t), 1, len(t_eq) - 1))
            w = float(np.clip((t - t_eq[i - 1]) / max(t_eq[i] - t_eq[i - 1], 1e-9), 0.0, 1.0))
            return (1 - w) * j_tor[:, :, i - 1] + w * j_tor[:, :, i]

        t = float(t_coils[0])
        self.I = np.zeros(len(self.passive))
        self.start(at(t), j_at(t))
        while t < t_end - 1e-9:
            dt = min(step, t_end - t)
            t += dt
            self.step(dt, at(t), j_at(t))
        return self.I

    def apply(self, extra=0.0):
        for name, value in zip(self.passive, self.I + extra):
            self.tok.set_coil_current(coil_label=name, current_value=float(value))

    def now(self):
        got = self.tok.getCurrents()
        return np.array([float(got[n]) for n in self.passive])
