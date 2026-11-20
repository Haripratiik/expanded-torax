"""FreeGSNKE as a separate process for TORAX (H46): TORAX's venv has numpy 2.5 and FreeGSNKE needs numpy under 2.5, so the two talk over pipes, each in its own environment. Commands are pickled (name, arguments) on stdin, replies pickled on stdout, arrays as numpy.
FreeGSNKE (Amorisco et al. 2024, UKAEA, LGPL-3) solves the free-boundary equilibrium with coils and passive structures, static or evolving. It has no transport or heating of its own: that is TORAX, which sends p' and FF' against normalized flux, and the plasma current.
Commands:
  build     machine ('ITER', 'MAST-U', or paths to its pickles), grid. Sets up the equilibrium and the static solver
  profiles  Ip, fvac, psi_n, pprime, ffprime: what TORAX says the plasma is
  currents  name to current, in amps: sets coils
  forward   a static solve at those coils and that plasma; returns the equilibrium
  inverse   R, Z points the boundary should pass through, and X-points: finds coil currents for them, then solves
  evolve_start  full_timestep, plasma_resistivity: the evolutive solver from the present equilibrium
  evolve    voltages (or None to hold the coils' present resistive voltages), n steps, optional new profiles and resistivity: steps it on; returns the equilibrium
Usage: not by hand; freeboundary.py starts it with FreeGSNKE's Python (XTORAX_FREEGSNKE_PYTHON or external/freegsnke-venv).
"""
import ctypes
import os
import pickle
import sys
import threading
import time
import traceback

import numpy as np

ROOT = os.path.abspath(os.environ.get("XTORAX_HOME") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the workspace: runs/twin and external/ (README, "Where things live")
FREEGSNKE = os.environ.get("XTORAX_FREEGSNKE_SRC") or os.path.join(ROOT, "external", "freegsnke")   # FreeGSNKE's source checkout
CONFIGS = os.path.join(FREEGSNKE, "machine_configs")
sys.path.insert(0, FREEGSNKE)       # the repository's code, one fix past release 3.1.1: that release cannot load the ITER machine (a resistance entry comes back as an array)
MACHINES = {"ITER": ("ITER", "ITER"), "MAST-U": ("MAST-U", "MAST-U_like"), "SPARC": ("SPARC", "SPARC")}


def unique_passive(path):
    """A copy of a machine's passive description with each name used once. FreeGSNKE keys elements by name, and its ITER description has two names twice (IS_0, OS_0), so each pair became one element counted twice and the inductance matrix singular."""
    folder = os.path.join(ROOT, "runs", "twin", "freeboundary", "iter_machine")
    os.makedirs(folder, exist_ok=True)
    passive = pickle.load(open(path, "rb"))
    seen = {}
    for x in passive:
        seen[x["name"]] = seen.get(x["name"], 0) + 1
        if seen[x["name"]] > 1:
            x["name"] = f"{x['name']}_{seen[x['name']]}"
    out = os.path.join(folder, "passive.pickle")
    pickle.dump(passive, open(out, "wb"))
    return out


class Server:
    def __init__(self):
        self.eq = self.profiles = self.solver = self.stepping = self.tokamak = None
        self.volts = None
        self.kept = {}                                 # branch points, by name (keep, recall)

    def build(self, machine, Rmin, Rmax, Zmin, Zmax, nx=65, ny=129, paths=None):
        from freegsnke import GSstaticsolver, build_machine, equilibrium_update
        if paths is None:
            folder, stem = MACHINES[machine]
            base = os.path.join(CONFIGS, folder, stem)
            paths = dict(active_coils_path=base + "_active_coils.pickle", passive_coils_path=unique_passive(base + "_passive_coils.pickle") if machine == "ITER" else base + "_passive_coils.pickle", limiter_path=base + "_limiter.pickle", wall_path=base + "_wall.pickle")
        self.paths = paths
        self.tokamak = build_machine.tokamak(**paths)
        self.eq = equilibrium_update.Equilibrium(tokamak=self.tokamak, Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax, nx=nx, ny=ny)
        self.solver = GSstaticsolver.NKGSsolver(self.eq)
        return dict(coils=list(self.tokamak.coils_dict.keys()) if hasattr(self.tokamak, "coils_dict") else None, active=self.active_names())

    def drawing(self):
        """The machine's own outline, for a picture: coils and passive structures as boxes, the wall as a line, in plain lists."""
        load = lambda key: pickle.load(open(self.paths[key], "rb"))
        line = lambda d: [[float(x["R"]), float(x["Z"])] for x in d] if isinstance(d, list) else [[float(r), float(z)] for r, z in zip(d["R"], d["Z"])]
        wall = line(load("wall_path"))
        coils = []

        def add(name, c, block=False):
            R, Z = np.atleast_1d(np.asarray(c["R"], float)), np.atleast_1d(np.asarray(c["Z"], float))
            dR, dZ = float(np.max(c.get("dR", 0.01))), float(np.max(c.get("dZ", 0.01)))
            groups = [np.ones_like(R, bool)] if block else [Z >= 0, Z < 0] if len(R) > 40 else [np.arange(len(R)) == i for i in range(len(R))]
            for m in groups:
                if m.any():
                    w, h = float(R[m].max() - R[m].min()) + dR, float(Z[m].max() - Z[m].min()) + dZ
                    coils.append(dict(name=name, R=float(0.5 * (R[m].max() + R[m].min())), Z=float(0.5 * (Z[m].max() + Z[m].min())), dR=w, dZ=h))

        for name, c in load("active_coils_path").items():
            if "R" in c:
                add(name, c)
            else:
                for sub in c.values():
                    add(name, sub, block=True)
        passive = []
        for c in load("passive_coils_path"):
            R, Z = np.atleast_1d(np.asarray(c["R"], float)), np.atleast_1d(np.asarray(c["Z"], float))
            if "dR" in c and len(R) == 1:
                passive.append([float(R[0]), float(Z[0]), float(c["dR"]), float(c["dZ"])])
            else:
                passive.append([float(0.5 * (R.max() + R.min())), float(0.5 * (Z.max() + Z.min())), float(R.max() - R.min()), float(Z.max() - Z.min())])
        Rw, Zw = np.array([x[0] for x in wall]), np.array([x[1] for x in wall])
        near = [c for c in coils if c["R"] < 1.6 * Rw.max()]       # a far coil standing in for position control would shrink the picture
        pad = 0.04 * float(Zw.max() - Zw.min())
        Rb = [min(Rw.min(), min(c["R"] - c["dR"] / 2 for c in near)) - pad, max(Rw.max(), max(c["R"] + c["dR"] / 2 for c in near)) + pad]
        Zb = [min(Zw.min(), min(c["Z"] - c["dZ"] / 2 for c in near)) - pad, max(Zw.max(), max(c["Z"] + c["dZ"] / 2 for c in near)) + pad]
        return dict(wall=wall, coils=near, passive=passive, box=[float(Rb[0]), float(Rb[1]), float(Zb[0]), float(Zb[1])])

    def active_names(self):
        names = getattr(self.tokamak, "coils_list", None)
        n = getattr(self.tokamak, "n_active_coils", None)
        return list(names[:n]) if names is not None and n else None

    def profiles_set(self, Ip, fvac, psi_n, pprime, ffprime, Raxis=1.0):
        from freegsnke.jtor_update import GeneralPprimeFFprime
        self.table = dict(Ip=float(Ip), fvac=float(fvac), psi_n=np.asarray(psi_n, float), pprime=np.asarray(pprime, float), ffprime=np.asarray(ffprime, float), Raxis=float(Raxis))
        self.profiles = GeneralPprimeFFprime(eq=self.eq, Ip=float(Ip), fvac=float(fvac), psi_n=np.asarray(psi_n, float), pprime_data=np.asarray(pprime, float), ffprime_data=np.asarray(ffprime, float), Raxis=float(Raxis), Ip_logic=True)
        return True

    def profiles_simple(self, Ip, fvac, Raxis, Beta0=0.6, alpha_m=2.0, alpha_n=1.4):
        """A simple parametrized plasma, to find coils before TORAX's detailed profiles are handed over."""
        from freegsnke.jtor_update import Fiesta_Topeol
        self.profiles = Fiesta_Topeol(eq=self.eq, Beta0=float(Beta0), Ip=float(Ip), fvac=float(fvac), Raxis=float(Raxis), alpha_m=float(alpha_m), alpha_n=float(alpha_n))
        return True

    def arrays(self, x):
        """The present profile's p' and FF' at normalized flux x, with its scale L in them."""
        x = np.asarray(x, float)
        L = float(getattr(self.profiles, "L", 1.0))
        return dict(pprime=np.asarray(self.profiles.pprime(x), float), ffprime=np.asarray(self.profiles.ffprime(x), float), L=L)

    def guess(self, psi_total):
        """Start from a known total flux, on this solver's grid, [R, Z]: free-boundary problems have several solutions, and the start picks one. Coils must be set first."""
        self.eq.tokamak_psi = self.eq.tokamak.calcPsiFromGreens(pgreen=self.eq._pgreen)
        self.eq.plasma_psi = np.asarray(psi_total, float) - self.eq.tokamak_psi
        self.eq._updatePlasmaPsi(self.eq.plasma_psi)
        return True

    def currents(self, values):
        for name, value in values.items():
            self.eq.tokamak.set_coil_current(coil_label=name, current_value=float(value))
        return True

    def forward(self, tol=1e-5, max_iter=60, symmetric=False):
        t0 = time.perf_counter()
        self.solver.solve(eq=self.eq, profiles=self.profiles, constrain=None, target_relative_tolerance=tol, max_solving_iterations=max_iter, verbose=0, force_up_down_symmetric=symmetric)
        return self.result(self.eq, self.profiles, wall=time.perf_counter() - t0)

    def anchor_passive(self, psi, j, R, Z):
        """At the start of a run: EFIT's flux and current density [z, R] on this grid. The vessel currents that make up what the coils do not (mast_machine.passive_fit), then EFIT's flux as the start of the solver. Returns how much was left."""
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import mast_machine
        psi, j = np.asarray(psi, float), np.asarray(j, float)
        R, Z = np.asarray(R, float), np.asarray(Z, float)
        left = mast_machine.passive_fit(self, psi, j, R, Z)
        self.guess(psi.T)
        return dict(left=left[0], rms=left[1], total=left[2], Ip_grid=float((j * (R[1] - R[0]) * (Z[1] - Z[0])).sum()))

    def save(self):
        """Remember the present solution, plasma flux and coils, to come back to if a solve goes astray."""
        self.saved = (self.eq.plasma_psi.copy(), {k: float(v) for k, v in self.currents_now().items()})
        return True

    def vessel_start(self, R, Z, t, coils, t_eq, j_tor, t_end, resist=1.0):
        """The vessel's currents up to t_end from their circuit equation, on the coil currents and plasma current measured up to then."""
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import vessel
        self.vessel = vessel.Vessel(self.eq.tokamak, np.asarray(R, float), np.asarray(Z, float), factor=resist)
        self.vessel.run_up(t, coils, t_eq, j_tor, t_end)
        self.vessel_keep = 0.0
        return dict(total=float(self.vessel.I.sum()), times_ms=(1e3 * self.vessel.times()[:4]).tolist())

    def vessel_fit(self, psi, j, R, Z):
        """At the start of a run: a flux map and a plasma current density on it, [z, R]. What the map wants in the vessel beyond the circuit's currents is kept for the run as a fixed set. It stands for what the machine's description and the plasma get wrong at the start, so the run carries differences from a right start. Sets them, and the map as the solver's start."""
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import mast_machine
        psi, j = np.asarray(psi, float), np.asarray(j, float)
        R, Z = np.asarray(R, float), np.asarray(Z, float)
        left = mast_machine.passive_fit(self, psi, j, R, Z)
        fit = self.vessel.now()
        self.vessel_keep = fit - self.vessel.I
        self.guess(psi.T)
        return dict(left=left[0], rms=left[1], total=left[2], circuit=float(self.vessel.I.sum()), kept=float(np.sum(self.vessel_keep)), Ip_grid=float((j * (R[1] - R[0]) * (Z[1] - Z[0])).sum()))

    def vessel_advance(self, dt, coils):
        """One step of the circuit equation, on the plasma of the last solve, and its currents set in the machine."""
        j = getattr(self.profiles, "jtor", None)       # none if the last solve failed: the plasma stays as it last was
        self.last_j = np.asarray(j).T if j is not None else self.last_j
        self.vessel.step(float(dt), coils, self.last_j)
        self.vessel.apply(self.vessel_keep)
        return float(self.vessel.I.sum())

    def branch_state(self):
        """Everything the next solve and the vessel's next step start from: the plasma's flux, every coil and vessel current, the vessel's circuit, the profiles
        of the last solve, the trial-solve slot, and the solver's random draws (it takes one when its Newton-Krylov search stalls)."""
        return dict(psi=self.eq.plasma_psi, currents=self.currents_now(), profiles=self.profiles, table=getattr(self, "table", None), saved=getattr(self, "saved", None),
                    vessel=getattr(self, "vessel", None), vessel_keep=getattr(self, "vessel_keep", None), last_j=getattr(self, "last_j", None),
                    rng=self.solver.rng.bit_generator.state if getattr(self.solver, "rng", None) is not None else None)

    def shared(self):
        return {id(x): x for x in (self.eq, self.tokamak, self.solver) if x is not None}

    def keep(self, label):
        """A branch point under a name (the harness's snapshot)."""
        import copy
        self.kept[label] = copy.deepcopy(self.branch_state(), self.shared())
        return True

    def recall(self, label):
        """Back to a branch point (the harness's restore). It can be recalled any number of times."""
        import copy
        k = copy.deepcopy(self.kept[label], self.shared())
        self.currents(k["currents"])
        self.eq.tokamak_psi = self.eq.tokamak.calcPsiFromGreens(pgreen=self.eq._pgreen)
        self.eq.plasma_psi = k["psi"]
        self.eq._updatePlasmaPsi(self.eq.plasma_psi)
        self.profiles = k["profiles"]
        for a in ("table", "saved", "vessel", "vessel_keep", "last_j"):
            if k[a] is not None:
                setattr(self, a, k[a])
        if k["rng"] is not None:
            self.solver.rng.bit_generator.state = k["rng"]
        return True

    def forget(self, label):
        self.kept.pop(label, None)
        return True

    def dump_state(self):
        """What it takes to replay a solve elsewhere: the state saved before it."""
        return dict(saved_psi=self.saved[0], saved_currents=self.saved[1], now=self.currents_now(), R=self.eq.R, Z=self.eq.Z, table=self.table)

    def back(self):
        psi, cur = self.saved
        self.currents(cur)
        self.eq.tokamak_psi = self.eq.tokamak.calcPsiFromGreens(pgreen=self.eq._pgreen)
        self.eq.plasma_psi = psi.copy()
        self.eq._updatePlasmaPsi(self.eq.plasma_psi)
        return True

    def currents_now(self):
        return dict(self.eq.tokamak.getCurrents())

    def inverse(self, R, Z, null_R=None, null_Z=None, tol=1e-4, limits=None, max_iter=100, l2_reg=1e-14, frozen=(), symmetric=False):
        """frozen: coils that stay where they are. limits: the largest current each coil may carry, by name: the boundary alone leaves a common part of the coils free, and without limits that part runs to 100s of MA."""
        from freegsnke.inverse import Inverse_optimizer
        t0 = time.perf_counter()
        names = [n for n in self.active_names() if n not in frozen]
        for name in self.active_names():
            self.eq.tokamak[name].control = name in names
        coil_current_limits = [[float(limits.get(n, 1e9)) for n in names], [-float(limits.get(n, 1e9)) for n in names]] if limits else None
        isoflux = [[np.asarray(R, float), np.asarray(Z, float)]]
        nulls = [np.asarray(null_R, float), np.asarray(null_Z, float)] if null_R is not None else None
        constrain = Inverse_optimizer(isoflux_set=isoflux, null_points=nulls, coil_current_limits=coil_current_limits)
        self.solver.inverse_solve(eq=self.eq, profiles=self.profiles, constrain=constrain, target_relative_tolerance=tol, target_relative_psit_update=1e-3, verbose=False, l2_reg=l2_reg, l2_reg_fj=max(1e-8, l2_reg), Picard_handover=1e-4, full_jacobian_handover=[1e-3, 1e-2], max_solving_iterations=max_iter, force_up_down_symmetric=symmetric)
        return self.result(self.eq, self.profiles, wall=time.perf_counter() - t0, currents_active={n: float(self.eq.tokamak.getCurrents()[n]) for n in self.active_names()})

    def growth(self, plasma_resistivity, tol=1e-6):
        """The vertical instability about the present solution, from FreeGSNKE's linear model of the plasma and all the metal: growth rates per second, and the Leuer ratio of stabilising to destabilising force gradients. Superconducting coils get a small resistance, as its circuit solver inverts them. On Windows the model builds on one worker (it forks otherwise)."""
        from freegsnke import nonlinear_solve
        from freegsnke.jtor_update import Lao85
        t0 = time.perf_counter()
        resist = np.asarray(self.eq.tokamak.coil_resist, float).copy()
        resist[resist <= 0] = 1e-7
        profiles = self.profiles
        if type(profiles).__name__ == "GeneralPprimeFFprime":     # its linear model takes parametrized profiles only: the same p' and FF' as polynomials
            x = np.asarray(self.table["psi_n"], float)
            A = np.stack([x ** k - x ** 4 for k in range(4)], axis=1)
            alpha, beta = np.linalg.lstsq(A, self.table["pprime"], rcond=None)[0], np.linalg.lstsq(A, self.table["ffprime"], rcond=None)[0]
            profiles = Lao85(eq=self.eq, Ip=self.table["Ip"], fvac=self.table["fvac"], alpha=alpha, beta=beta, alpha_logic=True, beta_logic=True, Raxis=self.table["Raxis"], Ip_logic=True)
            self.solver.forward_solve(self.eq, profiles, target_relative_tolerance=tol, verbose=False)
        nl = nonlinear_solve.nl_solver(eq=self.eq, profiles=profiles, GSStaticSolver=self.solver, full_timestep=1e-3, plasma_resistivity=float(plasma_resistivity), n_linearization_workers=1, target_relative_tolerance_linearization=tol, custom_coil_resist=resist)
        lin = nl.linearised_sol
        return dict(growth_per_s=[float(np.real(x)) for x in lin.growth_rates], margin=[float(np.real(x)) for x in np.ravel(lin.stability_margin)], modes=int(nl.n_metal_modes), wall=time.perf_counter() - t0)

    def evolve_start(self, full_timestep, plasma_resistivity, max_mode_frequency=10 ** 2.5):
        from freegsnke import nonlinear_solve
        resist = np.asarray(self.eq.tokamak.coil_resist, float).copy()
        resist[resist <= 0] = 1e-7                  # superconducting coils: its circuit solver inverts the resistances
        self.stepping = nonlinear_solve.nl_solver(eq=self.eq, profiles=self.profiles, GSStaticSolver=self.solver, full_timestep=float(full_timestep), plasma_resistivity=float(plasma_resistivity), max_mode_frequency=max_mode_frequency, custom_coil_resist=resist, target_relative_tolerance_linearization=1e-6, n_linearization_workers=1)
        self.stepping.initialize_from_ICs(self.eq, self.profiles)
        n = self.stepping.evol_metal_curr.n_active_coils
        self.volts = (self.stepping.vessel_currents_vec * self.stepping.evol_metal_curr.coil_resist)[:n].copy()
        return dict(n_active=int(n), voltages=self.volts.copy())

    def evolve(self, n=1, voltages=None, psi_n=None, pprime=None, ffprime=None, Ip=None, resistivity=None, linear_only=False, tol_currents=0.005, tol_gs=0.003):
        t0 = time.perf_counter()
        volts = self.volts if voltages is None else np.asarray(voltages, float)
        params = None
        if pprime is not None:
            params = dict(psi_n=np.asarray(psi_n, float), pprime_data=np.asarray(pprime, float), ffprime_data=np.asarray(ffprime, float))
        for _ in range(int(n)):
            self.stepping.nlstepper(active_voltage_vec=volts, profiles_parameters=params, plasma_resistivity=resistivity, linear_only=linear_only, target_relative_tol_currents=tol_currents, target_relative_tol_GS=tol_gs, verbose=0, max_solving_iterations=50)
        return self.result(self.stepping.eq1, self.stepping.profiles1, wall=time.perf_counter() - t0, Ip_now=float(self.stepping.profiles1.Ip) if hasattr(self.stepping.profiles1, "Ip") else None)

    def result(self, eq, profiles, wall=0.0, **extra):
        opt, xpt = eq.opt, eq.xpt
        pn = np.linspace(0.005, 0.995, 100)
        try:
            lcfs = eq.separatrix(ntheta=180)
        except Exception:
            lcfs = None
        try:
            q = np.asarray(eq.q(pn))
        except Exception:
            q = None
        F = None
        try:                                   # F from FF': FreeGSNKE holds it through the profile's scale L
            x = np.linspace(0.0, 1.0, 101)
            ffp = float(getattr(profiles, "L", 1.0)) * np.asarray(profiles.ffprime_func(x), float)
            tail = np.concatenate([np.cumsum((0.5 * (ffp[1:] + ffp[:-1]) * np.diff(x))[::-1])[::-1], [0.0]])
            F2 = float(profiles._fvac) ** 2 - 2 * (float(profiles.psi_bndry) - float(profiles.psi_axis)) * tail
            F = np.sqrt(np.maximum(F2, 1e-6))
        except Exception:
            pass
        out = dict(F=F, R=eq.R.copy(), Z=eq.Z.copy(), psi=np.asarray(eq.psi()).copy(), axis=np.array(opt[0][:3]) if len(opt) else None, xpt=np.array(xpt[:2]) if len(xpt) else None, psi_axis=float(profiles.psi_axis) if hasattr(profiles, "psi_axis") else None, psi_bndry=float(profiles.psi_bndry) if hasattr(profiles, "psi_bndry") else None, lcfs=None if lcfs is None else np.asarray(lcfs), pn=pn, q=q, Ip=float(eq.plasmaCurrent()), wall=wall)
        for name, f in (("kappa", eq.geometricElongation), ("delta", eq.triangularity), ("R_geo", eq.Rgeometric), ("a", eq.minorRadius)):
            try:
                out[name] = float(f())
            except Exception:
                out[name] = None
        try:
            out["currents"] = {k: float(v.current) for k, v in eq.tokamak.coils_dict.items()} if hasattr(eq.tokamak, "coils_dict") and False else None
        except Exception:
            out["currents"] = None
        out.update(extra)
        return out


SOLVE_LIMIT_S = dict(forward=240.0, inverse=240.0)      # a solve takes seconds at most, on ITER too


class SolveTimeout(Exception):
    pass


def limited(fn, kw, seconds):
    """A solve with a time limit. FreeGSNKE's solvers have loops with no bound: in inverse_solve the coil update shrinks for as long as the plasma shows no X-point, which is for ever if it has none (met on MAST 27714, once in about 100 runs). Past the limit this raises into the solve, and again, since those loops catch everything, until it comes out; the caller then has an error and goes back to its saved state."""
    box = {}

    def work():
        try:
            box["value"] = fn(**kw)
        except BaseException as err:
            box.setdefault("error", err)
            box.setdefault("trace", traceback.format_exc())

    th = threading.Thread(target=work, daemon=True)
    th.start()
    th.join(seconds)
    while th.is_alive():
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(th.ident), ctypes.py_object(SolveTimeout))
        th.join(0.02)
    if "error" in box:
        if isinstance(box["error"], SolveTimeout):
            raise RuntimeError(f"no answer in {seconds:.0f} s")
        raise box["error"]
    return box["value"]


def main():
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr                 # FreeGS prints: keep the pipe clean
    s = Server()
    table = dict(arrays=s.arrays, build=s.build, drawing=s.drawing, profiles=s.profiles_set, profiles_simple=s.profiles_simple, guess=s.guess, anchor_passive=s.anchor_passive, save=s.save, back=s.back, keep=s.keep, recall=s.recall, forget=s.forget, currents_now=s.currents_now, currents=s.currents, growth=s.growth, dump_state=s.dump_state, forward=s.forward, vessel_start=s.vessel_start, vessel_fit=s.vessel_fit, vessel_advance=s.vessel_advance, inverse=s.inverse, evolve_start=s.evolve_start, evolve=s.evolve)
    while True:
        try:
            name, kw = pickle.load(stdin)
        except EOFError:
            return
        if name == "quit":
            return
        try:
            reply = dict(ok=True, value=limited(table[name], kw, SOLVE_LIMIT_S[name]) if name in SOLVE_LIMIT_S else table[name](**kw))
        except Exception as err:
            reply = dict(ok=False, error=repr(err), trace=traceback.format_exc()[-2500:])
        pickle.dump(reply, stdout, protocol=4)
        stdout.flush()


if __name__ == "__main__":
    main()
