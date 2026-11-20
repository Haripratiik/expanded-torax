"""A 2D island for the reactor (H44, docs/digital-twin.md): the NTM testbed's full solver (solver/ntm.py) or its surrogate (H31's module architecture, runs/h31/s4/), with one interface, so either can stand in for the island equation in TORAX's loop.

Both evolve the same slab: doubly periodic box, two current sheets, resistive MHD with current drive at a spot on the first sheet (the ECCD analogue), temperature with anisotropic heat transport and a bootstrap current. Normalized units (eta 0.01, 64 by 64). One step is 2 time units, the surrogate's stride.
Inputs each step: drive a (0.05 to 0.35 in training; 0.25 suppresses, 0.1 is too little), the drive's position x0 along the sheet relative to the island's O-point (within 0.3 in training: this is a phase error, not a radial one, which the data do not cover), c_bs and kpar.
Outputs: the reconnected flux of the first sheet (kx = 1 amplitude of By along y = pi/2), as in interp/h31_eval.py, and the O-point's position.
The drive follows the island: the slab is the island's own frame (a phase lock, or the average of a continuous drive). When too much drive turns the island over, its O-point jumps half a box and the fields are shifted back so that the O-point stays at the drive. The slab is periodic in x, so this changes nothing else, and it keeps the surrogate in the range it was trained on (drive within 0.3 of the O-point).
Connection to physical units, all assumptions in Link below: width = C sqrt(flux), from island width being proportional to the square root of the flux; time, how many seconds one 2-unit step stands for; power, a per MW. They are not derived from a tokamak: the slab is a testbed with S about 100.
The module's bootstrap and heat blocks saw only time up to 50 units in training, so long rollouts are outside what it was trained on; the solver has no such limit, which is why it stays in the loop as the reference.
"""
import os
import pickle
import sys
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

ROOT = os.path.abspath(os.environ.get("XTORAX_HOME") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the workspace: runs/twin and external/ (README, "Where things live")
for sub in ("models", "solver"):
    p = os.path.join(ROOT, sub)
    if p not in sys.path:
        sys.path.append(p)

def load_testbed():
    """solver/ntm.py, by path: core/ntm.py has the same name. The 2D solver and its surrogate belong to a separate project and are not
    part of this release; a scenario with "plant2d" set needs them in the workspace (XTORAX_HOME/solver, XTORAX_HOME/models)."""
    import importlib.util
    if not os.path.exists(os.path.join(ROOT, "solver", "ntm.py")):
        raise RuntimeError("the 2D island solver (solver/ntm.py) is not part of this release; see README.md, 'Optional pieces'")
    spec = importlib.util.spec_from_file_location("ntm_testbed", os.path.join(ROOT, "solver", "ntm.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


N, W, ETA, DT, SUB, KPERP = 64, 1.2, 0.01, 0.005, 400, 1e-4        # as in data/generate_ntm.py; SUB steps make the surrogate's 2 time units


@dataclass
class Link:
    C_cm: float = 35.0            # width in cm = C sqrt(flux): flux 0.03, a typical seed in the data, is about 6 cm
    step_s: float = 0.1           # seconds of plasma per 2-unit step: the data's islands double in about 40 steps
    a_per_MW: float = 0.0125      # so 20 MW is a = 0.25, the drive that suppresses in the testbed
    c_bs: float = 0.6             # at the reference bootstrap current
    j_bs_ref: float = 0.0         # MA/m^2; 0: the first value seen sets it
    kpar: float = 0.1


def flux_of(By):
    return float(np.abs(np.fft.rfft(np.asarray(By)[:, N // 4]))[1] * 2.0 / N)


def phase_of(By):
    """Where the O-point sits along the sheet, in radians from x = pi."""
    c = np.fft.rfft(np.asarray(By)[:, N // 4])[1]
    return float(np.angle(c))


class Solver2D:
    name = "solver"

    def __init__(self, flux0=0.02, phase0=0.0, flux2=0.003, seed=0):
        testbed = load_testbed()
        from tearing import equilibrium
        self.tb = testbed
        self.A0h, self.J0h = equilibrium(N, W)
        self.T0h = testbed.temperature0(N, True)
        self.g_cache = {}
        self.reset(flux0, phase0, flux2, seed)
        J0h, T0h = self.J0h, self.T0h
        self.adv = jax.jit(lambda s, p, g: testbed.advance(s, p, J0h, g, T0h, jnp.float32(DT), n=N, sub=SUB))
        self.phys = jax.jit(lambda s: testbed.physical(s, N))

    def reset(self, flux0, phase0=0.0, flux2=0.003, seed=0):
        self.state = (jnp.zeros_like(self.A0h), self.A0h + self.tb.seed(N, flux0, phase0, flux2, seed * 1.3), self.T0h)

    def drive(self, x0):
        key = round(float(x0), 2)
        if key not in self.g_cache:
            self.g_cache[key] = self.tb.eccd(N, W, np.pi + key)[0]
        return self.g_cache[key]

    def step(self, a, x0=0.0, c_bs=0.6, kpar=0.1):
        p = tuple(jnp.float32(v) for v in (ETA / 10, ETA, a, kpar, KPERP, c_bs))
        self.state = self.adv(self.state, p, self.drive(x0))
        return self.read()

    def read(self):
        f = self.phys(self.state)
        return flux_of(f[3]), phase_of(f[3])

    def seed_island(self, flux, phase=0.0):
        self.state = (self.state[0], self.state[1] + self.tb.seed(N, flux, phase, 0.0, 0.0), self.state[2])

    def shift(self, dx):
        """Move everything along the sheet by dx."""
        KX = self.tb.make_operators(N)["KX"]
        self.state = tuple(s * jnp.exp(-1j * KX * dx) for s in self.state)

    def get(self):
        return self.state

    def set(self, x):
        self.state = x


class Surrogate2D:
    name = "surrogate"

    def __init__(self, path=os.path.join("runs", "h31", "s4", "seed0", "mod_k2.pkl"), flux0=0.02, phase0=0.0, flux2=0.003, seed=0):
        testbed = load_testbed()
        import train_multi
        from tearing import equilibrium
        with open(os.path.join(ROOT, path), "rb") as fh:
            m = pickle.load(fh)
        self.tb = testbed
        self.params, self.norm = m["params"], m["norm"]
        modes = m["args"]["modes"]
        self.f = train_multi.make_step(modes, self.norm)
        self.jstep = jax.jit(lambda x, c: self.f(self.params, x, c))
        self.A0h = equilibrium(N, W)[0]
        self.T0h = testbed.temperature0(N, True)
        self.map_cache = {}
        self.reset(flux0, phase0, flux2, seed)

    def reset(self, flux0, phase0=0.0, flux2=0.003, seed=0):
        state = (jnp.zeros_like(self.A0h), self.A0h + self.tb.seed(N, flux0, phase0, flux2, seed * 1.3), self.T0h)
        self.x = jnp.concatenate([self.tb.physical(state, N), jnp.zeros((1, N, N))])[None]

    def dmap(self, x0):
        key = round(float(x0), 2)
        if key not in self.map_cache:
            self.map_cache[key] = jnp.asarray(self.tb.eccd(N, W, np.pi + key)[1])
        return self.map_cache[key]

    def step(self, a, x0=0.0, c_bs=0.6, kpar=0.1):
        self.x = self.x.at[0, 5].set(self.dmap(x0) * a)
        self.x = self.jstep(self.x, jnp.asarray([[kpar / 0.1, c_bs, a]], jnp.float32))
        return self.read()

    def read(self):
        By = self.x[0, 3]
        return flux_of(By), phase_of(By)

    def seed_island(self, flux, phase=0.0):
        s = self.tb.seed(N, flux, phase, 0.0, 0.0)
        X = jnp.fft.irfft2(s, s=(N, N))
        ops_y = jnp.fft.irfft2(1j * self.tb.make_operators(N)["KY"] * s, s=(N, N))
        ops_x = jnp.fft.irfft2(-1j * self.tb.make_operators(N)["KX"] * s, s=(N, N))
        self.x = self.x.at[0, 2].add(ops_y).at[0, 3].add(ops_x)
        del X

    def shift(self, dx):
        """Move everything along the sheet by a whole number of grid points nearest dx."""
        self.x = jnp.roll(self.x, int(round(dx * N / (2 * np.pi))), axis=2)

    def get(self):
        return self.x

    def set(self, x):
        self.x = x


class Plant2D:
    """A 2D island in place of the island equation for TORAX's first mode. Each plasma step_s it advances one 2-unit step under the drive TORAX's source is getting. TORAX's width then comes from here (Mode.prescribed), so the plasma's confinement, the island's rotation and any lock follow it,
    and the etched chip, if there is one, sees this width in its coil signals. What goes across is deliberately little: width, one way; drive power, its radial aim error and the bootstrap current, the other. The radial aim error reduces the drive, as in EtchedChip, since the slab cannot represent it."""

    def __init__(self, harness, spec):
        self.h, self.link = harness, Link(**spec.get("link", {}))
        cls = dict(solver=Solver2D, surrogate=Surrogate2D)[spec.get("backend", "surrogate")]
        kw = {k: spec[k] for k in ("path",) if k in spec}
        self.b = cls(flux0=1e-3, flux2=spec.get("flux2", 1e-3), **kw)
        self.flux2 = spec.get("flux2", 1e-3)
        self.backend, self.clock, self.t_last = spec.get("backend", "surrogate"), 0.0, harness.reactor.t
        self.flux, self.phase = 0.0, 0.0
        self.active, self.steps_alive = False, 0                 # with no island nothing evolves: the slab sits in its equilibrium, and a surrogate is not asked to hold still for minutes, which it cannot do
        self.continuous_eff = spec.get("continuous_eff", 0.5)    # a drive that is not modulated on the O-point also lands on the X-point
        self.blown, self.walls, self.a_now = False, [], 0.0
        self.flux_max = spec.get("flux_max", 0.5)                # past this the solver has shown nothing; in the runs made with it, its largest is about 0.6
        self.seed_events = []
        self.turns = 0                                           # how many times the island turned over
        self.small = 0

    @property
    def width_cm(self):
        return self.link.C_cm * float(np.sqrt(max(self.flux, 0.0)))

    def width(self, t):
        w = self.width_cm / 100
        return 0.0 if w < 0.003 else w

    def seed(self, w_cm):
        """Raise the island to at least this width."""
        target = (w_cm / self.link.C_cm) ** 2
        if target <= self.flux:
            return
        if self.active:
            self.b.seed_island(target - self.flux)
        else:
            self.b.reset(target, 0.0, self.flux2, len(self.seed_events))
            self.active, self.steps_alive, self.clock = True, 0, 0.0
        self.flux, self.phase = self.b.read()
        self.seed_events.append((self.h.reactor.t, w_cm))

    def before_step(self):
        import ntm
        r, l = self.h.reactor, self.link
        t = r.t
        self.clock += t - self.t_last
        self.t_last = t
        if not self.active:
            self.clock = 0.0
        while self.active and self.clock >= l.step_s - 1e-9:
            self.clock -= l.step_s
            loc = r.locs[0] if r.locs else None
            P = float(ntm.COMMAND["P"])
            modulated = bool(self.h.etched is not None)
            eff = 1.0 if modulated else self.continuous_eff
            c_bs, kpar = l.c_bs, l.kpar
            if loc is not None:
                off = abs(float(ntm.COMMAND["rho"]) - loc["rho"]) * loc["a"]
                eff *= float(np.exp(-4 * (off / 0.084) ** 2))
                if l.j_bs_ref == 0.0 and abs(loc["j_bs"]) > 0:
                    l.j_bs_ref = abs(loc["j_bs"])
                if l.j_bs_ref > 0:
                    c_bs = float(np.clip(l.c_bs * abs(loc["j_bs"]) / l.j_bs_ref, 0.2, 0.8))
            a = float(np.clip(P * l.a_per_MW * eff, 0.0, 0.35))
            self.a_now = a
            t0 = time.perf_counter()
            self.flux, self.phase = self.b.step(a, 0.0, c_bs, kpar)
            off = (self.phase + np.pi / 2 + np.pi) % (2 * np.pi) - np.pi      # from where the O-point sits when it is at the drive
            if abs(off) > 0.2:
                self.b.shift(off)
                self.turns += abs(off) > 1.5
                self.flux, self.phase = self.b.read()
            self.walls.append(time.perf_counter() - t0)
            self.steps_alive += 1
            self.small = self.small + 1 if self.width_cm < 0.3 else 0
            if self.small >= 10:                                  # gone: small for a second of plasma, since the flux passes through zero on its way when an island turns over
                self.active, self.flux, self.small = False, 0.0, 0
                break
            if not np.isfinite(self.flux) or self.flux > self.flux_max:
                self.blown = True
                self.flux = min(self.flux, self.flux_max) if np.isfinite(self.flux) else self.flux_max
                self.b.set(self.safe) if getattr(self, "safe", None) is not None else None
            else:
                self.safe = self.b.get()
        r.carry.update(flux_2d=self.flux, a_2d=self.a_now, blown_2d=float(self.blown))

    def state(self):
        return dict(b=self.b.get(), clock=self.clock, t_last=self.t_last, flux=self.flux, phase=self.phase, blown=self.blown, safe=getattr(self, "safe", None), ref=self.link.j_bs_ref, active=self.active, steps=self.steps_alive, n_seeds=len(self.seed_events))

    def set_state(self, x):
        self.b.set(x["b"])
        self.clock, self.t_last, self.flux, self.phase, self.blown, self.safe = x["clock"], x["t_last"], x["flux"], x["phase"], x["blown"], x["safe"]
        self.link.j_bs_ref = x["ref"]
        self.active, self.steps_alive = x["active"], x["steps"]
        del self.seed_events[x["n_seeds"]:]


def proportional(P_max=20.0, physics=None, gain=0.8, floor_cm=0.3):
    """A decision model for the 2D island: power in proportion to the island's flux, so the square of its width. Steady full power overdrives the slab: once the island is small, a fixed current spot forces an island of its own (check_overdrive.py). gain is MW per cm^2."""
    def decide(obs):
        w_cm, rho = 100 * max(obs["w"][0], 0.0), obs["rho"][0]
        P = 0.0 if w_cm < floor_cm else min(P_max, gain * w_cm ** 2)
        return P, rho if np.isfinite(rho) else 0.5
    return decide


def schedule_test(backend, steps=150, seed=0, c_bs=0.6, drive=((30, 70, 0.25),)):
    """Free growth, a burst of drive, free again. Returns flux and phase against step, and the cost."""
    b = backend(flux0=0.02, seed=seed)
    out, t0 = [], None
    for k in range(steps):
        a = 0.0
        for k0, k1, v in drive:
            if k0 <= k < k1:
                a = v
        if k == 1:
            t0 = time.perf_counter()
        out.append((k, a) + b.step(a, 0.0, c_bs))
    return np.array(out), (time.perf_counter() - t0) / (steps - 1)


if __name__ == "__main__":
    rows = {}
    for name, cls in (("solver", Solver2D), ("surrogate", Surrogate2D)):
        rows[name], cost = schedule_test(cls)
        print(f"{name}: {1e3 * cost:.0f} ms per 2-unit step", flush=True)
    s, m = rows["solver"], rows["surrogate"]
    print(" step    t     a   flux solver  flux surrogate   phase solver  phase surrogate")
    for k in list(range(0, 30, 5)) + list(range(30, 150, 10)) + [149]:
        print(f"{k:5d} {2 * k:4d} {s[k, 1]:5.2f} {s[k, 2]:12.4f} {m[k, 2]:14.4f} {s[k, 3]:12.2f} {m[k, 3]:14.2f}")
    print("mean |flux error| / mean flux, to step 25, 50, 100, 150:", [round(float(np.abs(s[:n, 2] - m[:n, 2]).mean() / s[:n, 2].mean()), 3) for n in (25, 50, 100, 150)])
