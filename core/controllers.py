"""Controllers for the simulated reactor: TORAX plus the island model (core/ntm.py; H34, H35).

Each controller maps the observation after a TORAX step to the ECCD command for the next one: (power in MW, aim in
rho_norm). The observation is ntm.run's dict(t, w, rho, dwdt, P, aim), lists per mode; these controllers use mode 0.
  OpenLoop  fixed power from t_on at a fixed aim (the first ECCD checks)
  Tracking  the rule baseline: full power once the measured island passes w_on, aimed at the measured q = m/n
            surface plus aim_bias, and off again after it stays under w_off for hold seconds (P_idle while off:
            preventive drive); with a budget (MJ) it stops for good once that much energy is spent
  PhysMPC   model predictive control on the island equation itself, as real-time evaluations of the modified
            Rutherford equation do on TCV and ASDEX Upgrade: the fair physics baseline for any learned forecaster
  Measured  wraps a controller with what a real plant gives it: Gaussian noise on the width and the surface, a latency of
            some steps, and sensor faults (a bias on the width or surface between two times)
  Dropout   wraps a controller: no power between t1 and t2 (a gyrotron trip), whatever it commands
  Gyrotrons wraps a controller with the actuator: whole gyrotrons, trips of some of them, and a steering mirror with a
            finite rate and range
  Largest   points a controller that handles one mode at the largest measured island
Neural and chip controllers take the same observation (H35).
"""
import collections

import numpy as np

import ntm


class OpenLoop:
    def __init__(self, t_on, P, aim):
        self.t_on, self.P, self.aim = t_on, P, aim

    def __call__(self, obs):
        return (self.P if obs["t"] >= self.t_on else 0.0), self.aim


class Tracking:
    def __init__(self, P_max=20.0, w_on=0.03, w_off=0.005, hold=3.0, aim_bias=0.0, park=0.5, P_idle=0.0, budget=None, dt=0.5):
        self.P_max, self.w_on, self.w_off, self.hold, self.bias, self.park = P_max, w_on, w_off, hold, aim_bias, park
        self.P_idle, self.budget, self.dt, self.spent = P_idle, budget, dt, 0.0
        self.on, self.below_since = False, None

    def __call__(self, obs):
        w, rho, t = obs["w"][0], obs["rho"][0], obs["t"]
        if not self.on and w > self.w_on:
            self.on, self.below_since = True, None
        elif self.on:
            if w < self.w_off:
                self.below_since = t if self.below_since is None else self.below_since
                if t - self.below_since >= self.hold:
                    self.on = False
            else:
                self.below_since = None
        aim = rho + self.bias if np.isfinite(rho) else self.park
        P = self.P_max if self.on else self.P_idle
        if self.budget is not None:
            self.spent += obs["P"] * self.dt                    # what was delivered: a trip delivers nothing
            P = P if self.spent < self.budget else 0.0
        return P, aim


class PhysMPC:
    """After every step, roll the width forward over horizon seconds with ntm.rates, on the plant's latest local
    quantities held frozen, for powers {0, P_max / 2, P_max} aimed at the measured surface; take the smallest predicted
    width plus energy_cost times power times horizon / 2 (in cm per MW, so that horizon 2 s scores like NetMPC). coeffs are
    the equation's coefficients: the scenario's own for an oracle, the published fit otherwise. The driven current per MW
    is the last one measured (kappa0 until then, the median in the synthetic shots); the deposition is one grid cell."""

    def __init__(self, coeffs, P_max=20.0, horizon=2.0, energy_cost=0.03, h=0.25, kappa0=6.5e3, w_dep=0.02, cell=0.01):
        self.c, self.P_max, self.horizon, self.cost, self.kappa = coeffs, P_max, horizon, energy_cost, kappa0
        self.n, self.w_dep, self.cell = max(int(round(horizon / h)), 1), w_dep, cell

    def __call__(self, obs):
        w, rho, loc = obs["w"][0], obs["rho"][0], obs["loc"][0]
        if loc is None or not np.isfinite(rho):
            return 0.0, 0.5
        if obs["P"] > 0 and loc["I_cd"] != 0:
            self.kappa = abs(loc["I_cd"]) / obs["P"]
        dep = (np.floor(rho / self.cell) + 0.5) * self.cell     # the cell the drive lands in
        scores = {}
        for P in (0.0, self.P_max / 2, self.P_max):
            cand = dict(loc, rho=rho, I_cd=self.kappa * P, rho_dep=dep, w_dep=self.w_dep)
            scores[P] = 100 * ntm.advance(w, cand, self.c, self.horizon, self.n) + self.cost * P * self.horizon / 2
        return min(scores, key=scores.get), rho


class Measured:
    def __init__(self, inner, sigma_w=0.0, sigma_rho=0.0, latency=0, seed=0, faults=()):
        self.inner, self.sw, self.sr, self.faults = inner, sigma_w, sigma_rho, faults
        self.rng, self.queue = np.random.default_rng(seed), collections.deque(maxlen=latency + 1)

    def __call__(self, obs):
        live = [f for f in self.faults if f["t1"] <= obs["t"] < f["t2"]]
        bw, br = sum(f.get("width", 0.0) for f in live), sum(f.get("surface", 0.0) for f in live)
        o = dict(obs, w=[max(w + bw + self.rng.normal(0, self.sw), 0.0) for w in obs["w"]],
                 rho=[r + br + self.rng.normal(0, self.sr) for r in obs["rho"]])
        self.queue.append(o)
        return self.inner(dict(self.queue[0], t=obs["t"]))


class Dropout:
    def __init__(self, inner, t1, t2):
        self.inner, self.t1, self.t2 = inner, t1, t2

    def __call__(self, obs):
        P, aim = self.inner(obs)
        return (0.0 if self.t1 <= obs["t"] < self.t2 else P), aim


class Gyrotrons:
    """The actuator between a decision and the plasma. Power comes in whole gyrotrons (ITER: 170 GHz, about 1 MW each, up to 20 MW
    from the upper launchers at the 2/1 surface), some of them lost during trips; the aim moves at a finite rate inside the mirror's
    range and not at all while the mirror is stuck. trips: (t1, t2, units lost); stuck: (t1, t2). mirror_rate is in rho_norm per
    second (0.1 is 20 cm/s on ITER: an assumption, no published figure found yet); aim_range is the upper launcher's coverage,
    roughly, in rho_norm. Returns what is delivered, which is what the plant records as P_cmd and aim_cmd."""

    def __init__(self, inner, units=20, unit_MW=1.0, mirror_rate=0.1, aim_range=(0.35, 0.85), trips=(), stuck=(), dt=0.5):
        self.inner, self.units, self.unit_MW, self.rate, self.range = inner, units, unit_MW, mirror_rate, aim_range
        self.trips, self.stuck, self.dt, self.aim = trips, stuck, dt, None

    def __call__(self, obs):
        P, aim = self.inner(obs)
        t = obs["t"]
        left = max(self.units - sum(n for t1, t2, n in self.trips if t1 <= t < t2), 0)
        target = float(np.clip(aim, *self.range))
        if self.aim is None:
            self.aim = target
        elif not any(t1 <= t < t2 for t1, t2 in self.stuck):
            self.aim += float(np.clip(target - self.aim, -self.rate * self.dt, self.rate * self.dt))
        return min(int(P / self.unit_MW + 1e-9), left) * self.unit_MW, self.aim


class Largest:
    """Stays on its mode until another looks larger by margin (m), so that sensor noise does not swing the mirror between surfaces."""

    def __init__(self, inner, margin=0.015):
        self.inner, self.margin, self.k = inner, margin, 0

    def __call__(self, obs):
        k = int(np.argmax(obs["w"]))
        if obs["w"][k] > obs["w"][self.k] + self.margin:
            self.k = k
        return self.inner(dict(obs, **{name: [obs[name][self.k]] for name in ("w", "rho", "dwdt", "j_bs", "loc")}))
