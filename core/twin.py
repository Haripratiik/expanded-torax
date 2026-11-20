"""The digital twin (H36, docs/digital-twin.md): one entry point for a simulated reactor run.

A scenario (a JSON file or dict) says what plasma to run, what goes wrong and when, what the hardware can do, what the sensors see,
and which decision model is in charge. TORAX evolves the core plasma, ntm.py the islands; a decision model sees what the sensors see,
and only what the gyrotrons can deliver reaches the plasma.
  t_final, n_rho, dt        the run (defaults 150 s, 100 cells, 0.5 s)
  heating_factor            total heating relative to the base scenario (TORAX's ITER hybrid flat top, 51 MW)
  physics                   the island equation's coefficients: a_bs, a_cd, a_ggj, as ntm.Coefficients
  modes                     extra modes to follow, [[m, n], ...]; every mode named in an event is followed anyway
  rotation                  islands rotate and lock (default), a locked island quenching at 0.5 per second as measured on MAST (ntm.py); false, or a dict of ntm.Rotation's fields
  events                    what you inject, in any order:
      {"type": "island", "t": 20, "m": 2, "n": 1, "w_cm": 8}             a seed island at a set time
      {"type": "island_when", "m": 2, "n": 1, "w_cm": 6, "after": 100, "below_cm": 1}
                                                                         a seed once this mode is under below_cm, after that time
      {"type": "island_random", "m": 2, "n": 1, "rate_per_s": 0.02, "w_cm": [1, 10], "seed": 0}
                                                                         seeds at random times, log-uniform in width
      {"type": "island_from", "trigger": "sawtooth", "m": 2, "n": 1, "prob": 0.25, "w_cm": [2, 5], "after": 50, "seed": 0}
                                                                         a share of sawtooth crashes or ELMs ("elm") seed an island, how NTMs start in machines
      {"type": "gyrotron_trip", "t": 25, "duration": 10, "units": 12}    some gyrotrons lost
      {"type": "mirror_stuck", "t": 25, "duration": 10}                  the steering mirror frozen
      {"type": "sensor_bias", "t": 80, "duration": 20, "surface_cm": 2, "width_cm": 0}   a measurement fault
      {"type": "heating_step", "t": 40, "factor": 0.7}                   total heating changed
      {"cmd": "turbulence", "t": 30, "factor": 3, "rho": [0.5, 0.7], "duration": 5}  any harness command at a set time (Harness.COMMANDS: turbulence, turbulence_noise, kick, crash, elm, ...)
  edge                      TORAX's edge model on (default): target temperature and heat flux, and the separatrix temperature of the core; false for a fixed boundary
  elms                      Type I ELMs on (default): a crash and recovery of the pedestal top; false, or a dict of ntm.ELMs' fields. Needs dt well under the period (about 0.5 s on ITER)
  transport_model           in place of QLKNN: e.g. bohm-gyrobohm_table, for a spherical tokamak
  beams                     a machine's neutral beams, as beams.Beams: P_MW (a number or {t: MW}), E_keV, R_tan, sigma, f_lost, A_i, drive_current. Fast ions, slowing down, neutrons, driven current, rotation from their torque
  sawtooth                  TORAX's sawtooth model: true, or a dict of its settings (default off; needs a q = 1 surface, which the ITER hybrid does not have)
  machine_start             a machine with no equilibrium file: machine (in machines.py), Ip, B0, R0. Its first surfaces come from FreeGSNKE, on its coils and the shape it runs
  physical_current_after    from this time current diffuses at the physical rate (default 25 s; before it, 200 times faster, to settle). null keeps TORAX's example's 200 throughout
  gyrotrons                 units, unit_MW, mirror_rate (rho_norm per s), aim_range; see controllers.Gyrotrons
  sensors                   sigma_w_cm, sigma_rho_cm, latency_steps, seed
  decision                  the decision model:
      {"kind": "none"}
      {"kind": "rule", "w_on_cm": 3, "w_off_cm": 3, "hold": 3, "idle": 0, "budget": null}
      {"kind": "phys", "oracle": false, "horizon": 2, "cost": 0.15}      control on the island equation (controllers.PhysMPC)
      {"kind": "net", "model": "runs/torax_shots/forecaster.npz", "cost": 0.03, "chip": false}
                                                                         a forecaster with MPC; chip: true is 16-bit fixed point; "hgq",
                                                                         with "hgq_model", the exported chip design, bit for bit
      {"kind": "python", "target": "module:function", "args": {...}}     your own (a module, or a file as path/to/file.py:function): function(P_max=..., physics=..., **args) returns a
                                                                         callable taking obs and returning (power in MW, aim in rho_norm)
      {"kind": "etched", "final": true, "controller": "sim chip + decision chip", "seed": 0}  the companion chip project's chip, at its own 0.1 ms and 20 ms, in its own process (chip_server.py). The island's width and
                                                                         rotation then follow that project's model, fed by TORAX's plasma every step, and TORAX receives the width and the mean drive
  torax                     any TORAX settings, merged over the base config (time-dependent inputs included)
The decision model's observation is described in ntm.run. A model that handles one mode is pointed at the largest measured island.
Harness is the same reactor driven command by command: by a script, or by an agent through core/harness.py.
Outputs in <out>/<name>/: summary.json, traces.npz (every island term per step, delivered power and aim) and traces.png.
Usage (TORAX's venv): python core/twin.py scenarios/stress.json [--decision rule]
"""
import argparse
import copy
import dataclasses
import importlib
import json
import os
import subprocess
import sys
import time
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import controllers as ctl  # noqa: E402
import actuators  # noqa: E402
import forecaster as fc  # noqa: E402
import ntm  # noqa: E402
import run_ntm  # noqa: E402
import torax  # noqa: E402


def merge(a, b):
    for k, v in b.items():
        a[k] = merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return a


def torax_config(s, out=None):
    t_final = s.get("t_final", 150.0)
    cfg = run_ntm.base_config(t_final, n_rho=s.get("n_rho", 100), dt=s.get("dt", 0.5), physical_current_after=s.get("physical_current_after", 25.0))
    P0 = cfg["sources"]["generic_heat"]["P_total"] * s.get("heating_factor", 1.0)
    steps = sorted((e["t"], e["factor"]) for e in s.get("events", []) if e.get("type") == "heating_step")
    trace, level = {0.0: P0}, P0
    for t, f in steps:
        trace[t - 1e-3], trace[t] = level, P0 * f
        level = P0 * f
    cfg["sources"]["generic_heat"]["P_total"] = trace if steps else P0
    if s.get("machine_start"):                           # another machine: its first surfaces from FreeGSNKE, from its coils and the shape it runs, in place of the ITER file
        import freeboundary
        m = s["machine_start"]
        cfg["geometry"] = dict(freeboundary.start_geometry(m["machine"], m["Ip"], m["B0"], m["R0"], n_rho=cfg["geometry"]["n_rho"]), n_rho=cfg["geometry"]["n_rho"])
    if s.get("transport_model"):                         # where QLKNN is not valid (R/a below 2.7): TORAX's Bohm-gyroBohm, whose level the coupled loop holds to the confinement law
        cfg["transport"] = dict(model_name=s["transport_model"])
    species = s.get("impurities")
    if s.get("edge", True):
        run_ntm.with_edge(cfg, **({"species": species} if species else {}))
    if species:
        run_ntm.with_impurities(cfg, species)
    if s.get("sawtooth"):
        # TORAX's own test settings: with its defaults the shear just after a crash already passes the trigger, and it fires every step. It also fires every step
        # while current diffuses 200 times too fast, so natural periods need physical_current_after, and seeds need "after" past the settling.
        st = {"trigger_model": {"model_name": "simple", "minimum_radius": 0.15, "s_critical": 0.25},
              "redistribution_model": {"model_name": "simple", "flattening_factor": 1.001, "mixing_radius_multiplier": 1.3}, "crash_step_duration": 1e-3}
        cfg["mhd"] = {"sawtooth": merge(st, s["sawtooth"]) if isinstance(s["sawtooth"], dict) else st}
    if isinstance(s.get("beams"), dict) and s["beams"].get("drive_current"):
        cfg["sources"]["generic_current"] = dict(model_name="beams")
    return merge(cfg, copy.deepcopy(s.get("torax", {})))


def followed_modes(s):
    named = [tuple(m) for m in s.get("modes", [])]
    for e in s.get("events", []):
        if e.get("type", "").startswith("island") and (e.get("m", 2), e.get("n", 1)) not in named:
            named.append((e.get("m", 2), e.get("n", 1)))
    named = named or [(2, 1)]
    rot = s.get("rotation", True)
    return [ntm.Mode(m, n, seeds=tuple(sorted((e["t"], e["w_cm"] / 100) for e in s.get("events", [])
                                              if e.get("type") == "island" and (e.get("m", 2), e.get("n", 1)) == (m, n))),
                     rotation=None if rot is False else ntm.Rotation(**{"quench_rate": 0.5, **(rot if isinstance(rot, dict) else {})}))
            for m, n in named]


def injector(s, modes):
    """Islands raised during the run: conditional and random events."""
    index = {(md.m, md.n): k for k, md in enumerate(modes)}
    whens = [dict(e, k=index[(e.get("m", 2), e.get("n", 1))], done=False) for e in s.get("events", []) if e.get("type") == "island_when"]
    froms = [dict(e, k=index[(e.get("m", 2), e.get("n", 1))], rng=np.random.default_rng(e.get("seed", 0))) for e in s.get("events", []) if e.get("type") == "island_from"]
    randoms = [dict(e, k=index[(e.get("m", 2), e.get("n", 1))], rng=np.random.default_rng(e.get("seed", 0)))
               for e in s.get("events", []) if e.get("type") == "island_random"]
    log = []
    if not whens and not randoms and not froms:
        return None, log

    def inject(t, widths, events):
        out = []
        for e in froms:
            if t >= e.get("after", 0.0) and events.get(e["trigger"]) and e["rng"].random() < e.get("prob", 1.0):
                w = float(np.exp(e["rng"].uniform(*np.log(e["w_cm"]))))
                out.append((e["k"], w / 100))
                log.append(dict(t=t, mode=f'{e.get("m", 2)}/{e.get("n", 1)}', w_cm=w, kind=f'from {e["trigger"]}'))
        for e in whens:
            if not e["done"] and t >= e.get("after", 0.0) and widths[e["k"]] < e.get("below_cm", 1e9) / 100:
                e["done"] = True
                out.append((e["k"], e["w_cm"] / 100))
                log.append(dict(t=t, mode=f'{e.get("m", 2)}/{e.get("n", 1)}', w_cm=e["w_cm"], kind="island_when"))
        for e in randoms:
            if e.get("t1", 0.0) <= t < e.get("t2", 1e9) and e["rng"].random() < e["rate_per_s"] * s.get("dt", 0.5):
                w = float(np.exp(e["rng"].uniform(*np.log(e["w_cm"]))))
                out.append((e["k"], w / 100))
                log.append(dict(t=t, mode=f'{e.get("m", 2)}/{e.get("n", 1)}', w_cm=w, kind="island_random"))
        return out
    return inject, log


class ChipClient:
    """chip_server.py in WSL, where the chip's Keras models live; TORAX lives on Windows. One line of JSON each way per TORAX step. One server per process, kept between decision models."""
    live = {}

    def __init__(self, final, background):
        server = os.environ.get("XTORAX_CHIP_SERVER")
        if not server:
            raise RuntimeError("the etched chip is not part of this release: its networks and its server belong to a separate project. "
                               "XTORAX_CHIP_SERVER may name a shell command that starts a server speaking the same protocol (one line of JSON each way).")
        flags = ("--final_chip " if final else "") + (f"--background {background} " if background else "")
        os.makedirs(os.path.join("runs", "twin"), exist_ok=True)
        self.log = open(os.path.join("runs", "twin", "chip_server.log"), "w")
        argv = ["wsl.exe", "-e", "bash", "-lc", server + " " + flags] if os.name == "nt" else ["bash", "-lc", server + " " + flags]
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        self._read()

    def _read(self):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("the chip server stopped: see runs/twin/chip_server.log")
            if line.startswith("{"):
                return json.loads(line)

    def ask(self, **c):
        self.p.stdin.write(json.dumps(c) + "\n")
        self.p.stdin.flush()
        return self._read()

    @classmethod
    def get(cls, final=True, background=None):
        key = (final, background)
        if key not in cls.live:
            cls.live[key] = cls(final, background)
        return cls.live[key]


BANK = os.environ.get("XTORAX_MAST_BACKGROUND")   # recorded MAST coil background for the etched chip's input (a file of the chip's project); None: none


def wsl_path(path):
    """A Windows path as WSL sees it (C:\\a\\b to /mnt/c/a/b); unchanged elsewhere."""
    p = os.path.abspath(path)
    if os.name != "nt" or len(p) < 2 or p[1] != ":":
        return p
    return "/mnt/" + p[0].lower() + p[2:].replace("\\", "/")


class EtchedChip:
    """The chip as the reactor's decision model. Each TORAX step: TORAX's plasma now goes to the chip's loop as the island model's parameters, the loop runs the step's samples and decisions, and its width and mean drive come back."""

    def __init__(self, harness, spec):
        self.h, self.spec, self.dt = harness, spec, harness.dt
        self.client = ChipClient.get(spec.get("final", True), spec.get("background", BANK))     # real background by default since version 5: the design floor hides the chip's false drive (docs/digital-twin.md, Step 3). "background": null gives the floor
        self.seed_next, self.trigger, self.brake, self.w_next, self.noise, self.aim = 0.0, False, None, 0.0, spec.get("noise", 1.0), None
        self.kappa, self.extra = 7.0e3, {}
        self.rng = np.random.default_rng(spec.get("seed", 0))
        self.quench_rate = spec.get("quench_rate", 0.5)
        self.t_prev, self.owed, self.last = None, 0.0, None
        self.client.ask(op="reset", controller=spec.get("controller", "sim chip + decision chip"), seed=spec.get("seed", 0), noise=self.noise, **{k: spec[k] for k in ("cl", "k_bs") if k in spec})

    def __deepcopy__(self, memo):
        return self

    def width(self, t):
        return self.w_next

    def plant(self, l):
        eps = l["rbar"] / l["R0"]
        rho_theta = np.sqrt(2 * 2.5 * 1.66053907e-27 * l["T_i"] * 1e3 * ntm.QE) / (ntm.QE * eps * l["B0"] / l["q"])
        p = dict(r_s=l["rbar"], a=l["a"], j_bs=abs(l["j_bs"]), j_par=l["j_total"], L_q=l["rbar"] / l["s"], tau_r=ntm.MU0 * l["rbar"] ** 2 / l["eta"], w_ib=float(np.sqrt(eps) * rho_theta), eta_cd=self.kappa)
        if self.h.saved_rotation:                    # error_field commands reach the chip's plant too
            p["w_vac"] = self.h.saved_rotation.w_vac
        if self.spec.get("plant_form") == "eq":      # the plant is the island equation checked against shots (ntm.rates, with the reactor's own coefficients), not the paper's: what chip_iter.py tests
            c, rb, q, s, R0, B0 = self.h.reactor.c, l["rbar"], l["q"], l["s"], l["R0"], l["B0"]
            w_d = c.w_d if c.w_d is not None else ntm.small_island_width(l)
            L_q, L_p = rb / s, l["p"] / l["dpdr"] if l["dpdr"] != 0 else -np.inf
            D_R = -(q ** 2 - 1) * L_q ** 2 * (2 * ntm.MU0 * l["p"] / B0 ** 2) / (rb * L_p)
            w_small = c.k_pol * l.get("w_bi", 0.0) if c.k_pol else 0.0
            # the drive: ntm.rates' form goes to 16 mu0 R q / (pi s B) x 0.375 I / w^2 for a wide island. The chip's table of efficiency, which knows modulation and locking, has the same shape for continuous drive and tends to 0.37 (checked), hence the ratio
            p["eq"] = dict(form=1.0, d0=c.rdelta0 if c.rdelta0 is not None else -float(l["m"]), A_bs=c.a_bs * rb * (2 * ntm.MU0 * R0 * q / (s * B0)) * l["j_bs"], ws2=w_small ** 2, wd2=w_d ** 2, soft=0.0 if c.k_pol else 1.0,
                           G=-6 * c.a_ggj * rb * D_R, A_cd=c.a_cd * (16 * ntm.MU0 * R0 * q / (np.pi * s * B0)) * 0.375 / 0.37)
            for k in ("C_M", "f0", "tau_M0"):
                if k in self.spec:
                    p[k] = self.spec[k]
        return p

    def __call__(self, obs):
        loc, rho, g = obs["loc"][0], obs["rho"][0], self.h.chain
        if loc is None or not np.isfinite(rho):
            return 0.0, 0.5
        t = obs["t"]
        dt_now = self.dt if self.t_prev is None else t - self.t_prev      # TORAX's step, which is not always the scenario's dt: a sawtooth crash takes 1 ms
        self.t_prev = t
        self.owed += dt_now
        if obs["P"] > 0 and loc["I_cd"] != 0:
            self.kappa = abs(loc["I_cd"]) / obs["P"]
        left = max(g.units - sum(n for t1, t2, n in g.trips if t1 <= t < t2), 0) * g.unit_MW
        bias = sum(f.get("surface", 0.0) for f in g.inner.faults if f["t1"] <= t < f["t2"])    # the surface the launcher is told
        target = rho + bias
        if self.aim is None:
            self.aim = target
        elif not any(t1 <= t < t2 for t1, t2 in g.stuck):
            self.aim += float(np.clip(target - self.aim, -g.rate * dt_now, g.rate * dt_now))
        off = abs(self.aim - rho) * loc["a"]
        eff = float(np.exp(-4 * (off / 0.084) ** 2))     # a Gaussian deposition off the island, at the chip project's typical deposition width
        n = int(self.owed / 0.02 + 1e-6)                 # the chip's clock follows TORAX's: decisions of 20 ms, as many as this step has time for
        if n < 1:
            self.last = dict(self.last, dt=0.0) if self.last else None
            return (self.last["P_avg"] if self.last else 0.0), self.aim
        self.owed -= n * 0.02
        two_d = self.h.island2d
        a = self.client.ask(op="advance", n=n, plant=self.plant(loc), seed_cm=self.seed_next or None, trigger=self.trigger, brake=self.brake, P_limit=left, eff=eff, noise=self.noise, w_cm=two_d.width_cm if two_d else None)
        self.last = dict(a, dt=n * 0.02)
        self.seed_next, self.trigger, self.brake = 0.0, False, None
        self.w_next = two_d.width(t) if two_d else a["w"]
        self.extra = dict(f_2_1=a["f"], chip_seen=float(a["seen"]), chip_act=float(a["act"]), chip_mod=float(a["mod"]))
        if a["locked"]:
            self.extra["locked_2_1"] = 1.0
            if self.rng.random() < 1 - np.exp(-self.quench_rate * dt_now):
                self.h.reactor.disrupted = True
                self.extra["disrupted"] = 1.0
        return a["P_avg"], self.aim


class Fixed:
    """A set drive, aimed at the measured surface unless told otherwise. P = 0 is no drive at all."""

    def __init__(self, P=0.0, aim=None):
        self.P, self.aim = P, aim

    def __call__(self, obs):
        rho = obs["rho"][0]
        return self.P, self.aim if self.aim is not None else (rho if np.isfinite(rho) else 0.5)


def decision_model(spec, P_max, coeffs, dt=0.5):
    kind = spec.get("kind", "none")
    if kind in ("none", "etched"):                      # the etched chip is attached by Harness
        return Fixed()
    if kind == "fixed":
        return Fixed(spec.get("P", 0.0), spec.get("aim"))
    if kind == "rule":
        return ctl.Tracking(P_max=P_max * spec.get("frac", 1.0), w_on=spec.get("w_on_cm", 3.0) / 100, w_off=spec.get("w_off_cm", 3.0) / 100,
                            hold=spec.get("hold", 3.0), P_idle=P_max * spec.get("idle", 0.0), budget=spec.get("budget"))
    if kind == "phys":
        c = coeffs if spec.get("oracle") else ntm.Coefficients()
        return ctl.PhysMPC(c, P_max=P_max, horizon=spec.get("horizon", 2.0), energy_cost=spec.get("cost", 0.15))
    if kind == "net":
        chip = spec.get("chip", False)
        model = spec.get("model", os.path.join("runs", "torax_shots", "forecaster.npz"))
        model = [fc.load_model(p, k=4) for p in model] if isinstance(model, list) else fc.load_model(model, k=4, hgq=spec.get("hgq_model"))
        return fc.NetMPC(model, P_max=P_max, energy_cost=spec.get("cost", 0.03), chip=chip, dw_steps=max(int(round(0.5 / dt)), 1))
    if kind == "python":
        module, function = spec["target"].rsplit(":", 1)
        if module.endswith(".py"):          # a file of your own, by path: absolute, from the working folder, or from the repository
            import importlib.util
            here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", module)
            path = module if os.path.isabs(module) or os.path.exists(module) else here
            loader = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
            mod = importlib.util.module_from_spec(loader)
            loader.loader.exec_module(mod)
        else:
            mod = importlib.import_module(module)
        return getattr(mod, function)(P_max=P_max, physics=coeffs, **spec.get("args", {}))
    raise ValueError(f"unknown decision model {kind}")


def controller_chain(s, coeffs):
    g, m = s.get("gyrotrons", {}), s.get("sensors", {})
    units, unit = g.get("units", 20), g.get("unit_MW", 1.0)
    decision = decision_model(s.get("decision", {}), units * unit, coeffs, s.get("dt", 0.5))
    ev = s.get("events", [])
    faults = [dict(t1=e["t"], t2=e["t"] + e["duration"], surface=e.get("surface_cm", 0.0) / 200, width=e.get("width_cm", 0.0) / 100)
              for e in ev if e.get("type") == "sensor_bias"]
    sensed = ctl.Measured(ctl.Largest(decision), m.get("sigma_w_cm", 0.5) / 100, m.get("sigma_rho_cm", 0.5) / 200, m.get("latency_steps", 2),
                          seed=m.get("seed", 0), faults=faults)
    return ctl.Gyrotrons(sensed, units=units, unit_MW=unit, mirror_rate=g.get("mirror_rate", 0.1), aim_range=tuple(g.get("aim_range", (0.35, 0.85))),
                         trips=[(e["t"], e["t"] + e["duration"], e["units"]) for e in ev if e.get("type") == "gyrotron_trip"],
                         stuck=[(e["t"], e["t"] + e["duration"]) for e in ev if e.get("type") == "mirror_stuck"], dt=s.get("dt", 0.5))


def plot(s, modes, data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True, height_ratios=[2, 1, 1])
    colors = ["#2a78d1", "#d9711c", "#3a8f4e"]
    t = data["time"]
    for md, col in zip(modes, colors):
        tag = f"{md.m}_{md.n}"
        axes[0].plot(t, 100 * data[f"w_{tag}"], color=col, lw=2, label=f"island {md.m}/{md.n}")
        if f"rho_{tag}" in data:
            axes[2].plot(t, data[f"rho_{tag}"], color=col, lw=1, label=f"q = {md.m}/{md.n} surface")
    axes[0].set_ylabel("island width (cm)")
    axes[0].legend(frameon=False, fontsize=8)
    if "P_cmd" in data:
        axes[1].step(t, data["P_cmd"], where="post", color="#444444", lw=1.5)
        axes[2].plot(t, data["aim_cmd"], color="#444444", lw=1, ls="--", label="ECCD aim")
    axes[1].set_ylabel("ECCD (MW)")
    axes[2].set_ylabel("rho_norm")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(frameon=False, fontsize=8)
    for e in s.get("events", []):
        if "t" in e:
            for ax in axes:
                ax.axvline(e["t"], color="#999999", lw=0.7, ls=":")
            axes[0].annotate(e["type"], (e["t"], axes[0].get_ylim()[1]), rotation=90, fontsize=6, va="top", ha="right", color="#666666")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f'{s.get("name", "twin")}: decision model {s.get("decision", {}).get("kind", "none")}', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def branch_models(h):
    """Every object a step calls besides the harness, the reactor and the plant: the coupled plasma, the free boundary, EPED-NN and any other hook."""
    r, out = h.reactor, []
    for f in list(r.before_step) + list(r.after_solve) + [h.natural, h.free, h.eped]:
        o = getattr(f, "__self__", f)
        if o is None or o is h or o is r or o is h.plant or isinstance(o, (types.FunctionType, types.BuiltinFunctionType)) or not hasattr(o, "__dict__") or any(o is x for x in out):
            continue
        out.append(o)
    return out


def branch_memo(h, models, extra=()):
    """What a branch's copies point to and never copy: the harness, reactor and plant (each restored by its own code), the models themselves (restored in place),
    and modules. Objects that hold a process, a pipe or fixed weights say so themselves (a __deepcopy__ that returns the object)."""
    keep = [h, h.reactor, h.plant, h.plant.beam, getattr(h.plant, "imp", None), h.island2d, h.measured] + list(models) + list(extra)
    memo = {id(x): x for x in keep if x is not None}
    for o in models:
        for v in vars(o).values():
            if isinstance(v, types.ModuleType):
                memo[id(v)] = v
    return memo


def model_state(o, memo):
    """A model's attributes, copied, but for the constants its class names in SNAPSHOT_SHARED; any that cannot be copied are kept as they are, and named."""
    shared = set(getattr(type(o), "SNAPSHOT_SHARED", ()))
    out, uncopied = {}, []
    for k, v in vars(o).items():
        if k in shared:
            continue
        try:
            out[k] = copy.deepcopy(v, memo)
        except Exception:
            out[k] = v
            uncopied.append(f"{type(o).__name__}.{k}")
    return out, uncopied


def set_model_state(o, st, memo):
    """Back to a model_state: attributes made since are removed (the density loop's integral is made on first use), the rest copied in again, so the same snapshot
    can be restored any number of times."""
    shared = set(getattr(type(o), "SNAPSHOT_SHARED", ()))
    for k in [k for k in vars(o) if k not in st and k not in shared]:
        delattr(o, k)
    for k, v in st.items():
        try:
            setattr(o, k, copy.deepcopy(v, memo))
        except Exception:
            setattr(o, k, v)


class Harness:
    """One reactor under a scenario, driven by named commands. The same commands come from a script, from a scenario's events (with a time "t") and from an agent (harness.py), so a test written one way can be
    replayed another. apply() takes a dict with "cmd"; see COMMANDS. Turbulence here is transport-level: multipliers on the transport coefficients against radius and time, steady bursts or a random
    field. TORAX cannot hold real turbulence, which lives at cm and microsecond scales; this is its slow effect on the profiles."""

    COMMANDS = dict(
        step="advance: n steps, or until time t. Returns observe()",
        observe="the plasma now: scalars, islands, drive; profiles=true adds T_e, T_i, n_e, q against rho",
        island="raise an island now: m, n, w_cm (a new mode is followed from here on)",
        profile="multiply a profile in a zone: name (T_e, T_i, n_e), factor, rho [a, b]",
        kick="sudden energy loss: frac of T_e and T_i inside rho [a, b] (a minor disruption, a fast-ion burst)",
        crash="flatten T_e and T_i inside rho [a, b] keeping the energy: a forced sawtooth-like crash",
        elm="an ELM now",
        pedestal="scale the pedestal top, until changed: T, n (a back transition: T 0.5)",
        couple="all the models on one plasma, from now on, with no number fitted to a machine (coupled.py): L-H, the edge, density, ELMs, sawteeth, islands and their seeds, the beta limit. models: which of them; H0: H-mode now; n_line: the density programme in m^-3, a number or {t: value} (without one, the Greenwald fraction it has now is held); machine: its name in machines.py",
        pedestal_model="the pedestal's height and place from EPED-NN, from now on: name (EPED1NNmodel, EPED1NNensemble, or off), interval s",
        quench="what follows the onset. On a MAST shot of the hands-off set, FreeGSNKE from the shot's EFIT at this moment with the run's beta_pol: mode static (default, about a minute): which way the plasma goes once the pressure is gone, from static equilibria (checked on real MAST disruptions), with MAST's measured quench length for that way; growth: 1 adds the vertical growth rate (about 100 s more); mode full: the time-dependent run, hours, with the probes read through it. On other machines, DREAM (1D) in WSL with the run's current, size, field, core temperature and density: material none, neon or neon+deuterium, arrive_ms after the onset (0: with the thermal quench), tpost the held temperature in eV; returns the runaway current and the quench time against ITER's limits. Adds quench_2d or quench_dream to the disruption card",
        free_boundary="plasma shape from FreeGSNKE, from now on: machine (ITER, or MAST on a MAST shot, with the shot's coil traces), mode (control: the coils keep the boundary; coils: the coils stay; off), interval s (default half an energy time, at least ten steps), shape (machine: the programme's in machines.py; plasma: the boundary TORAX holds now; auto, the default: the machine's if the run started from it), growth (also the vertical growth rate, about 25 s; ITER only)",
        turbulence="a burst: multiply transport by factor inside rho [a, b] for duration s; channel heat, e, i, particles or all",
        turbulence_noise="a random transport field: log-normal with sigma, correlation tau s and length in rho; rho [a, b], duration, channel, seed",
        heating="scale the generic heating, until changed: factor",
        brake="multiply a mode's rotation by factor, once: m, n, factor (a torque impulse)",
        error_field="scale the error field, until changed: factor",
        physics="change the island equation's coefficients: a_bs, a_cd, a_ggj, ...",
        trip="lose gyrotrons: units, duration",
        mirror_stuck="freeze the steering mirror: duration",
        sensor_bias="a measurement fault: surface_cm, width_cm, duration",
        decision="change the decision model: spec, as in a scenario",
        drive="set the drive by hand, replacing the decision model: P in MW, aim in rho_norm (optional)",
        set="change a parameter of TORAX live: path (as in its runtime parameters, 'sources.gas_puff.S_total'), value",
        get="read a parameter of TORAX: path",
        snapshot="remember this moment: name",
        restore="go back to a remembered moment: name",
        save="write the traces so far to npz: path",
        help="this list",
    )

    def __init__(self, s, keep_history=True):
        self.s, self.dt = s, s.get("dt", 0.5)
        self.coeffs = ntm.Coefficients(**s.get("physics", {}))
        self.modes = followed_modes(s)
        inner_inject, self.injected = injector(s, self.modes)
        self.etched = None

        def inject(t, widths, events):                   # the chip's loop owns the width of its mode: seeds and triggers go there
            out = inner_inject(t, widths, events) if inner_inject else []
            if self.island2d:
                for k, ws in out:
                    if k == 0:
                        self.island2d.seed(ws * 100)
            if self.etched:
                if events.get("sawtooth") and self.etched.rng.random() < 0.8:      # the chip's trigger input sees 80% of crashes, as in that project
                    self.etched.trigger = True
                for k, ws in out:
                    if k == 0:
                        self.etched.seed_next = max(self.etched.seed_next, ws * 100)
            return out
        e = s.get("elms", True)
        elms = None if e is False else ntm.ELMs(**(e if isinstance(e, dict) else {}))
        self.chain = controller_chain(s, self.coeffs)
        self.measured = None
        if s.get("mast"):                                # a real MAST shot's plasma, from a cut: see mast_anchor.py
            import mast_anchor
            config, self.measured = mast_anchor.scenario_config(s)
        else:
            config = torax_config(s)
        self.reactor = ntm.Reactor(torax.ToraxConfig.from_dict(config), self.modes, coeffs=self.coeffs, controller=self.chain, inject=inject, elms=elms, keep_history=keep_history)
        if s.get("mast") and s["mast"].get("lmode_edge"):      # an L-mode start: no barrier outside the pedestal top, set before the anchoring balances the start (ntm.LMODE_EDGE, H83)
            ntm.LMODE_EDGE[0] = 1.0
        if s.get("mast") and s["mast"].get("noncoronal"):      # non-coronal impurity radiation from the start's own residence times, also before the anchoring (ntm.set_residence, H84)
            ntm.set_residence(self.reactor)
        self.bursts, self.fields, self.snapshots = [], [], {}
        self.plant = actuators.Plant(self, s.get("impurities"))
        if s.get("beams") and self.measured:             # a shot's beams, from its injected power against time
            import beams
            truth = self.measured["truth"]
            self.plant.beam = beams.make(self.reactor, lambda t: float(np.interp(t, truth.t, truth.P_heat)), **(mast_anchor.BEAMS_FROZEN if s["beams"] == "frozen" else {**mast_anchor.BEAMS, **(s["beams"] if isinstance(s["beams"], dict) else {})}))
            self.plant.before_step()                     # the plasma the anchoring balances has its beams on
        if isinstance(s.get("beams"), dict) and not self.measured:      # another machine's beams, from its programme: beams.Beams with no number of MAST's
            import beams
            settings = dict(s["beams"])
            P_MW = settings.pop("P_MW")
            if isinstance(P_MW, dict):
                times, values = np.array(sorted(float(k) for k in P_MW)), np.array([P_MW[k] for k in sorted(P_MW, key=float)], float)
                power = lambda t: float(np.interp(t, times, values))
            else:
                power = lambda t: float(P_MW)
            self.plant.beam = beams.make(self.reactor, power, **settings)
            self.plant.before_step()
        if s.get("mast"):
            if s["mast"].get("table") is not None:       # transport from a fit to other shots: the shot's own power balance is not used
                ntm.TURBULENCE[:] = np.asarray(s["mast"]["table"])
            else:
                mast_anchor.anchor(self.reactor, rates=mast_anchor.measured_rates(self.measured, self.reactor, channels=s["mast"]["anchor_rates"]) if s["mast"].get("anchor_rates") else None, **s["mast"].get("anchor_bounds", {}))
        self.table0 = ntm.TURBULENCE.copy()              # the base of the turbulence table: ones, or a real shot's anchoring
        self.island2d = None
        if s.get("plant2d"):                             # the first mode's width comes from the 2D solver or its surrogate (island2d.py)
            import island2d
            self.island2d = island2d.Plant2D(self, s["plant2d"])
            self.modes[0].prescribed = self.island2d.width
        self.script = sorted((dict(c) for c in s.get("events", []) if "cmd" in c), key=lambda c: c["t"])
        self.reactor.before_step.append(self._tick)
        self.eped, self.free, self.natural = None, None, None
        self.saved_rotation = None
        if s.get("decision", {}).get("kind") == "etched":
            self._attach(s["decision"])

    def _attach(self, spec):
        """The etched chip takes over mode 0: its width and rotation come from the chip's loop, and known seeds become commands."""
        md = self.modes[0]
        self.script = sorted(self.script + [dict(cmd="island", m=md.m, n=md.n, w_cm=100 * w, t=t) for t, w in md.seeds], key=lambda c: c["t"])
        md.seeds = ()
        self.etched = EtchedChip(self, spec)
        md.prescribed, self.saved_rotation = (self.island2d.width if self.island2d else self.etched.width), md.rotation or self.saved_rotation
        md.rotation = None
        self.reactor.controller = self.etched

    def _detach(self):
        if self.etched:
            md = self.modes[0]
            md.prescribed, md.rotation = (self.island2d.width if self.island2d else None), self.saved_rotation
            self.etched = None

    # the transport table, each step
    def _rows(self, channel):
        return dict(heat=[0, 1], e=[0], i=[1], particles=[2], all=[0, 1, 2])[channel]

    def _tick(self, reactor):
        t = reactor.t
        while self.script and self.script[0]["t"] <= t:
            self.apply(self.script.pop(0))
        self.plant.before_step()
        if self.island2d:
            self.island2d.before_step()
        if not self.bursts and not self.fields and np.array_equal(ntm.TURBULENCE, self.table0):
            return
        x = ntm.TURB_RHO
        table = self.table0.copy()
        self.bursts = [b for b in self.bursts if b["until"] > t]
        for b in self.bursts:
            table[b["rows"]] *= 1 + (b["factor"] - 1) * reactor.zone(b["rho"], grid=x, smooth=0.02)
        self.fields = [f for f in self.fields if f["until"] > t]
        for f in self.fields:
            a = np.exp(-self.dt / f["tau"])
            f["xi"] = a * f["xi"] + np.sqrt(1 - a * a) * f["sigma"] * f["shape"] @ f["rng"].standard_normal(len(x))
            table[f["rows"]] *= np.exp(f["xi"] * reactor.zone(f["rho"], grid=x, smooth=0.02))
        ntm.TURBULENCE[:] = table

    def _mode(self, m, n):
        for k, md in enumerate(self.modes):
            if (md.m, md.n) == (m, n):
                return k
        rot = self.modes[0].rotation
        self.modes.append(ntm.Mode(m, n, rotation=dataclasses.replace(rot) if rot else None))
        r = self.reactor
        r.widths.append(0.0)
        r.locs.append(None)
        r.spins.append((2 * np.pi * rot.f0, 0.0) if rot else None)
        r.done_seeds.append(set())
        return len(self.modes) - 1

    def apply(self, c):
        """One command; returns a JSON-ready answer."""
        cmd, r, g = c["cmd"], self.reactor, self.chain
        if cmd in actuators.SPEC:
            return self.plant.command(c)
        if cmd == "_valve":
            return self.plant.valve(c)
        if cmd not in self.COMMANDS:
            return dict(error=f"unknown command {cmd}", commands=sorted(self.COMMANDS) + sorted(actuators.SPEC))
        rho = tuple(c.get("rho", (0.0, 1.0)))
        if cmd == "step":
            n = c.get("n")
            until = c.get("until", c.get("t") if n is None else None)
            while (r.t < until - 1e-9 if until is not None else n > 0) and r.step() is not None:
                n = None if n is None else n - 1
            self.collapsed()
            return self.observe()
        if cmd == "observe":
            return self.observe(c.get("profiles", False))
        if cmd == "island":
            k = self._mode(c.get("m", 2), c.get("n", 1))
            r.widths[k] = max(r.widths[k], c["w_cm"] / 100)
            if self.island2d and k == 0:
                self.island2d.seed(c["w_cm"])
            elif self.etched and k == 0:
                self.etched.seed_next = max(self.etched.seed_next, c["w_cm"])
            self.injected.append(dict(t=r.t, mode=f'{self.modes[k].m}/{self.modes[k].n}', w_cm=c["w_cm"], kind="command"))
        elif cmd == "profile":
            r.edit_profile(c["name"], c["factor"], rho)
        elif cmd == "kick":
            for name in ("T_e", "T_i"):
                r.edit_profile(name, 1 - c["frac"], rho)
        elif cmd == "crash":
            r.flatten_zone(rho)
        elif cmd == "elm":
            r.elm_period = 0.0
        elif cmd == "pedestal":
            r.ped_scale = np.array([c.get("T", r.ped_scale[0]), c.get("n", r.ped_scale[1])], float)
        elif cmd == "pedestal_model":
            import pedestal as pedestal_lib
            if self.eped is not None and self.eped in r.before_step:
                r.before_step.remove(self.eped)
            self.eped = None
            if c.get("name", "EPED1NNmodel") != "off":
                self.eped = pedestal_lib.EPEDPedestal(r, name=c.get("name", "EPED1NNmodel"), interval=c.get("interval", 0.5), mass=c.get("mass"))
                r.before_step.append(self.eped)
        elif cmd == "couple":
            import coupled
            if self.natural is None:
                self.natural = coupled.attach(self, models=tuple(c.get("models", ("hmode", "elms", "sawtooth", "modes", "collapse"))), seed=c.get("seed", 0), H0=c.get("H0", True), n_line=c.get("n_line"), machine=c.get("machine"), follow=c.get("follow"))
        elif cmd == "free_boundary":
            import freeboundary
            if self.free is not None:
                self.free.close()
                self.free = None
            if c.get("mode", "control") != "off":
                if c.get("machine") == "MAST":
                    import natural
                    self.free = natural.free_boundary_now(self, interval=c.get("interval", 0.01))
                else:
                    self.free = freeboundary.Loop(self, machine=c.get("machine", "ITER"), mode=c.get("mode", "control"), interval=c.get("interval"), growth=c.get("growth", False), shape=c.get("shape", "auto"))
        elif cmd == "quench":
            return self.quench_now(c)
        elif cmd == "turbulence":
            self.bursts.append(dict(rows=self._rows(c.get("channel", "heat")), rho=rho, factor=c["factor"], until=r.t + c.get("duration", 1e9)))
        elif cmd == "turbulence_noise":
            x = ntm.TURB_RHO
            length = c.get("length", 0.1)
            shape = np.exp(-0.5 * ((x[:, None] - x[None, :]) / length) ** 2)
            shape /= np.sqrt((shape ** 2).sum(1, keepdims=True))
            self.fields.append(dict(rows=self._rows(c.get("channel", "heat")), rho=rho, sigma=c.get("sigma", 0.5), tau=c.get("tau", 1.0), shape=shape, xi=np.zeros(len(x)),
                                    rng=np.random.default_rng(c.get("seed", 0)), until=r.t + c.get("duration", 1e9)))
        elif cmd == "heating":
            ntm.HEAT[:] = c["factor"]
        elif cmd == "brake":
            k = self._mode(c.get("m", 2), c.get("n", 1))
            if self.etched and k == 0:
                self.etched.brake = c["factor"]
            elif r.spins[k]:
                r.spins[k] = (r.spins[k][0] * c["factor"], r.spins[k][1])
        elif cmd == "error_field":
            for rot in {id(x): x for x in [md.rotation for md in self.modes] + [self.saved_rotation] if x}.values():
                rot.w_vac *= c["factor"]
        elif cmd == "physics":
            for key, v in c.items():
                if key not in ("cmd", "t"):
                    setattr(r.c, key, v)
        elif cmd == "trip":
            g.trips = list(g.trips) + [(r.t, r.t + c.get("duration", 1e9), c.get("units", g.units))]
        elif cmd == "mirror_stuck":
            g.stuck = list(g.stuck) + [(r.t, r.t + c.get("duration", 1e9))]
        elif cmd == "sensor_bias":
            g.inner.faults = list(g.inner.faults) + [dict(t1=r.t, t2=r.t + c.get("duration", 1e9), surface=c.get("surface_cm", 0.0) / 200, width=c.get("width_cm", 0.0) / 100)]
        elif cmd == "decision":
            self._detach()
            r.controller = g
            if c["spec"].get("kind") == "etched":
                self._attach(c["spec"])
            else:
                g.inner.inner = ctl.Largest(decision_model(c["spec"], g.units * g.unit_MW, r.c, self.dt))
        elif cmd == "drive":
            self._detach()
            r.controller = g
            g.inner.inner = ctl.Largest(Fixed(c.get("P", 0.0), c.get("aim")))
        elif cmd == "snapshot":
            name = c.get("name", "a")
            models = branch_models(self)
            memo = branch_memo(self, models)
            keep = list(memo.values())                   # what stays shared, before the copies add their own entries
            states, uncopied = [], []
            for o in models:
                st, bad = model_state(o, memo)
                states.append((o, st))
                uncopied += bad
            free = self.free if self.free is not None and getattr(self.free, "c", None) is not None and not self.free.c.dead else None
            if free is not None:
                free.c("keep", label=name)                # FreeGSNKE's own state: flux, coil and vessel currents, the vessel circuit, the last plasma current
            self.snapshots[name] = dict(models=states, refs=(self.natural, self.free, self.eped), table0=self.table0.copy(), free_kept=free is not None,
                                        keep=keep,
                                        reactor=r.snapshot(), bursts=copy.deepcopy(self.bursts), fields=copy.deepcopy(self.fields), script=copy.deepcopy(self.script),
                                                      n_modes=len(self.modes), rotations=[dataclasses.replace(md.rotation) if md.rotation else None for md in self.modes], injected=len(self.injected), plant=self.plant.state(), island2d=self.island2d.state() if self.island2d else None)
        elif cmd == "restore":
            snap = self.snapshots[c.get("name", "a")]
            self._detach()
            r.restore(snap["reactor"])
            self.chain = r.controller
            self.bursts, self.fields, self.script = copy.deepcopy(snap["bursts"]), copy.deepcopy(snap["fields"]), copy.deepcopy(snap["script"])
            del self.modes[snap["n_modes"]:]
            for md, rot in zip(self.modes, snap["rotations"]):
                md.rotation = dataclasses.replace(rot) if rot else None
            del self.injected[snap["injected"]:]
            self.plant.set_state(snap["plant"])
            if self.island2d:
                self.island2d.set_state(snap["island2d"])
            if "models" in snap:                         # snapshots from package 1.1 on: the models, the transport table, FreeGSNKE
                before = self.free
                self.natural, self.free, self.eped = snap["refs"]
                self.table0 = snap["table0"].copy()
                memo = branch_memo(self, [o for o, _ in snap["models"]], snap["keep"])
                for o, st in snap["models"]:
                    set_model_state(o, st, memo)
                if snap["free_kept"]:
                    if self.free.c.dead:
                        raise RuntimeError("FreeGSNKE's server stopped after this snapshot, so its state cannot come back: take a new snapshot")
                    self.free.c("recall", label=c.get("name", "a"))
                if before is not None and before is not self.free and getattr(before, "c", None) is not None:
                    before.c.close()                     # a free boundary built inside the branch
        elif cmd == "set":
            r.set_param(c["path"], c["value"])
        elif cmd == "get":
            return dict(path=c["path"], value=r.get_param(c["path"]))
        elif cmd == "save":
            keys = sorted({k for rec in r.records for k in rec if k != "t"})
            np.savez(c["path"], time=np.array([rec["t"] for rec in r.records]), **{k: np.array([rec.get(k, np.nan) for rec in r.records], float) for k in keys})
        elif cmd == "help":
            return dict(self.COMMANDS, **{k: f'[{v["level"]}] {v["doc"]} Parameters: {", ".join(v["params"])}' for k, v in actuators.SPEC.items()})
        if cmd == "snapshot" and uncopied:
            return dict(ok=True, cmd=cmd, t=r.t, kept_by_reference=uncopied)
        return dict(ok=True, cmd=cmd, t=r.t)

    def observe(self, profiles=False):
        r = self.reactor
        rec = r.records[-1]
        out = dict(t=r.t, finished=r.finished, disrupted=r.disrupted, P_fusion_MW=rec.get("P_fusion", 0.0) / 1e6, Q=rec.get("Q_fusion"), W_thermal_MJ=rec.get("W_thermal", 0.0) / 1e6, H98=rec.get("H98"),
                   P_SOL_MW=rec.get("P_SOL", 0.0) / 1e6, q_min=float(np.min(np.asarray(r.state.core_profiles.q_face))), P_cmd_MW=rec.get("P_cmd"), aim=rec.get("aim_cmd"), heating_factor=float(ntm.HEAT[0]),
                   pedestal_scale=[float(v) for v in ntm.PEDESTAL], bursts=len(self.bursts), noise_fields=len(self.fields), snapshots=sorted(self.snapshots),
                   islands={f"{md.m}/{md.n}": dict(w_cm=100 * r.widths[k], rho=rec.get(f"rho_{md.m}_{md.n}"), f_hz=rec.get(f"f_{md.m}_{md.n}"), locked=bool(rec.get(f"locked_{md.m}_{md.n}"))) for k, md in enumerate(self.modes)},
                   plant=self.plant.summary(), **{k: rec.get(k) for k in ("P_rad", "P_LH", "beta_N", "f_GW", "q95", "li3", "n_line", "T_e0", "T_i0", "Z_eff_avg", "f_bs")}, sim_error=str(r.sim_error.name) if hasattr(r.sim_error, "name") else str(r.sim_error))
        if self.eped is not None:
            out["eped"] = self.eped.last
        if self.free is not None:
            out["free_boundary"] = self.free.summary()
        if profiles:
            cp, geo = r.state.core_profiles, r.state.geometry
            out["profiles"] = dict(rho=np.asarray(geo.rho_norm).tolist(), T_e=np.asarray(cp.T_e.value).tolist(), T_i=np.asarray(cp.T_i.value).tolist(), n_e=np.asarray(cp.n_e.value).tolist(),
                                   rho_face=np.asarray(geo.rho_face_norm).tolist(), q=np.asarray(cp.q_face).tolist())
        return out

    def quench_now(self, c):
        """The harness's quench command (H78). On a MAST shot of the hands-off set: mast_vde.py quench_at in FreeGSNKE's environment, from the shot's EFIT
        slice nearest before now, its coil traces and the run's beta_pol (TORAX's, thermal: the fast ions' pressure is not in it). mode static (default,
        about a minute): which way the plasma goes, from static equilibria, and MAST's measured quench length for that way; mode full: the time-dependent
        run, hours, with the probes and flux loops read through it. On any other machine: DREAM in WSL (dream_iter.py), with the run's current, size,
        field, core temperature and mean density; material: none, neon or neon+deuterium, arriving arrive_ms after the onset (0: with the thermal quench).
        The result goes on the disruption card as quench_2d or quench_dream."""
        r = self.reactor
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.environ.get("XTORAX_HOME") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the workspace
        shot = (self.measured or {}).get("shot")
        tag = f"{shot or 'run'}_{1e3 * r.t:.1f}ms_{int(time.time())}"
        if shot is not None:
            py = os.path.join(root, "external", "freegsnke-venv", "Scripts", "python.exe") if os.name == "nt" else os.path.join(root, "external", "freegsnke-venv", "bin", "python")
            mode = c.get("mode", "static")
            cmd = [py, "-u", os.path.join(here, "mast_vde.py"), "quench_at", "--shot", str(shot), "--t", repr(float(r.t)), "--betap", repr(float(np.asarray(r.ppo.beta_pol))),
                   "--tag", tag, "--mode", mode, "--growth", str(int(c.get("growth", 0))), "--after", repr(float(c.get("after", 0.004))), "--an_after", repr(float(c.get("an_after", 0.3)))]
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=float(c.get("timeout", 600 if mode == "static" else 14400)), env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            line = [x for x in done.stdout.splitlines() if x.startswith("QUENCH ")]
            if not line:
                raise RuntimeError("quench failed: " + (done.stderr or done.stdout)[-1500:])
            out = json.loads(line[-1][len("QUENCH "):])
            if mode == "full":
                path = os.path.join(root, "runs", "twin", "disruptions", "vde", "harness", f"{tag}.npz")
                out["file"] = path
                try:
                    import sensors
                    geo = json.load(open(os.path.join(root, "runs", "twin", "sensors", "geometry.json")))
                    q = np.load(path)
                    reads = [sensors.read_magnetics(q["psi"][:, :, k].astype(float), q["R"], q["Z"], geo) for k in range(q["psi"].shape[-1])]
                    out["sensors"] = dict(t_ms=q["t_ms"].tolist(), ip=q["Ip"].tolist(), ccbv=[x["ccbv"].tolist() for x in reads], flux=[x["flux"].tolist() for x in reads])
                except Exception as err:
                    out["sensors"] = dict(error=repr(err)[:300])
            key = "quench_2d"
        else:
            geo, cp = r.state.geometry, r.state.core_profiles
            dv = np.asarray(geo.volume_face)
            dv = np.diff(dv) if dv.size == np.asarray(cp.n_e.value).size + 1 else np.ones(np.asarray(cp.n_e.value).size)
            ne = float((np.asarray(cp.n_e.value) * dv).sum() / dv.sum())
            T0 = 1e3 * float(np.asarray(cp.T_e.value)[0])
            tq = getattr(self.plant, "tq_start", None) if self.plant is not None else None
            if tq is not None:                       # the run has collapsed already: the plasma from just before, not one filled with the pellet's electrons
                before = [x for x in r.records if x.get("t", 0.0) < tq["t"] - 1e-9 and x.get("n_line")]
                if before:
                    ne, T0 = float(before[-1]["n_line"]), 1e3 * float(before[-1]["T_e0"])
            a_m = float(geo.a_minor)
            sets = dict(Ip=abs(float(r.get_param("profile_conditions.Ip"))), R0=float(geo.R_major), a=a_m, b=1.075 * a_m, B0=float(geo.B_0), T0=T0, ne=ne)
            material, at = c.get("material", "none"), float(c.get("arrive_ms", 0.0)) / 1e3
            nne, nnd = float(c.get("ne_inj", 1e20 if material == "neon" else 8e18)), float(c.get("nd_inj", 7e20 if material == "neon+deuterium" else 0.0))
            if material == "none":
                args = ["natural", "--tpost", repr(float(c.get("tpost", 10.0)))]
            elif at <= 0:
                args = ["check", "--ne", repr(nne), "--nd", repr(nnd)]
            else:
                args = ["late", "--tpost", repr(float(c.get("tpost", 10.0))), "--arrive", repr(at), "--ne", repr(nne), "--nd", repr(nnd)]
            name = f"harness_{tag}"
            bash = (f"cd {wsl_path(root)} && export XTORAX_HOME={wsl_path(root)} DREAMPATH={os.environ.get('XTORAX_DREAMPATH', '$HOME/DREAM')} PYTHONDONTWRITEBYTECODE=1 && "
                    f"{os.environ.get('XTORAX_DREAM_PYTHON', 'python3')} {wsl_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dream_iter.py'))} "
                    + " ".join(args) + f" --tend {float(c.get('tend', 0.3))} --name {name} --set " + " ".join(f"{k}={v:.6g}" for k, v in sets.items()))
            done = subprocess.run(["wsl.exe", "--exec", "bash", "-lc", bash] if os.name == "nt" else ["bash", "-lc", bash], capture_output=True, text=True, timeout=float(c.get("timeout", 3600)))
            line = [x for x in done.stdout.splitlines() if x.startswith("{")]
            if not line:
                raise RuntimeError("DREAM quench failed: " + (done.stderr or done.stdout)[-1500:])
            out = dict(json.loads(line[-1]), plasma=sets, material=material, arrive_ms=1e3 * at, file=os.path.join(root, "runs", "twin", "disruptions", "iter", name, "summary.json"),
                       limits=dict(cq_ms=[50, 150], runaways_MA=0.15, source="ITER at 15 MA, Vallhagen et al. 2024"))
            key = "quench_dream"
        if self.plant is not None:
            self.plant.card = {**(self.plant.card or {}), key: {k: v for k, v in out.items() if k != "sensors"}}
        return out

    def collapsed(self):
        """TORAX stopped: after a shattered pellet that is the quench it was meant to cause."""
        r = self.reactor
        if r.finished and not r.disrupted and self.plant.tq_start is not None:
            self.plant.end_shot("mitigated")
        elif r.disrupted and self.plant.card is None:
            self.plant.card = self.plant.disruption_card("locked mode")

    def run_to_end(self):
        while self.reactor.step() is not None:
            pass
        return self.reactor.finish()


def run(s, out="runs/twin", tag=None):
    name = s.get("name", "twin") + (f"_{tag}" if tag else "")
    folder = os.path.join(out, name)
    os.makedirs(folder, exist_ok=True)
    t0 = time.perf_counter()
    h = Harness(s)
    modes, injected = h.modes, h.injected
    tree, recs = h.run_to_end()
    wall = time.perf_counter() - t0
    keys = sorted({k for r in recs for k in r if k != "t"})
    data = dict(time=np.array([r["t"] for r in recs]), **{k: np.array([r.get(k, np.nan) for r in recs], float) for k in keys})
    np.savez(os.path.join(folder, "traces.npz"), **data)
    dt = s.get("dt", 0.5)
    sc = tree["scalars"].to_dataset()
    step = np.diff(data["time"], prepend=data["time"][0])     # sawtooth crash steps are shorter
    n_elm, n_st = int(np.nansum(data.get("elm", 0.0))), int(np.nansum(data.get("sawtooth", 0.0)))
    summary = dict(name=name, wall_s=round(wall, 1), steps=len(recs), injected=injected, E_eccd_MJ=float(np.nansum(data.get("P_cmd", np.zeros(1)) * step)),
                   elms=n_elm, sawteeth=n_st, dW_elm_MJ=float(np.nanmean(data["dW_elm_MJ"])) if n_elm else None, f_elm_Hz=n_elm / float(data["time"][-1]) if n_elm else None,
                   q_min_end=float(tree["profiles"].to_dataset()["q"].values[-1].min()),
                   disrupted=bool(recs[-1].get("disrupted")), t_end_s=float(recs[-1]["t"]), mean_P_fusion_MW=float(np.mean(data["P_fusion"]) / 1e6 * data["time"][-1] / s.get("t_final", 150.0)),
                   mean_Q=float(np.mean(data["Q_fusion"])),
                   Q_fusion_end=float(sc["Q_fusion"].values[-1]), H98_end=float(sc["H98"].values[-1]), W_thermal_MJ_end=float(sc["W_thermal_total"].values[-1] / 1e6))
    if "edge" in tree.children:
        e = tree["edge"].to_dataset()
        last = lambda k: float(np.ravel(e[k].values)[-1])
        summary["edge"] = dict(T_e_target_eV=last("T_e_target"), q_perp_target_MW_m2=last("q_perpendicular_target") / 1e6, T_e_separatrix_keV=last("T_e_separatrix"), P_SOL_MW=float(sc["P_SOL_total"].values[-1] / 1e6))
    for md in modes:
        w = 100 * np.nan_to_num(data[f"w_{md.m}_{md.n}"])
        summary[f"{md.m}/{md.n}"] = dict(w_int_cm_s=float(w.sum() * dt), w_max_cm=float(w.max()), t_above_5_s=float((w > 5).sum() * dt), w_end_cm=float(w[-1]))
    json.dump(summary, open(os.path.join(folder, "summary.json"), "w"), indent=1)
    plot(s, modes, data, os.path.join(folder, "traces.png"))
    return summary, tree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--decision", default=None, help="override with a named decision from the scenario's 'decisions' table")
    ap.add_argument("--out", default=os.path.join("runs", "twin"))
    args = ap.parse_args()
    s = json.load(open(args.scenario))
    if args.decision:
        s["decision"] = s["decisions"][args.decision]
    summary, _ = run(s, args.out, args.decision)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
