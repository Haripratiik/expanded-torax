"""The harness's front end (H42): a local web instrument for watching and driving a simulated shot. Serves studio/ and a small JSON API on this machine only (127.0.0.1), named commands only.

One simulation thread owns the harness; HTTP handlers queue commands for it, so a command lands between two TORAX steps. The browser polls /api/trace with the number of records it already has.
  GET  /api/meta           scenarios, commands, actuators and instabilities (actuators.SPEC), decision models, geometry
  GET  /api/trace?since=n&rev=r   records and profile frames since n; a new rev means the timeline was rewound and the client starts over
  POST /api/load           {"scenario": name}: a new plasma (compiles for 20 s or so)
  POST /api/run            {"speed": simulated s per wall s, 0 for flat out, "until": t}
  POST /api/pause, /api/step {"n": 1}
  POST /api/command        any harness command: {"cmd": "spi", ...}; with "t" in the future it is scheduled
  POST /api/branch         {"name": ...}: remember this moment. /api/rewind {"name": ...}: go back to it; what ran since stays as a ghost trace
  POST /api/save           {"name": ...}: write the shot to runs/twin/studio/. GET /api/saved lists them, GET /api/shot?name= returns one
  GET  /api/export         the scenario and every command so far, replayable with harness.py; GET /api/csv the traces
  GET  /api/measured?t=    a real shot's measured points near t, when the scenario is one
Usage (TORAX's venv): python core/studio.py [--scenario studio_iter] [--port 8801]
"""
import argparse
import csv
import io
import json
import mimetypes
import os
import queue
import sys
import threading
import time
import traceback
import urllib.parse
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.environ.get("XTORAX_HOME") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the workspace: runs/twin and external/ (README, "Where things live")
STATIC = os.path.join(HERE, "..", "studio")          # studio
SCENARIOS = os.path.join(HERE, "..", "scenarios")    # scenarios
SAVED = os.path.join(ROOT, "runs", "twin", "studio")
KNOTS = np.linspace(0.0, 1.0, 41)
KEEP = ("P_fusion", "Q_fusion", "W_thermal", "H98", "P_SOL", "P_rad", "P_brems", "P_LH", "beta_N", "f_GW", "q95", "li3", "n_line", "T_e0", "T_i0", "n_e0", "f_bs", "P_alpha", "P_ohmic", "Z_eff_avg", "P_cmd", "aim_cmd", "gas_rate", "seed_rate", "A_rwm", "z_cm",
        "elm", "sawtooth", "H", "dW_elm_MJ", "disrupted", "T_e_target", "q_target", "T_e_sep", "flux_2d", "a_2d", "beam_W_fast", "beam_neutrons", "beam_through", "beam_P_abs")
DECISIONS = {
    "No control": {"kind": "none"},
    "Plain rule (3 cm)": {"kind": "rule", "w_on_cm": 3, "w_off_cm": 3},
    "Physics controller": {"kind": "phys", "oracle": False, "horizon": 2, "cost": 0.15},
    "Forecaster network": {"kind": "net", "cost": 0.03},
    "Forecaster on a chip design": {"kind": "net", "cost": 0.03, "chip": "hgq", "hgq_model": "runs/torax_chip/hgq_forecaster.npz"},
    "Etched chip (WSL)": {"kind": "etched", "final": True, "controller": "sim chip + decision chip", "seed": 1, "cl": -1.0},
    "Etched chip on MAST background (WSL)": {"kind": "etched", "final": True, "controller": "sim chip + decision chip", "seed": 1, "cl": -1.0, "background": "runs/mast_background/bank_dc622_clean.npz"},
}


def _available(spec):
    """A menu entry only when what it needs is there: a forecaster's trained weights in the workspace (not included), or a chip
    server (XTORAX_CHIP_SERVER; the etched chip belongs to a separate project and is not included)."""
    if spec["kind"] == "etched":
        return bool(os.environ.get("XTORAX_CHIP_SERVER"))
    if spec["kind"] == "net":
        paths = [spec.get("model", os.path.join("runs", "torax_shots", "forecaster.npz"))] + ([spec["hgq_model"]] if spec.get("hgq_model") else [])
        return all(os.path.exists(os.path.join(ROOT, path)) for path in paths)
    return True


DECISIONS = {name: spec for name, spec in DECISIONS.items() if _available(spec)}


def q(default, lo=None, hi=None, unit="", doc="", options=None):
    return {k: v for k, v in dict(default=default, min=lo, max=hi, unit=unit, doc=doc, options=options).items() if v not in (None, "")}


# the harness's older commands, described for the console in the same shape as actuators.SPEC
BASE_SPEC = dict(
    island=dict(kind="instability", level="reduced", title="Tearing mode (an island)", doc="Raise an island on the q = m/n surface. From then on it follows the modified Rutherford equation, rotates, can lock, and heat crosses it faster. Any m/n: a new mode is followed from here on.", params=dict(m=q(2, 1, 8), n=q(1, 1, 5), w_cm=q(6.0, 0.5, 60, "cm"))),
    elm=dict(kind="instability", level="prescribed", title="An ELM, now", doc="A crash of the pedestal top, then its recovery, whatever the cycle says.", params={}),
    crash=dict(kind="instability", level="prescribed", title="Forced sawtooth crash", doc="Flatten T_e and T_i inside a radius, keeping the energy. TORAX's own sawtooth model runs by itself where there is a q = 1 surface.", rho_pair=True, params=dict(rho0=q(0.0, 0, 0.5, "rho"), rho1=q(0.4, 0.1, 0.9, "rho"))),
    kick=dict(kind="instability", level="prescribed", title="Minor disruption", doc="A sudden loss of a share of T_e and T_i inside a radius.", rho_pair=True, params=dict(frac=q(0.25, 0, 0.9), rho0=q(0.0, 0, 0.5, "rho"), rho1=q(0.6, 0.1, 1, "rho"))),
    turbulence=dict(kind="instability", level="prescribed", title="Turbulence burst", doc="Transport times a factor in a zone for a time. TORAX cannot hold real turbulence, which lives at cm and microsecond scales; this is its slow effect on the profiles.", rho_pair=True, params=dict(factor=q(3.0, 0.1, 20), rho0=q(0.3, 0, 1, "rho"), rho1=q(0.7, 0, 1, "rho"), duration=q(5.0, 0, 1000, "s"), channel=q("heat", options=["heat", "e", "i", "particles", "all"]))),
    turbulence_noise=dict(kind="instability", level="prescribed", title="Turbulent field", doc="A random transport field in radius and time, log-normal.", rho_pair=True, params=dict(sigma=q(0.5, 0, 2), tau=q(1.0, 0.01, 50, "s"), length=q(0.1, 0.01, 0.5, "rho"), rho0=q(0.0, 0, 1, "rho"), rho1=q(1.0, 0, 1, "rho"), duration=q(20.0, 0, 1000, "s"), channel=q("heat", options=["heat", "e", "i", "particles", "all"]), seed=q(0, 0, 9999))),
    pedestal=dict(kind="instability", level="prescribed", title="Set the pedestal", doc="Scale the pedestal top until changed.", params=dict(T=q(1.0, 0.1, 1.5), n=q(1.0, 0.1, 1.5))),
    pedestal_model=dict(kind="actuator", level="coupled", title="Pedestal from EPED-NN", doc="The pedestal's height and place from EPED-NN, the open-source network over the EPED model, from now on, from the plasma as it stands. It reports how far outside its training range the plasma is; for a spherical tokamak, far.", params=dict(name=q("EPED1NNmodel", doc="which network, or off", options=["EPED1NNmodel", "EPED1NNensemble", "off"]), interval=q(0.5, 0.01, 10, "s"))),
    free_boundary=dict(kind="actuator", level="coupled", title="Plasma shape from FreeGSNKE", doc="A free-boundary equilibrium with the machine's coils and vessel, every interval, in place of held surfaces. Control: the coils move to keep the boundary, and their currents are the answer. Coils: the coils stay and the plasma goes where the fields send it. Finding the coils takes about a minute at the start; the run pauses for it.", params=dict(machine=q("ITER", doc="a machine FreeGSNKE describes", options=["ITER"]), mode=q("control", doc="what moves", options=["control", "coils", "off"]), interval=q(1.0, 0.05, 10, "s"), growth=q(0, 0, 1, doc="1: also the vertical growth rate, about 25 s, once"))),
    brake=dict(kind="instability", level="reduced", title="Torque on a mode", doc="Multiply a mode's rotation once.", params=dict(m=q(2, 1, 8), n=q(1, 1, 5), factor=q(0.5, 0, 1))),
    error_field=dict(kind="instability", level="reduced", title="Error field", doc="Scale the error field until changed: it brakes islands and locks them sooner.", params=dict(factor=q(2.0, 0, 10))),
    trip=dict(kind="fault", level="margin", title="Gyrotron trip", doc="Lose gyrotrons for a time.", params=dict(units=q(10, 1, 26), duration=q(3.0, 0, 1000, "s"))),
    mirror_stuck=dict(kind="fault", level="margin", title="Stuck mirror", doc="The launcher's steering mirror stops following.", params=dict(duration=q(3.0, 0, 1000, "s"))),
    sensor_bias=dict(kind="fault", level="margin", title="Sensor fault", doc="A false reading of the surface's position or the island's width.", params=dict(surface_cm=q(2.0, -20, 20, "cm"), width_cm=q(0.0, -20, 20, "cm"), duration=q(5.0, 0, 1000, "s"))),
    drive=dict(kind="actuator", level="torax", title="Gyrotrons by hand", doc="Set the current drive by hand, replacing the decision model. Choose a model in the decision loop panel to go back.", params=dict(P=q(20.0, 0, 26, "MW"), aim=q(0.64, 0.3, 0.95, "rho"))),
    heating=dict(kind="actuator", level="torax", title="Scale the scenario's heating", doc="Multiply the scenario's heating, until changed.", params=dict(factor=q(1.0, 0, 3))),
)


def clean(x):
    """JSON has no NaN; numbers are shortened, which is most of the size."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        x = float(x)
        if x != x or x in (float("inf"), float("-inf")):
            return None
        return float(f"{x:.5g}")
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


class Studio:
    def __init__(self, scenario):
        self.jobs = queue.Queue()
        self.h, self.state, self.message = None, "loading", "starting"
        self.running, self.speed, self.until = False, 1.0, None
        self.frames, self.rows, self.commands, self.ghosts, self.marks = [], [], [], [], {}
        self.rev, self.scenario_name, self.geometry, self.lock = 0, scenario, None, threading.Lock()
        self.machine, self.counts = None, dict(eped=0, last_eped=None)
        self.wall_per_step = 0.0
        threading.Thread(target=self.loop, daemon=True).start()
        self.call("load", scenario, wait=False)

    def call(self, what, *args, wait=True):
        f = Future()
        self.jobs.put((what, args, f))
        return f.result(timeout=600) if wait else f

    # the simulation thread
    def loop(self):
        last = time.perf_counter()
        while True:
            try:
                what, args, f = self.jobs.get(timeout=0 if self.running else 0.05)
                try:
                    f.set_result(getattr(self, "job_" + what)(*args))
                except Exception as err:
                    traceback.print_exc()
                    f.set_result(dict(error=repr(err)[:500]))
                    if what == "load":
                        self.state, self.message = "ended", f"could not load: {repr(err)[:300]}"
                continue
            except queue.Empty:
                pass
            if not self.running or self.h is None:
                continue
            r = self.h.reactor
            if self.until is not None and r.t >= self.until - 1e-9:
                self.running, self.until = False, None
                self.state = "paused"
                continue
            t0 = time.perf_counter()
            before = r.t
            alive = r.step() is not None
            self.h.collapsed()
            self.record()
            if not alive or r.finished:
                self.running = False
                self.state = "ended"
                self.message = (self.h.plant.card or {}).get("kind") or str(getattr(r.sim_error, "name", ""))
                continue
            self.wall_per_step = time.perf_counter() - t0
            if self.speed > 0:                         # hold to the asked pace; the browser's rate of polling is unaffected
                wait = (r.t - before) / self.speed - self.wall_per_step
                if wait > 0:
                    time.sleep(min(wait, 0.5))

    def job_load(self, name):
        import actuators
        import twin
        self.running, self.state, self.message = False, "loading", f"building {name}"
        s = json.load(open(os.path.join(SCENARIOS, name + ".json")))
        if self.h is not None and self.h.free is not None:
            self.h.free.close()
        self.frames, self.rows, self.commands, self.ghosts, self.marks = [], [], [], [], {}
        self.machine, self.counts = None, dict(eped=0, last_eped=None)
        self.rev += 1
        self.scenario_name, self.scenario = name, s
        self.decision_name = next((k for k, v in DECISIONS.items() if v.get("kind") == s.get("decision", {}).get("kind")), s.get("decision", {}).get("kind"))
        if s.get("hands_off"):                         # a real shot from its start, by itself: natural.py's run, stepped from here
            import natural
            self.h = natural.hands_off(**s["hands_off"])
        else:
            self.h = twin.Harness(s, keep_history=False)
        self.geometry = self.read_geometry()
        self.record_all()
        self.message = "compiling TORAX's step, about 20 s"
        settle = s.get("settle", 0.0)
        r = self.h.reactor
        while True:
            if r.step() is None:
                break
            self.record()
            if settle and self.message.startswith("compiling"):
                self.message = f"settling to {settle:g} s"
            if r.t >= settle - 1e-9:
                break
        self.h.collapsed()
        self.state, self.message = ("ended" if r.finished else "paused"), "ready"
        return dict(ok=True, spec=list(actuators.SPEC))

    def job_command(self, c):
        h, r = self.h, self.h.reactor
        c = dict(c)
        if c.get("cmd") == "decision":
            self.decision_name = next((k for k, v in DECISIONS.items() if v == c.get("spec")), str(c.get("spec")))
        if c.get("cmd") == "drive":
            self.decision_name = "by hand"
        if c.get("t") is not None and c["t"] > r.t:
            h.script = sorted(h.script + [c], key=lambda x: x["t"])
            answer = dict(ok=True, scheduled=c["t"])
        else:
            c.pop("t", None)
            answer = h.apply(c)
        self.commands.append(dict(t=r.t, command=c, answer={k: v for k, v in answer.items() if k not in ("ok", "cmd", "t")} if isinstance(answer, dict) else {}))
        self.record()
        return clean(answer)

    def job_run(self, speed, until):
        if self.state == "ended":
            return dict(error="the shot has ended: rewind to a branch, or load a new plasma")
        self.speed, self.until, self.running, self.state = speed, until, True, "running"
        return dict(ok=True)

    def job_pause(self):
        self.running = False
        self.state = "paused" if self.state != "ended" else "ended"
        return dict(ok=True)

    def job_step(self, n):
        r = self.h.reactor
        for _ in range(n):
            if r.step() is None:
                break
            self.h.collapsed()
            self.record()
        if r.finished:
            self.state = "ended"
            self.message = (self.h.plant.card or {}).get("kind") or str(getattr(r.sim_error, "name", ""))
        return dict(ok=True, t=r.t)

    def job_branch(self, name):
        if getattr(self.h, "natural", None) is not None:
            return dict(error="no branches on a hands-off shot yet: its laws and its vessel currents carry state a snapshot does not hold")
        self.h.apply(dict(cmd="snapshot", name=name))
        self.marks[name] = dict(t=self.h.reactor.t, n_rows=len(self.rows), n_frames=len(self.frames), n_commands=len(self.commands))
        return dict(ok=True, t=self.h.reactor.t, branches=self.branch_list())

    def job_rewind(self, name):
        m = self.marks[name]
        self.running = False
        self.ghosts.append(dict(name=f"{name} run {len(self.ghosts) + 1}", rows=self.rows[m["n_rows"]:][::max(1, (len(self.rows) - m["n_rows"]) // 1500)], card=self.h.plant.card))
        self.h.apply(dict(cmd="restore", name=name))
        with self.lock:
            del self.rows[m["n_rows"]:]
            del self.frames[m["n_frames"]:]
            del self.commands[m["n_commands"]:]
            self.rev += 1
        self.state, self.message = "paused", f"back at {name}"
        return dict(ok=True, t=self.h.reactor.t)

    def job_save(self, name):
        os.makedirs(SAVED, exist_ok=True)
        path = os.path.join(SAVED, "".join(ch for ch in name if ch.isalnum() or ch in "-_") + ".json")
        json.dump(dict(name=name, scenario=self.scenario_name, rows=self.rows, frames=self.frames[::max(1, len(self.frames) // 800)], commands=self.commands, card=self.h.plant.card, geometry=self.geometry), open(path, "w"))
        return dict(ok=True, path=os.path.relpath(path, ROOT))

    def read_geometry(self):
        geo = self.h.reactor.state.geometry
        rho = np.asarray(geo.rho_norm)
        rf = np.asarray(geo.rho_face_norm)
        pick = lambda v, x=rho: np.interp(KNOTS, x, np.asarray(v))
        kappa = pick(geo.elongation_face, rf) if hasattr(geo, "elongation_face") else np.full_like(KNOTS, 1.7)
        delta = pick(geo.delta_face, rf) if hasattr(geo, "delta_face") else np.zeros_like(KNOTS)
        return clean(dict(rho=KNOTS, R_in=pick(geo.R_in), R_out=pick(geo.R_out), kappa=kappa, delta=delta, R0=float(geo.R_major), a=float(geo.a_minor), B0=float(geo.B_0)))

    def record_all(self):
        h = self.h
        have = len(self.rows)
        for rec in h.reactor.records[have:]:
            self.rows.append(self.row(rec))
        self.frames.append(self.frame())

    def row(self, rec):
        out = dict(t=rec["t"])
        for k, v in rec.items():
            if k in KEEP or k.startswith(("w_", "rho_", "f_", "locked_")) and not k.startswith(("w_d_", "w_dep_", "rho_dep_", "f_GW", "f_bs")):
                out[k] = v
        return clean(out)

    def record(self):
        r = self.h.reactor
        if len(r.records) > len(self.rows):
            self.edge_values(r.records[-1])
            with self.lock:
                for rec in r.records[len(self.rows):]:
                    self.rows.append(self.row(rec))
                self.frames.append(self.frame())

    def edge_values(self, rec):
        e = getattr(self.h.reactor.state, "edge_outputs", None)
        if e is not None:
            rec.update(T_e_target=float(np.ravel(e.T_e_target)[0]), q_target=float(np.ravel(e.q_perpendicular_target)[0]) / 1e6, T_e_sep=float(np.ravel(e.T_e_separatrix)[0]))

    def frame(self):
        h, r = self.h, self.h.reactor
        st = r.state
        cp, geo, tr = st.core_profiles, st.geometry, st.core_transport
        rho, rf = np.asarray(geo.rho_norm), np.asarray(geo.rho_face_norm)
        on = lambda v, x=rho: np.interp(KNOTS, x, np.asarray(v))
        f = dict(t=r.t, T_e=on(cp.T_e.value), T_i=on(cp.T_i.value), n_e=on(cp.n_e.value) / 1e20, q=on(cp.q_face, rf), chi_e=on(tr.chi_face_el, rf), chi_i=on(tr.chi_face_ion, rf), Z_eff=on(cp.Z_eff), j=on(cp.j_total) / 1e6)
        src = st.core_sources.T_e
        for name, key in (("impurity_radiation", "rad"), ("ecrh", "ecrh")):
            if name in src:
                f[key] = on(src[name]) / 1e6
        if h.plant.imp is not None:
            f["imp"] = {s: on(h.plant.imp.excess[s]) / 1e19 for s in h.plant.imp.species if np.any(h.plant.imp.excess[s] > 0)}
        f["islands"] = [dict(m=md.m, n=md.n, w=r.widths[k], rho=(r.locs[k] or {}).get("rho"), phi=r.spins[k][1] if r.spins[k] else 0.0, f=r.spins[k][0] / (2 * np.pi) if r.spins[k] else None) for k, md in enumerate(h.modes)]
        g = h.chain
        f["shards"] = [dict(x=s["x"], r=s["r"]) for s in h.plant.shards]
        lost = sum(n for t1, t2, n in g.trips if t1 <= r.t < t2)
        f["gyro"] = dict(units=g.units, unit_MW=g.unit_MW, lost=lost, stuck=any(t1 <= r.t < t2 for t1, t2 in g.stuck), P=r.records[-1].get("P_cmd", 0.0), aim=r.records[-1].get("aim_cmd"))
        m = g.inner
        f["sensors"] = dict(sigma_w_cm=100 * m.sw, sigma_rho=m.sr, latency=m.queue.maxlen - 1, faults=sum(1 for x in m.faults if x["t1"] <= r.t < x["t2"]))
        f["turb"] = [float(np.max(ntm_turbulence()[i])) for i in range(3)]
        f["heat"] = float(heat_factor())
        if h.free is not None:                          # the shape moves now: each frame carries its own
            if self.machine is None:
                self.machine = h.free.c("drawing")
            f["shape"] = self.read_geometry()
            sm = h.free.summary()
            f["shape"].update(coils_A=sm.get("coils_A"), lcfs=sm["lcfs"], measured_lcfs=sm.get("measured_lcfs"), Z_axis=sm["shape"]["Z_axis"], coils_moved_kA={k: v / 1e3 for k, v in sm["coils_moved_A"].items() if abs(v) > 10.0}, rejected=sm["rejected"], stability=sm["stability"])
        if h.eped is not None and h.eped.last:
            f["eped"] = {k: h.eped.last.get(k) for k in ("p_kPa", "width", "T_e_ped", "n_e_ped", "rho_top", "extrapolation", "worst")}
        return clean(f)

    def branch_list(self):
        return [dict(name=k, t=v["t"]) for k, v in self.marks.items()]

    # read by the HTTP threads
    def trace(self, since, rev, frames_since):
        with self.lock:
            full = rev != self.rev
            a, b = (0, 0) if full else (since, frames_since)
            out = dict(rev=self.rev, state=self.state, message=self.message, running=self.running, speed=self.speed, scenario=self.scenario_name, n=len(self.rows), n_frames=len(self.frames), rows=self.rows[a:], frames=self.frames[b:], branches=self.branch_list(),
                       wall_per_step=self.wall_per_step, has_machine=self.machine is not None)
        h = self.h
        if h is not None:
            r = h.reactor
            out.update(t=r.t, decision_name=self.decision_name, commands=self.commands[-200:], ghosts=[dict(name=g["name"], n=len(g["rows"])) for g in self.ghosts], plant=clean(h.plant.summary()), events=clean(h.plant.event_log[-200:]), injected=clean(h.injected[-100:]),
                       decision=h.s.get("decision", {}).get("kind"), scheduled=clean([c for c in h.script if not str(c.get("cmd", "")).startswith("_")][:20]), bursts=len(h.bursts), fields=len(h.fields), params=self.params(), coupling=self.coupling())
        return out

    def coupling(self):
        """What the backplane on the page shows: each model, whether it is on, how many times it has passed data, what it reads now."""
        h, r = self.h, self.h.reactor
        with self.lock:
            rec = dict(self.rows[-1]) if self.rows else {}
            frame = self.frames[-1] if self.frames else {}
        get = lambda k, d=0.0: rec.get(k) if rec.get(k) is not None else d
        n = len(self.rows)
        out = dict(state=f"T_e {get('T_e0'):.3g} keV   T_i {get('T_i0'):.3g} keV   n {get('n_line') / 1e20:.3g}e20   W {get('W_thermal') / 1e6:.3g} MJ   beta_N {get('beta_N'):.3g}   q95 {get('q95'):.3g}")
        beam, imp = h.plant.beam, h.plant.imp
        loop = getattr(h, "natural", None)
        st = getattr(loop, "status", None) or {}
        out["programme"] = dict(on=True, n=n, a=f"Ip {abs(float(np.asarray(r.state.core_profiles.Ip_profile_face)[-1])) / 1e6:.3g} MA", b=f"heat out {(get('P_SOL') + get('P_rad')) / 1e6:.3g} MW")
        out["beams"] = dict(on=beam is not None, n=n, a=f"{get('beam_P_abs') / 1e6:.3g} MW absorbed", b=f"fast {get('beam_W_fast') / 1e3:.3g} kJ") if beam is not None else dict(on=False, a="off")
        gas = get("gas_rate")
        fn = getattr(getattr(h, "natural", None), "fn", None)
        out["fuel"] = dict(on=gas > 0 or fn is not None, n=n if gas > 0 else None, a=f"gas {gas:.3g} /s" if gas else (f"programme x {fn:.2f}" if fn is not None else ""))
        out["impurities"] = dict(on=imp is not None or get("Z_eff_avg") > 1.01, a=f"Z_eff {get('Z_eff_avg'):.3g}", b=f"radiated {get('P_rad') / 1e6:.3g} MW")
        if h.free is not None:
            sm = h.free.summary()
            out["free"] = dict(on=True, n=sm["solves"], flag="warn" if sm["rejected"] else None, a=f"kappa {sm['shape']['kappa']:.3g}  R {sm['shape']['R_out']:.3g}", b=f"{sm['solves']} solves, {sm['rejected']} no")
        else:
            out["free"] = dict(on=False, a="prescribed surfaces")
        if st.get("eped") and h.eped is None:
            e = st["eped"]
            self.counts["eped"] += 1
            out["eped"] = dict(on=True, n=self.counts["eped"], flag="warn" if e.get("extrapolation", 0) > 0 else None, a=f"limit {e['p'] / 1e3:.3g} kPa", b=f"{e.get('extrapolation', 0):.2g} out of range" if e.get("extrapolation", 0) > 0 else "in range")
        elif h.eped is not None and h.eped.last:
            e = h.eped.last
            if e.get("t") != self.counts["last_eped"]:
                self.counts.update(last_eped=e.get("t"), eped=self.counts["eped"] + 1)
            out["eped"] = dict(on=True, n=self.counts["eped"], flag="warn" if e.get("extrapolation", 0) > 0 else None, a=f"p {e['p_kPa']:.3g} kPa", b=f"width {e['width']:.2g}" if e.get("width") else "")
        else:
            out["eped"] = dict(on=False, a="set by the scenario")
        g = frame.get("gyro") or {}
        out["gyro"] = dict(on=(g.get("P") or 0) > 0, n=n if (g.get("P") or 0) > 0 else None, a=f"{g.get('P', 0):.3g} MW" + (f" at {g['aim']:.2f}" if g.get("aim") is not None else ""), b=f"{g.get('lost', 0)} tripped" if g.get("lost") else "")
        P_LH, P_SOL = get("P_LH"), get("P_SOL")
        H = rec.get("H")
        if st.get("P_LH"):
            P_SOL, P_LH = st["P_net"], st["P_LH"]
        out["lh"] = dict(on=H is not None or P_LH > 0, n=len(getattr(loop, "h_log", []) or []), a=f"P/P_LH {P_SOL / P_LH:.3g}" if P_LH > 0 else "", b=("H-mode" if H else "L-mode") if H is not None else "")
        elms = len(getattr(r, "elm_times", []) or [])
        out["elms"] = dict(on=r.elms is not None, n=elms, a=f"{elms} crashes" if r.elms is not None else "off")
        crashes = sum(1 for x in self.rows if x.get("sawtooth"))
        saw_on = r.provider.mhd.sawtooth is not None
        out["saw"] = dict(on=saw_on, n=crashes, a=f"{crashes} crashes" if saw_on else "off", b=f"q=1 at {get('rho_q1'):.2f}" if get("rho_q1") else "")
        if st.get("ratio") is not None:
            out["limit"] = dict(on=True, n=st.get("collapses", 0), flag="warn" if st["ratio"] > 0.9 * (st.get("x_crit") or 1.0) else None, a=f"beta / 4 li {st['ratio']:.2f}", b=f"{st.get('collapses', 0)} collapses")
        else:
            out["limit"] = dict(on=False, a="not in this loop")
        modes = [i for i in frame.get("islands") or [] if (i.get("w") or 0) > 0]
        out["islands"] = dict(on=bool(modes), n=n if modes else None, flag="off-limit" if any(i.get("f") is not None and abs(i["f"]) < 1 for i in modes) else None, a="; ".join(f"{i['m']}/{i['n']} {100 * i['w']:.2g} cm" for i in modes[:2]) if modes else "none", b=f"{modes[0]['f'] / 1e3:.3g} kHz" if modes and modes[0].get("f") is not None else "")
        s = frame.get("sensors") or {}
        out["sensors"] = dict(on=True, a=f"noise {s.get('sigma_w_cm', 0):.2g} cm", b=f"delay {s.get('latency', 0)}" + (f", {s['faults']} fault" if s.get("faults") else ""))
        kind = h.s.get("decision", {}).get("kind", "none")
        out["decision"] = dict(on=kind != "none", n=n if kind != "none" else None, a=str(self.decision_name or kind)[:22])
        return clean(out)

    def params(self):
        r = self.h.reactor
        out = {}
        for path in ("profile_conditions.Ip", "sources.generic_heat.P_total", "sources.gas_puff.S_total", "pedestal.T_e_ped", "pedestal.n_e_ped"):
            try:
                out[path] = r.get_param(path)
            except Exception:
                pass
        return clean(out)


def ntm_turbulence():
    import ntm
    return ntm.TURBULENCE


def heat_factor():
    import ntm
    return ntm.HEAT[0]


def serve(studio, port):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, body, kind="application/json", code=200):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            try:
                self.end_headers()
                self.wfile.write(body)
            except OSError:         # the browser went away
                self.close_connection = True

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            query = dict(urllib.parse.parse_qsl(url.query))
            try:
                if url.path == "/" or not url.path.startswith("/api/"):
                    name = "index.html" if url.path == "/" else os.path.normpath(url.path.lstrip("/"))
                    path = os.path.join(STATIC, name)
                    if not os.path.abspath(path).startswith(os.path.abspath(STATIC)) or not os.path.isfile(path):
                        return self._send(dict(error="not found"), code=404)
                    return self._send(open(path, "rb").read(), mimetypes.guess_type(path)[0] or "application/octet-stream")
                if url.path == "/api/meta":
                    import actuators
                    import twin
                    return self._send(dict(scenarios=sorted(f[:-5] for f in os.listdir(SCENARIOS) if f.endswith(".json")), spec={**BASE_SPEC, **(actuators.spec_for(studio.h) if studio.h is not None else actuators.SPEC)}, commands=twin.Harness.COMMANDS, decisions=DECISIONS, geometry=studio.geometry, scenario=studio.scenario_name, knots=KNOTS.tolist(),
                                           species=sorted(studio.h.plant.imp.species) if studio.h is not None and studio.h.plant.imp else [], measured=bool(studio.h and studio.h.measured), machine=studio.machine))
                if url.path == "/api/trace":
                    whole = lambda k, d: int(query[k]) if str(query.get(k, "")).lstrip("-").isdigit() else d     # a page that lost its place gets everything again
                    return self._send(studio.trace(whole("since", 0), whole("rev", -1), whole("frames", 0)))
                if url.path == "/api/ghost":
                    return self._send(studio.ghosts[int(query["i"])])
                if url.path == "/api/saved":
                    return self._send(sorted(f[:-5] for f in os.listdir(SAVED) if f.endswith(".json")) if os.path.isdir(SAVED) else [])
                if url.path == "/api/shot":
                    name = "".join(ch for ch in query["name"] if ch.isalnum() or ch in "-_")
                    return self._send(open(os.path.join(SAVED, name + ".json"), "rb").read())
                if url.path == "/api/export":
                    return self._send(dict(scenario=studio.scenario, commands=[dict(c["command"], t=c["t"]) for c in studio.commands]))
                if url.path == "/api/csv":
                    keys = sorted({k for row in studio.rows for k in row})
                    buf = io.StringIO()
                    w = csv.DictWriter(buf, fieldnames=keys)
                    w.writeheader()
                    w.writerows(studio.rows)
                    return self._send(buf.getvalue().encode(), "text/csv")
                if url.path == "/api/measured":
                    return self._send(measured(studio, float(query.get("t", 0.0))))
                if url.path == "/api/measured_traces":
                    return self._send(measured_traces(studio))
                return self._send(dict(error="unknown"), code=404)
            except Exception as err:
                traceback.print_exc()
                return self._send(dict(error=repr(err)), code=500)

        def do_POST(self):
            try:
                c = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                path = urllib.parse.urlparse(self.path).path
                if path == "/api/load":
                    studio.call("load", c["scenario"], wait=False)
                    return self._send(dict(ok=True))
                if path == "/api/run":
                    return self._send(studio.call("run", float(c.get("speed", 1.0)), c.get("until")))
                if path == "/api/pause":
                    return self._send(studio.call("pause"))
                if path == "/api/step":
                    return self._send(studio.call("step", int(c.get("n", 1))))
                if path == "/api/command":
                    return self._send(studio.call("command", c))
                if path == "/api/branch":
                    return self._send(studio.call("branch", c["name"]))
                if path == "/api/rewind":
                    return self._send(studio.call("rewind", c["name"]))
                if path == "/api/save":
                    return self._send(studio.call("save", c["name"]))
                return self._send(dict(error="unknown"), code=404)
            except Exception as err:
                traceback.print_exc()
                return self._send(dict(error=repr(err)), code=500)

        def log_message(self, *args):
            pass

    print(f"studio at http://127.0.0.1:{port}/", flush=True)

    class Server(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            if isinstance(sys.exc_info()[1], (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
                return                                  # the page closed or reloaded with a request open: nothing went wrong here
            super().handle_error(request, client_address)

    Server(("127.0.0.1", port), Handler).serve_forever()


def measured(studio, t):
    """A real shot's raw points nearest in time: Thomson T_e and n_e, charge exchange T_i, against rho_norm."""
    if not studio.h or not studio.h.measured:
        return {}
    g = studio.h.measured["g"]
    k = int(np.argmin(np.abs(g["t"] - t)))
    out = dict(shot=studio.h.measured["shot"], t=float(g["t"][k]), T_e=[], n_e=[], T_i=[])
    if abs(g["t"][k] - t) > 0.006:
        return out
    r, te, ne = g["ts_rho"][k], g["ts_te"][k], g["ts_ne"][k]
    keep = np.isfinite(r) & np.isfinite(te) & (te > 0) & (r <= 1.05)
    scale = 1e3 if np.nanmax(te) > 50 else 1.0
    out["T_e"] = clean(np.stack([r[keep], te[keep] / scale], 1))
    keep_n = np.isfinite(r) & np.isfinite(ne) & (ne > 0) & (r <= 1.05)
    out["n_e"] = clean(np.stack([r[keep_n], ne[keep_n] / 1e20], 1))
    r, ti = g["cx_rho"][k], g["cx_ti"][k]
    keep = np.isfinite(r) & np.isfinite(ti) & (ti > 0) & (r <= 1.05)
    scale = 1e3 if keep.any() and np.nanmax(ti[keep]) > 50 else 1.0
    out["T_i"] = clean(np.stack([r[keep], ti[keep] / scale], 1))
    return out


def measured_traces(studio):
    """A real shot's traces to set beside the run, keyed as the run's records are, in the same units."""
    if not studio.h or not studio.h.measured:
        return {}
    g, shot = studio.h.measured["g"], studio.h.measured["shot"]
    truth = studio.h.measured["truth"]
    t = g["t"]
    rho = truth.rho
    out = dict(t=clean(t), W_thermal=clean(g["W_th"]), T_e0=clean(truth.T_e[:, 0]), n_line=clean(g["nel"] / 1.0), W_dia=clean(g["W_dia"]))
    if truth.T_i is not None:
        out["T_i0"] = clean(truth.T_i[:, 0])
    extra = os.path.join(ROOT, "runs", "twin", "mast_natural", "extras", f"{shot}.npz")
    if os.path.exists(extra):
        x = np.load(extra)
        keep = (x["t_sum"] >= t[0] - 0.05) & (x["t_sum"] <= t[-1] + 0.05)
        every = max(1, int(keep.sum() // 300))
        out["t_sum"] = clean(x["t_sum"][keep][::every])
        for key, name in (("neutrons", "neutron_rates_total"), ("P_rad", "power_radiated"), ("n_line_sum", "line_average_n_e")):
            if name in x:                              # not every shot has them all (28825 has no bolometer)
                out[key] = clean(x[name][keep][::every])
    del rho
    if getattr(studio.h, "natural", None) is not None:
        import natural
        wins = natural.catalog_shot(shot)["phases"]["hmode"]["windows"]
        out["H"] = clean(np.array([float(any(w["t0"] <= x <= w["t1"] for w in wins)) for x in t]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="studio_iter")
    ap.add_argument("--port", type=int, default=8801)
    args = ap.parse_args()
    serve(Studio(args.scenario), args.port)


if __name__ == "__main__":
    main()
