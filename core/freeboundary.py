"""TORAX and FreeGSNKE together (H46): TORAX evolves heat, particles and current on flux surfaces; FreeGSNKE (a separate process, freegsnke_server.py) finds the free-boundary equilibrium those surfaces belong to, from the coils and the plasma TORAX describes. TORAX holds its geometry fixed on its own, so this is what makes plasma shape and position respond to the plasma and the coils.
The loop, every interval: TORAX's p', FF' against normalized flux and its plasma current go to FreeGSNKE; FreeGSNKE solves; its flux map returns as a TORAX geometry through TORAX's own EQDSK reader, and TORAX's step takes it through geo_overrides, interpolating in time so the change of toroidal flux is counted.
Nothing here is machine-specific but the machine description FreeGSNKE reads.
Usage (TORAX's venv): python core/freeboundary.py iter_check
"""
import argparse
import dataclasses
import json
import os
import pickle
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import machines  # noqa: E402
ROOT = os.path.abspath(os.environ.get("XTORAX_HOME") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the workspace: runs/twin and external/ (README, "Where things live")
PYTHON = os.environ.get("XTORAX_FREEGSNKE_PYTHON") or os.path.join(ROOT, "external", "freegsnke-venv", *(("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")))   # FreeGSNKE's own environment
SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "freegsnke_server.py")
OUT = os.path.join(ROOT, "runs", "twin", "freeboundary")


REPLY_LIMIT_S = 300.0      # the server's own limit (SOLVE_LIMIT_S, 240 s) works inside the solve and cannot stop a loop in compiled code: MAST 18886 spun for 8 minutes past it (H83). Past this one the client stops the server


class Client:
    def __deepcopy__(self, memo):
        return self                                    # a branch shares it: it holds a process, a pipe or fixed weights, and is never copied

    def __init__(self, log=None):
        self.lock = threading.Lock()                   # one pipe: the front end's page asks for drawings from another thread than the one solving
        self.log = open(log or os.path.join(OUT, "server.log"), "w")
        self.p = subprocess.Popen([PYTHON, "-u", SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log)
        self.dead = False                              # set once the server is stopped or lost: every later call raises at once, and the run keeps its last shape

    def __call__(self, name, **kw):
        if self.dead:
            raise RuntimeError(f"{name}: the FreeGSNKE server was stopped earlier in this run")
        with self.lock:
            box = {}

            def read():
                try:
                    box["reply"] = pickle.load(self.p.stdout)
                except BaseException as err:
                    box["error"] = err
            try:
                pickle.dump((name, kw), self.p.stdin, protocol=4)
                self.p.stdin.flush()
            except OSError as err:
                self.dead = True
                raise RuntimeError(f"{name}: the FreeGSNKE server is gone ({err!r})")
            th = threading.Thread(target=read, daemon=True)
            th.start()
            th.join(REPLY_LIMIT_S)
            if th.is_alive():
                self.dead = True
                try:
                    self.p.kill()
                except Exception:
                    pass
                raise RuntimeError(f"{name}: no reply from FreeGSNKE in {REPLY_LIMIT_S:.0f} s; the server was stopped and the run keeps its last shape")
            if "error" in box:
                self.dead = True
                raise RuntimeError(f"{name}: the server's reply was lost ({box['error']!r})")
            reply = box["reply"]
        if not reply["ok"]:
            raise RuntimeError(f"{name}: {reply['error']}\n{reply['trace']}")
        return reply["value"]

    def close(self):
        if not self.dead:
            try:
                pickle.dump(("quit", {}), self.p.stdin)
                self.p.stdin.close()
            except Exception:
                pass
        try:
            self.p.wait(timeout=20)
        except Exception:
            self.p.kill()


def plasma_from_torax(reactor, order=0, p_fast=None):
    """What FreeGSNKE needs from TORAX's present state: plasma current, F at the edge, and p' and FF' against normalized poloidal flux. FreeGSNKE rescales them together to the plasma current, so only their shapes and their ratio matter."""
    ppo, st = reactor.ppo, reactor.state
    pn = np.asarray(ppo.psi_norm, float)
    pp, ffp = np.asarray(ppo.pprime, float), np.asarray(ppo.FFprime, float)
    if p_fast is not None:               # TORAX's p' holds the thermal plasma. The fast ions' pressure goes in with TORAX's current profile kept, so FF' gives up what p' takes, as in its own calc_FFprime
        psi_cell, psi_face = np.asarray(st.core_profiles.psi.value, float), np.asarray(st.core_profiles.psi.face_value(), float)
        swing = float(psi_face[-1] - psi_face[0])
        x = (psi_cell - psi_face[0]) / swing
        extra = np.interp(pn, x, np.gradient(np.asarray(p_fast, float), x)) / swing
        pp = pp + extra
        ffp = ffp - extra * 4e-7 * np.pi / np.asarray(st.geometry.g3_face)
    keep = np.concatenate([[False], np.diff(pn) > 1e-6]) & (pn > 0.01)     # TORAX's two derivatives are singular at the axis: drop it, and carry the next value in
    pn, pp, ffp = pn[keep], pp[keep], ffp[keep]
    pp, ffp = np.convolve(np.pad(pp, 1, mode="edge"), np.ones(3) / 3, mode="valid"), np.convolve(np.pad(ffp, 1, mode="edge"), np.ones(3) / 3, mode="valid")      # second differences of TORAX's flux: smooth the cell-to-cell noise
    if order:                            # a low-order fit vanishing at the edge: the boundary and position hang on the broad shape of the current, and the solver converges poorly on the spikes a pinned pedestal puts in p'
        A = np.stack([pn ** k * (1 - pn) for k in range(order)], axis=1)
        grid = np.linspace(0.0, 1.0, 41)
        B = np.stack([grid ** k * (1 - grid) for k in range(order)], axis=1)
        Ip_now = float(np.abs(np.asarray(st.core_profiles.Ip_profile_face)[-1]))
        return dict(Ip=Ip_now, fvac=float(st.geometry.B_0) * float(st.geometry.R_major), psi_n=grid, pprime=B @ np.linalg.lstsq(A, pp, rcond=None)[0], ffprime=B @ np.linalg.lstsq(A, ffp, rcond=None)[0])
    geo = st.geometry
    fvac = float(geo.B_0) * float(geo.R_major)
    Ip = float(np.abs(np.asarray(st.core_profiles.Ip_profile_face)[-1]))
    grid = np.linspace(0.0, 1.0, 41)
    return dict(Ip=Ip, fvac=fvac, psi_n=grid, pprime=np.interp(grid, pn, pp), ffprime=np.interp(grid, pn, ffp))


def boundary_from_torax(reactor, n=48):
    """The last closed surface TORAX holds, as Miller's shape from its own radii, elongation and triangularity. Good enough to ask the coils for this plasma."""
    geo = reactor.state.geometry
    R_in, R_out = float(np.asarray(geo.R_in_face)[-1]), float(np.asarray(geo.R_out_face)[-1])
    kappa = float(np.asarray(geo.elongation_face)[-1])
    delta = float(np.asarray(geo.delta_face)[-1])
    R0, a = 0.5 * (R_in + R_out), 0.5 * (R_out - R_in)
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return R0 + a * np.cos(th + np.arcsin(np.clip(delta, -0.9, 0.9)) * np.sin(th)), kappa * a * np.sin(th), dict(R0=R0, a=a, kappa=kappa, delta=delta)


def geometry_from_result(result, reactor_or_F, n_rho, name="FreeGSNKE", Ip=None, B0=None, R0=None, last_surface=0.99, to_file=None):
    """A TORAX geometry from FreeGSNKE's flux map, through TORAX's EQDSK reader: psi grown outward in Wb, COCOS 11."""
    import contourpy
    from matplotlib.path import Path
    from scipy.interpolate import RectBivariateSpline
    from torax._src.geometry import eqdsk as eqdsk_geo
    R1, Z1, psi = result["R"][:, 0], result["Z"][0, :], result["psi"]
    spl = RectBivariateSpline(R1, Z1, psi, kx=3, ky=3)
    Rf, Zf = np.linspace(R1[0], R1[-1], 257), np.linspace(Z1[0], Z1[-1], 513)
    ax = float(spl(result["axis"][0], result["axis"][1])[0, 0])
    bd = float(result["psi_bndry"])
    sgn = np.sign(bd - ax)
    wb = sgn * 2 * np.pi * spl(Rf, Zf)
    # The boundary from this flux map itself. FreeGSNKE's comes from rays out of the axis, which miss an X-point's tip, and TORAX's reader clips its contours at the boundary's bounding box and takes the first one at each level. So: the outermost closed contour around the axis, and outside it the flux kept above its value, so no stray contour exists.
    level_a, level_b = sgn * 2 * np.pi * ax, sgn * 2 * np.pi * bd
    gen = contourpy.contour_generator(Rf, Zf, wb.T)
    boundary = None
    for f in (0.999, 0.995, 0.99, 0.985, 0.98, 0.975, 0.97):
        for line in gen.create_contour(level_a + f * (level_b - level_a)):
            if len(line) > 20 and np.hypot(*(line[0] - line[-1])) < 1e-6 and Path(line).contains_point((float(result["axis"][0]), float(result["axis"][1]))):
                boundary = line
                break
        if boundary is not None:
            break
    if boundary is None:
        raise ValueError("no closed flux surface around the axis at 0.97 or above")
    inside = Path(boundary).contains_points(np.stack(np.meshgrid(Rf, Zf, indexing="ij"), axis=-1).reshape(-1, 2)).reshape(len(Rf), len(Zf))
    wb = np.where(inside, wb, np.maximum(wb, level_a + f * (level_b - level_a)))
    pn = np.linspace(0.0, 1.0, len(Rf))
    q = np.interp(pn, np.r_[0.0, result["pn"], 1.0], np.r_[result["q"][0], result["q"], result["q"][-1] + 2 * (result["q"][-1] - result["q"][-2])])
    F = reactor_or_F(pn) if callable(reactor_or_F) else np.full(len(Rf), float(reactor_or_F))
    lr, lz = boundary[:, 0], boundary[:, 1]
    wall_r, wall_z = np.array([R1[0], R1[-1], R1[-1], R1[0], R1[0]]), np.array([Z1[0], Z1[0], Z1[-1], Z1[-1], Z1[0]])
    none = np.zeros(0)
    n = len(Rf)
    Rm, Zm = float(result["axis"][0]), float(result["axis"][1])
    obj = dict(bcentre=float(B0), cplasma=float(Ip if Ip is not None else result["Ip"]), dxc=none, dzc=none, ffprime=np.zeros(n), fpol=F, Ic=none, name=name, nbdry=len(lr), ncoil=0, nlim=5, nx=n, nz=len(Zf), pprime=-np.ones(n), pressure=np.zeros(n), psi=wb, psibdry=sgn * 2 * np.pi * bd, psimag=sgn * 2 * np.pi * ax, xbdry=lr, xc=none, xcentre=float(R0 if R0 is not None else Rm), xdim=float(R1[-1] - R1[0]), xgrid1=float(R1[0]), xlim=wall_r, xmag=Rm, zbdry=lz, zc=none, zdim=float(Z1[-1] - Z1[0]), zlim=wall_z, zmag=Zm, zmid=float(0.5 * (Z1[0] + Z1[-1])), qpsi=q)
    if to_file is not None:
        from eqdsk import EQDSKInterface
        EQDSKInterface(**obj).write(to_file, file_format="eqdsk")
        return to_file
    return eqdsk_geo.EQDSKConfig(geometry_type="eqdsk", cocos=11, eqdsk_object=obj, n_rho=n_rho, Ip_from_parameters=True, last_surface_factor=last_surface, n_surfaces=100).build_geometry()


MACHINES = {name: m["free"] for name, m in machines.MACHINES.items() if "grid" in m["free"]}


def start_geometry(machine, Ip, B0, R0, n_rho=50):
    """A first equilibrium for a machine with no shots: the shape its programme asks for, FreeGSNKE's own simple plasma, written where TORAX reads geometry. Returns TORAX's geometry settings. So a new machine needs no equilibrium file."""
    settings = MACHINES[machine]
    g = settings["shape"]
    folder = os.path.join(ROOT, "runs", "twin", "machines")
    os.makedirs(folder, exist_ok=True)
    c = Client(log=os.path.join(OUT, f"server_start_{os.getpid()}.log"))
    try:
        c("build", machine=machine, **settings["grid"])
        c("profiles_simple", Ip=Ip, fvac=B0 * R0, Raxis=R0, Beta0=0.6, alpha_m=2.0, alpha_n=1.4)
        res = c("inverse", R=np.array(g["R"], float), Z=np.array(g["Z"], float), null_R=g["null_R"], null_Z=g["null_Z"], tol=1e-3, max_iter=150, l2_reg=1e-12, symmetric=settings.get("symmetric", False))
    finally:
        c.close()
    name = f"{machine}_start_{os.getpid()}.geqdsk"     # one per process: several runs of a machine at once would remove each other's
    path = os.path.join(folder, name)
    if os.path.exists(path):
        os.remove(path)
    geometry_from_result(res, B0 * R0, n_rho, Ip=Ip, B0=B0, R0=R0, to_file=path)
    return dict(geometry_type="eqdsk", geometry_directory=folder, geometry_file=name, cocos=11, Ip_from_parameters=True, last_surface_factor=0.99, n_surfaces=100)


class Loop:
    SNAPSHOT_SHARED = ("res0", "coils0", "limits", "frozen")       # the start's solve and the machine's settings: fixed for the run
    """FreeGSNKE in a harness run, on a machine it describes (H46). Every interval TORAX's plasma goes over and the surfaces come back, through geo_overrides.
    mode "control": the coils move to keep the boundary TORAX started with, as a shape controller would; their currents are what the run says about the machine. "coils": the coils stay, and the plasma goes where the fields send it; with no vertical control a static solve drifts off, so that is for short looks.
    A branch restored from a snapshot goes on from the solver's latest state, not the branch's: the next solve starts near enough on the same boundary."""

    def __init__(self, h, machine="ITER", mode="control", interval=None, l2=1e-10, inverse_iter=300, control_iter=40, control_tol=3e-4, growth=False, shape="auto"):
        """shape: "machine", the shape the machine file's programme gives; "plasma", the boundary TORAX holds now; "auto", the machine's if the run started from it (machine_start), otherwise the plasma's."""
        self.h, self.r, self.mode, self.interval, self.l2, self.control_iter, self.control_tol = h, h.reactor, mode, interval, l2, control_iter, control_tol
        self.frozen = MACHINES[machine].get("frozen", ())
        r = self.r
        settings = MACHINES[machine]
        beam = h.plant.beam
        plasma = plasma_from_torax(r, order=4, p_fast=beam.pressure() if beam is not None else None)
        self.R, self.Z, target_shape = boundary_from_torax(r)
        self.Z = self.Z + settings["z0"]
        self.c = Client(log=os.path.join(OUT, f"server_loop_{os.getpid()}.log"))
        self.active = self.c("build", machine=machine, **settings["grid"])["active"]
        self.null = dict(null_R=[target_shape["R0"] - target_shape["delta"] * target_shape["a"] * 1.15], null_Z=[-target_shape["kappa"] * target_shape["a"] * 1.04 + settings["z0"]])
        if settings.get("shape") and (shape == "machine" or (shape == "auto" and h.s.get("machine_start"))):       # the programme says what shape it runs
            g = settings["shape"]
            self.R, self.Z, self.null = np.array(g["R"], float), np.array(g["Z"], float), dict(null_R=g["null_R"], null_Z=g["null_Z"])
            Rn, an = 0.5 * (self.R.max() + self.R.min()), 0.5 * (self.R.max() - self.R.min())
            target = dict(R0=Rn, a=an, kappa=float(np.max(self.Z) - np.min(self.Z)) / (2 * an), delta=0.0)
        else:
            target = target_shape
        self.limits = settings.get("limits")
        self.symmetric = settings.get("symmetric", False)      # a double null: a static solve has no vertical control and slips off the midplane by itself
        self.res0, reached = find_coils(self.c, plasma, self.R, self.Z, self.null, target, max_iter=inverse_iter, limits=self.limits, symmetric=self.symmetric)
        if reached < 1.0:
            self.c.close()
            raise RuntimeError(f"FreeGSNKE could not reach TORAX's profiles on {machine}: stopped at a blend of {reached}")
        self.s0 = shape_of(self.res0)
        self.geo_now = geo_of(self.res0, r)
        self.t_next = r.t
        r.geo_overrides = provider_between(r.t, self.geo_now, r.t + 1e3, self.geo_now)
        r.before_step.append(self)
        self.res, self.rejected, self.solves, self.wall = self.res0, 0, 0, 0.0
        self.log = []
        self.coils0 = self.c("currents_now")
        self.currents = self.coils0
        self.stability = None
        if growth:
            self.check_growth()

    def physical_current(self, t):
        """While current diffuses faster than it does (the 200 times of the settling phase), TORAX's flux-conservation term in the current equation stays at its physical size against a sped-up clock, as its own code has it (calc_coeffs: toc_psi is divided by the multiplier, the Phi_b_dot source is not). Surfaces moving then wreck the current: q0 went from 0.2 to 800 in 60 ms on SPARC. So that term waits for the physical clock."""
        t1 = self.h.s.get("physical_current_after", 25.0)
        return t1 is not None and t >= t1

    def check_growth(self):
        """The vertical growth rate about the present solution, from FreeGSNKE's linear model: about 25 s on a 65 by 129 grid, so on request."""
        cp = self.r.state.core_profiles
        self.stability = self.c("growth", plasma_resistivity=float(1.0 / np.mean(np.asarray(cp.sigma))))
        return self.stability

    def __call__(self, r=None):
        r = self.r
        if r.t < self.t_next - 1e-9:
            return
        interval = self.interval
        if interval is None:                           # half an energy time, and at least ten steps: fast enough for what moves the equilibrium, as docs/lit/30 has it
            interval = max(10 * float(r.get_param("numerics.fixed_dt")), 0.5 * float(r.ppo.W_thermal_total) / max(float(r.ppo.P_SOL_total) - float(r.ppo.P_radiation_e), 1e3))
        self.t_next = r.t + interval
        t0 = time.perf_counter()
        beam = self.h.plant.beam
        p = plasma_from_torax(r, order=4, p_fast=beam.pressure() if beam is not None else None)
        self.c("profiles", Ip=p["Ip"], fvac=p["fvac"], psi_n=p["psi_n"], pprime=p["pprime"], ffprime=p["ffprime"], Raxis=1.0)
        self.c("save")
        try:
            trial = self.c("forward", tol=1e-4, max_iter=40) if self.mode == "coils" else self.c("inverse", R=self.R, Z=self.Z, tol=self.control_tol, max_iter=self.control_iter, l2_reg=self.l2, frozen=self.frozen, limits=self.limits, symmetric=self.symmetric, **self.null)
        except RuntimeError as err:                    # the surfaces stay as they are until a later solve goes through
            print("solve failed", repr(err)[:120], flush=True)
            self.c("back")
            self.rejected += 1
            self.wall += time.perf_counter() - t0
            return
        self.solves += 1
        ok = trial["lcfs"] is not None and trial["q"] is not None and trial.get("axis") is not None
        trial_shape = shape_of(trial) if ok else None
        if ok and self.mode == "control":              # static solves have other branches, and once on one the next solve starts from it: on ITER one such solve took kappa from 1.84 to 1.72 and delta from 0.45 to 0.38 for the rest of the run, though the same plasmas replayed from the right branch hold 1.83 and 0.48. The controller was asked to hold this shape, so anything else is refused
            s_, s0_ = shape_of(trial), self.s0
            ok = abs(s_["a"] - s0_["a"]) < 0.05 * s0_["a"] and abs(s_["R0"] - s0_["R0"]) < 0.05 * s0_["a"] and abs(s_["kappa"] - s0_["kappa"]) < 0.08 and abs(s_["delta"] - s0_["delta"]) < 0.12 and abs(s_["Z_axis"] - s0_["Z_axis"]) < 0.1 * s0_["a"]      # delta loosely: read from where the flat top of the boundary peaks, it moves by 0.07 with nothing else moving
        elif ok:
            ok = abs(shape_of(trial)["a"] - self.s0["a"]) < 0.15 * self.s0["a"]
        if ok:
            try:
                geo_new = geo_of(trial, r)
                r.geo_overrides = provider_between(r.t, self.geo_now, r.t + interval, geo_new, phibdot=self.physical_current(r.t))
                self.geo_now, self.res = geo_new, trial
            except Exception as err:
                print("geometry failed", repr(err)[:160], flush=True)
                ok = False
        if not ok:
            self.c("back")
            self.rejected += 1
        self.wall += time.perf_counter() - t0
        self.currents = self.c("currents_now")
        self.log.append(dict(t=r.t, ok=bool(ok), wall=time.perf_counter() - t0, shape=trial_shape, plasma=p, W=float(r.ppo.W_thermal_total), li3=float(r.ppo.li3), currents={k: float(v) for k, v in self.currents.items() if k in self.active}))

    def summary(self):
        s_ = shape_of(self.res)
        coils = {k: float(self.currents[k]) for k in self.active}       # as of the last solve: this is read from the front end's thread while a solve may be running
        return dict(mode=self.mode, solves=self.solves, rejected=self.rejected, wall_s=self.wall, shape=s_, coils_A=coils, coils_moved_A={k: v - float(self.coils0[k]) for k, v in coils.items()}, lcfs=np.round(self.res["lcfs"][::4], 3).tolist(), stability=self.stability)

    def close(self):
        if self in self.r.before_step:
            self.r.before_step.remove(self)
        self.r.geo_overrides = provider_between(self.r.t, self.geo_now, self.r.t + 1e3, self.geo_now)
        self.c.close()


def iter_check(args):
    """On TORAX's ITER hybrid plasma: can FreeGSNKE's ITER machine hold what TORAX holds? The coils for TORAX's boundary and p', FF'; then shape and q from the free-boundary solve against TORAX's."""
    import check_plant
    import twin
    os.makedirs(OUT, exist_ok=True)
    t0 = time.perf_counter()
    h = twin.Harness(check_plant.scenario(False), keep_history=False)
    r = h.reactor
    h.apply(dict(cmd="step", until=args.settle))
    print(f"TORAX settled in {time.perf_counter() - t0:.0f} s, t = {r.t:.1f} s", flush=True)
    plasma = plasma_from_torax(r, order=4)
    R, Z, shape = boundary_from_torax(r)
    geo = r.state.geometry
    q_t = np.asarray(r.state.core_profiles.q_face)
    print("TORAX:", {k: round(v, 3) for k, v in shape.items()}, "Ip", round(plasma["Ip"] / 1e6, 2), "MA, B0", round(float(geo.B_0), 2), "T, q0", round(float(q_t[0]), 3), "q95", round(float(r.ppo.q95), 3), "li3", round(float(r.ppo.li3), 3), "beta_N", round(float(r.ppo.beta_N), 3), flush=True)
    print("p' shape", np.round(plasma["pprime"][[0, 10, 20, 30, 38, 40]] / np.abs(plasma["pprime"]).max(), 3).tolist(), "FF'", np.round(plasma["ffprime"][[0, 10, 20, 30, 38, 40]] / np.abs(plasma["ffprime"]).max(), 3).tolist(), flush=True)
    c = Client()
    t1 = time.perf_counter()
    info = c("build", machine="ITER", Rmin=3.2, Rmax=8.8, Zmin=-5.0, Zmax=5.0, nx=65, ny=65)
    print(f"FreeGSNKE up in {time.perf_counter() - t1:.0f} s", info, flush=True)
    print("p' at psi_n 0, .05, .25, .5, .75, .95, 1 (kPa/Wb)", np.round(plasma["pprime"][[0, 2, 10, 20, 30, 38, 40]] / 1e3, 1).tolist(), "FF'", np.round(plasma["ffprime"][[0, 2, 10, 20, 30, 38, 40]], 2).tolist(), flush=True)
    null = dict(null_R=[shape["R0"] - shape["delta"] * shape["a"] * 1.15], null_Z=[-shape["kappa"] * shape["a"] * 1.04])
    t1 = time.perf_counter()
    c("profiles_simple", Ip=plasma["Ip"], fvac=plasma["fvac"], Raxis=shape["R0"], Beta0=0.6, alpha_m=2.0, alpha_n=1.4)
    res = c("inverse", R=R, Z=Z, **null)
    print(f"inverse solve, simple plasma, {time.perf_counter() - t1:.1f} s; axis {np.round(res['axis'], 3).tolist()}, Ip {res['Ip'] / 1e6:.2f} MA", flush=True)
    t1 = time.perf_counter()
    c("profiles", Ip=plasma["Ip"], fvac=plasma["fvac"], psi_n=plasma["psi_n"], pprime=plasma["pprime"], ffprime=plasma["ffprime"], Raxis=1.0)
    res = c("inverse", R=R, Z=Z, **null)
    print(f"inverse solve, TORAX's plasma, {time.perf_counter() - t1:.1f} s; axis {np.round(res['axis'], 3).tolist()}, kappa {res['kappa']}, delta {res['delta']}, R_geo {res['R_geo']}, a {res['a']}, Ip {res['Ip'] / 1e6:.2f} MA", flush=True)
    q = res["q"]
    if q is not None:
        pn_t = np.asarray(r.ppo.psi_norm)
        print("q against normalized flux at 0.1, 0.3, 0.5, 0.7, 0.9: FreeGSNKE", np.round(np.interp([0.1, 0.3, 0.5, 0.7, 0.9], res["pn"], q), 3).tolist(), "TORAX", np.round(np.interp([0.1, 0.3, 0.5, 0.7, 0.9], pn_t, q_t), 3).tolist(), flush=True)
    np.savez(os.path.join(OUT, "iter_check.npz"), R=res["R"], Z=res["Z"], psi=res["psi"], lcfs=res["lcfs"], target_R=R, target_Z=Z)
    if res["lcfs"] is not None:
        lc = res["lcfs"]
        d = np.array([np.min(np.hypot(lc[:, 0] - a, lc[:, 1] - b)) for a, b in zip(R, Z)])
        print(f"distance from TORAX's boundary to the solved one: median {100 * np.median(d):.1f} cm, largest {100 * d.max():.1f} cm", flush=True)
    c.close()


def find_coils(c, plasma, R, Z, null, shape, steps=(0.25, 0.5, 0.75, 1.0), tol=1e-3, max_iter=150, limits=None, symmetric=False):
    """Coils for TORAX's boundary and TORAX's plasma. Inverse solves can report success on a wrong solution (a small plasma off to one side), so: first a simple plasma, whose solve is easy; then blend towards TORAX's p' and FF' in steps, starting each from the last and rejecting any whose minor radius has jumped. Returns the last good result, the blend reached."""
    c("profiles_simple", Ip=plasma["Ip"], fvac=plasma["fvac"], Raxis=shape["R0"], Beta0=0.6, alpha_m=2.0, alpha_n=1.4)
    res = c("inverse", R=R, Z=Z, tol=1e-4, limits=limits, symmetric=symmetric, **null)
    s = shape_of(res)
    print(f"simple plasma: {({k: round(v, 3) for k, v in s.items()})}", flush=True)
    base = c("arrays", x=plasma["psi_n"])
    # the same units as TORAX's up to a common factor: scale so the two blend by shape, with p' setting the scale
    a0 = np.abs(base["pprime"]).max()
    p0, f0 = base["pprime"] / a0, base["ffprime"] / a0
    a1 = np.abs(plasma["pprime"]).max()
    p1, f1 = np.sign(p0[len(p0) // 2]) * np.sign(plasma["pprime"][len(p0) // 2]) * plasma["pprime"] / a1, np.sign(p0[len(p0) // 2]) * np.sign(plasma["pprime"][len(p0) // 2]) * plasma["ffprime"] / a1
    reached = 0.0
    good = res
    step = steps[0]
    while reached < 1.0 and step >= 0.03:                      # a step that lands on a wrong solution is taken again, smaller, from the last good one
        lam = min(1.0, reached + step)
        c("save")
        c("profiles", Ip=plasma["Ip"], fvac=plasma["fvac"], psi_n=plasma["psi_n"], pprime=(1 - lam) * p0 + lam * p1, ffprime=(1 - lam) * f0 + lam * f1, Raxis=1.0 if lam == 1.0 else shape["R0"])
        try:
            r2 = c("inverse", R=R, Z=Z, tol=tol, max_iter=max_iter, limits=limits, symmetric=symmetric, **null)
            s2 = shape_of(r2) if r2["lcfs"] is not None else None
        except RuntimeError as err:
            r2, s2 = None, None
            print("blend solve failed", repr(err)[:120], flush=True)
        ok = s2 is not None and abs(s2["a"] - s["a"]) < 0.15 * s["a"] and abs(s2["R0"] - s["R0"]) < 0.3 * s["a"]
        print(f"blend {lam:.3f}: {'kept' if ok else 'rejected'} {({k: round(v, 3) for k, v in s2.items()}) if s2 else None}", flush=True)
        if not ok:
            c("back")
            step /= 2
            continue
        good, reached = r2, lam
    return good, reached


def shape_of(res):
    lc = res["lcfs"]
    R_out, R_in = float(lc[:, 0].max()), float(lc[:, 0].min())
    Z_top, Z_bot = float(lc[:, 1].max()), float(lc[:, 1].min())
    R_top, R_bot = float(lc[np.argmax(lc[:, 1]), 0]), float(lc[np.argmin(lc[:, 1]), 0])
    R0, a = 0.5 * (R_out + R_in), 0.5 * (R_out - R_in)
    return dict(R0=R0, a=a, kappa=(Z_top - Z_bot) / (2 * a), delta=(R0 - R_top) / a, delta_lower=(R0 - R_bot) / a, R_axis=float(res["axis"][0]), Z_axis=float(res["axis"][1]), R_out=R_out, R_in=R_in)


def synthetic_whmd(res, R_c=1.0):
    """The reading MAST's store calls whmd, from this equilibrium: EFIT++'s diamagnetic energy with no diamagnetic flux, which leaves 3/(8 mu0) times the integral of
    B_p^2 [(R - R_c) n_R + Z n_Z] dS over the last closed surface (docs/lit/36: equal to the store's whmd within 0.8% on 653 EFIT slices). R_c = 1 m is the reference
    radius of MAST's EFIT (level 2 bvac_r), a setting of the reading. A sensor of the simulated plasma (H68), not a stored energy: no pressure enters it."""
    from scipy.interpolate import RectBivariateSpline
    lc, R, Z, psi = res.get("lcfs"), np.asarray(res["R"]), np.asarray(res["Z"]), np.asarray(res["psi"])
    if lc is None or len(lc) < 10:
        return None
    spl = RectBivariateSpline(R[:, 0], Z[0, :], psi)          # FreeGS grids: R varies along the first axis; psi per radian
    r0, z0 = np.asarray(lc[:, 0], float), np.asarray(lc[:, 1], float)
    r1, z1 = np.roll(r0, -1), np.roll(z0, -1)
    dl = np.hypot(r1 - r0, z1 - z0)
    m = dl > 0
    r0, z0, r1, z1, dl = r0[m], z0[m], r1[m], z1[m], dl[m]
    rm, zm = 0.5 * (r0 + r1), 0.5 * (z0 + z1)
    s = np.sign(np.sum(r0 * z1 - r1 * z0))                    # +1 for a boundary run counter-clockwise in (R, Z)
    nR, nZ = s * (z1 - z0) / dl, -s * (r1 - r0) / dl          # outward normal
    Bp2 = (spl.ev(rm, zm, dx=1) ** 2 + spl.ev(rm, zm, dy=1) ** 2) / rm ** 2
    return float(3.0 / (8.0 * 4e-7 * np.pi) * np.sum(Bp2 * ((rm - R_c) * nR + zm * nZ) * 2.0 * np.pi * rm * dl))


def provider_between(t0, geo0, t1, geo1, phibdot=True):
    from torax._src.geometry import standard_geometry
    return standard_geometry.StandardGeometryProvider.create_provider({float(t0): geo0, float(t1): geo1}, calcphibdot=phibdot)


def geo_of(res, r, last_surface=0.99):
    pn = np.linspace(0.0, 1.0, 101)
    F = res["F"]
    Ff = (lambda x: np.interp(x, pn, F)) if F is not None else float(r.state.geometry.B_0) * float(r.state.geometry.R_major)
    geo = geometry_from_result(res, Ff, len(np.asarray(r.state.geometry.rho_norm)), Ip=float(np.abs(np.asarray(r.state.core_profiles.Ip_profile_face)[-1])), B0=float(r.state.geometry.B_0), R0=float(r.state.geometry.R_major), last_surface=last_surface)
    if r.state.geometry._z_magnetic_axis is None:      # a CHEASE start holds none, and TORAX's crash step needs the old and new geometry to have one structure
        geo = dataclasses.replace(geo, _z_magnetic_axis=None)
    return geo


def iter_couple(args):
    """TORAX's ITER hybrid with FreeGSNKE's free boundary in the loop. Two branches from one moment, both on FreeGSNKE's geometry: held, and updated every interval with the coils fixed. At a set time the heating rises, and the held branch cannot show what the plasma's shape does about it."""
    import check_plant
    import ntm
    import twin
    os.makedirs(OUT, exist_ok=True)
    t0 = time.perf_counter()
    h = twin.Harness(check_plant.scenario(False), keep_history=False)
    r = h.reactor
    h.apply(dict(cmd="step", until=args.settle))
    plasma = plasma_from_torax(r, order=4)
    R, Z, shape = boundary_from_torax(r)
    Z = Z + args.z0                      # TORAX's surfaces are centred on Z = 0; ITER's plasma sits about 0.6 m above the machine's midplane (FreeGSNKE's example has its O-point at Z = 0.66)
    print(f"TORAX settled in {time.perf_counter() - t0:.0f} s; {shape}; beta_N {float(r.ppo.beta_N):.2f}", flush=True)
    c = Client()
    c("build", machine="ITER", Rmin=3.2, Rmax=8.8, Zmin=-5.0, Zmax=5.0, nx=65, ny=65)
    null = dict(null_R=[shape["R0"] - shape["delta"] * shape["a"] * 1.15], null_Z=[-shape["kappa"] * shape["a"] * 1.04 + args.z0])
    t1 = time.perf_counter()
    res0, reached = find_coils(c, plasma, R, Z, null, shape, max_iter=args.inverse_iter, limits=MACHINES["ITER"]["limits"])
    s0 = shape_of(res0)
    if reached < 1.0:
        print(f"could not reach TORAX's profiles: stopped at a blend of {reached}", flush=True)
        c.close()
        return
    d = np.array([np.min(np.hypot(res0["lcfs"][:, 0] - a, res0["lcfs"][:, 1] - b)) for a, b in zip(R, Z)])
    print(f"coils found in {time.perf_counter() - t1:.0f} s; boundary from TORAX's: median {100 * np.median(d):.1f} cm, largest {100 * d.max():.1f} cm; {({k: round(v, 3) for k, v in s0.items()})}; F axis {res0['F'][0]:.3f} edge {res0['F'][-1]:.3f} T m", flush=True)
    t1 = time.perf_counter()
    geo0 = geo_of(res0, r)
    print(f"TORAX geometry from it in {time.perf_counter() - t1:.1f} s; TORAX's own: volume {float(np.asarray(r.state.geometry.volume_face)[-1]):.0f}, FreeGSNKE's: {float(np.asarray(geo0.volume_face)[-1]):.0f} m3; a {float(np.asarray(r.state.geometry.a_minor)):.3f}, {float(np.asarray(geo0.a_minor)):.3f}", flush=True)
    t_start = r.t
    r.geo_overrides = provider_between(t_start, geo0, t_start + 1e3, geo0)
    h.apply(dict(cmd="step", until=t_start + args.relax))      # let TORAX settle on the new surfaces
    h.apply(dict(cmd="snapshot", name="start"))
    t_start = r.t
    coils0 = c("currents_now")
    print(f"after {args.relax} s on FreeGSNKE's surfaces: fusion {float(r.ppo.P_fusion) / 1e6:.0f} MW, W {float(r.ppo.W_thermal_total) / 1e6:.0f} MJ, q95 {float(r.ppo.q95):.2f}, beta_N {float(r.ppo.beta_N):.2f}", flush=True)
    rows = {}
    print("coils at the start, kA:", {k: round(v / 1e3, 1) for k, v in coils0.items() if not k.startswith(("I", "O", "T", "D")) and abs(v) > 1.0 and len(k) <= 4}, flush=True)
    for branch in args.branches:
        h.apply(dict(cmd="restore", name="start"))
        if branch == "control":
            c("currents", values=coils0)
            c("guess", psi_total=res0["psi"])
        r.geo_overrides = provider_between(t_start, geo0, t_start + 1e3, geo0)
        rows[branch] = []
        geo_now, t_now = geo0, t_start
        heated = False
        res = res0
        rejected = 0
        t2 = time.perf_counter()
        while r.t < t_start + args.window - 1e-6:
            if not heated and r.t >= t_start + args.window / 3:
                h.apply(dict(cmd="nbi", P=args.extra_MW))
                heated = True
            if branch != "held":
                p = plasma_from_torax(r, order=4)
                c("profiles", Ip=p["Ip"], fvac=p["fvac"], psi_n=p["psi_n"], pprime=p["pprime"], ffprime=p["ffprime"], Raxis=1.0)
                c("save")
                trial = c("forward", tol=1e-4, max_iter=40) if branch == "free" else c("inverse", R=R, Z=Z, tol=args.control_tol, max_iter=args.control_iter, l2_reg=args.l2, frozen=args.frozen, limits=MACHINES["ITER"]["limits"], **null)        # free: the coils stay; control: the outer coils move to keep the boundary, as a shape controller would. The solenoid stays: the boundary alone leaves its currents free to run off, and in a flat top the solenoid's job is the flux
                ok = trial["lcfs"] is not None and trial["q"] is not None and abs(shape_of(trial)["a"] - s0["a"]) < 0.15 * s0["a"]
                if ok:
                    res = trial
                    geo_new = geo_of(res, r)
                    r.geo_overrides = provider_between(r.t, geo_now, r.t + args.interval, geo_new)
                    geo_now = geo_new
                else:
                    c("back")
                    rejected += 1
            target = r.t + args.interval
            while r.t < target - 1e-6 and r.step() is not None:
                pass
            s_ = shape_of(res) if branch != "held" else s0
            if branch == "control":
                moved = {k: v - coils0[k] for k, v in c("currents_now").items() if k in coils0 and abs(coils0[k]) > 1.0 and not k.startswith(("vessel", "passive"))}
                if int(round((r.t - t_start) / args.interval)) % 5 == 0:
                    print("  coils moved, kA:", {k: round(v / 1e3, 2) for k, v in moved.items() if abs(v) > 50.0}, flush=True)
            rows[branch].append(dict(t=r.t - t_start, W=float(r.ppo.W_thermal_total) / 1e6, P_fus=float(r.ppo.P_fusion) / 1e6, beta_N=float(r.ppo.beta_N), q95=float(r.ppo.q95), li=float(r.ppo.li3), volume=float(np.asarray(r.state.geometry.volume_face)[-1]), **s_))
            x = rows[branch][-1]
            print(f"{branch} t {x['t']:5.1f} W {x['W']:6.1f} fusion {x['P_fus']:6.1f} beta_N {x['beta_N']:.3f} q95 {x['q95']:.3f} li {x['li']:.3f} vol {x['volume']:.0f} R_axis {x['R_axis']:.3f} Z_axis {x['Z_axis']:.3f} R_out {x['R_out']:.3f} R_in {x['R_in']:.3f} kappa {x['kappa']:.3f} delta {x['delta']:.3f}", flush=True)
        print(f"{branch}: {time.perf_counter() - t2:.0f} s of wall time for {args.window} s of plasma; solves rejected {rejected}", flush=True)
    json.dump(rows, open(os.path.join(OUT, "iter_couple.json"), "w"), indent=0)
    c.close()


def iter_vde(args):
    """Loss of vertical control on TORAX's ITER hybrid plasma, from FreeGSNKE's evolving solver: the coils stay at their voltages, plasma and vessel free to move, after a small push from VS3. A vertical displacement comes out of the equations here, in place of the harness's 0D margin. Its growth rate is the test, against the linear model's on the same plasma."""
    import check_plant
    import twin
    os.makedirs(OUT, exist_ok=True)
    h = twin.Harness(check_plant.scenario(False), keep_history=False)
    r = h.reactor
    h.apply(dict(cmd="step", until=args.settle))
    plasma = plasma_from_torax(r, order=4)
    R, Z, shape = boundary_from_torax(r)
    Z = Z + args.z0
    cp = r.state.core_profiles
    eta = float(1.0 / np.mean(np.asarray(cp.sigma)))
    c = Client(log=os.path.join(OUT, "server_vde.log"))
    c("build", machine="ITER", Rmin=3.2, Rmax=8.8, Zmin=-5.0, Zmax=5.0, nx=65, ny=129)
    null = dict(null_R=[shape["R0"] - shape["delta"] * shape["a"] * 1.15], null_Z=[-shape["kappa"] * shape["a"] * 1.04 + args.z0])
    c("profiles_simple", Ip=plasma["Ip"], fvac=plasma["fvac"], Raxis=shape["R0"], Beta0=args.beta0, alpha_m=2.0, alpha_n=args.alpha_n)      # the evolving solver takes parametrized profiles
    res0 = c("inverse", R=R, Z=Z, tol=1e-4, limits=MACHINES["ITER"]["limits"], **null)
    print("simple plasma", {k: round(v, 3) for k, v in shape_of(res0).items()}, "plasma resistivity from TORAX", f"{eta:.2e}", "TORAX li3", round(float(r.ppo.li3), 3), flush=True)
    linear = c("growth", plasma_resistivity=eta)
    print("linear model on this plasma:", linear, flush=True)
    t1 = time.perf_counter()
    info = c("evolve_start", full_timestep=args.vde_dt, plasma_resistivity=eta)
    print(f"evolving solver built in {time.perf_counter() - t1:.0f} s; active coils {info['n_active']}", flush=True)
    rows = []
    volts = np.asarray(info["voltages"], float)
    push = volts.copy()
    push[3] = volts[3] + args.vde_push                 # VS3, the fourth circuit on ITER
    for k in range(args.vde_steps):
        t1 = time.perf_counter()
        out = c("evolve", n=1, voltages=push if k < args.vde_push_steps else volts)
        rows.append(dict(t=(k + 1) * args.vde_dt, Z_axis=float(out["axis"][1]), R_axis=float(out["axis"][0]), Ip=out["Ip"], wall=time.perf_counter() - t1))
        if k % 5 == 0:
            print(f"t {rows[-1]['t'] * 1e3:6.1f} ms Z {rows[-1]['Z_axis']:.4f} R {rows[-1]['R_axis']:.4f} Ip {rows[-1]['Ip'] / 1e6:.3f} MA, {rows[-1]['wall']:.1f} s per step", flush=True)
        json.dump(dict(linear=linear, rows=rows), open(os.path.join(OUT, "iter_vde.json"), "w"))
        if abs(rows[-1]["Z_axis"] - rows[0]["Z_axis"]) > 1.0:
            break
    t = np.array([x["t"] for x in rows])
    dz = np.abs(np.array([x["Z_axis"] for x in rows]) - rows[0]["Z_axis"])
    good = (t > args.vde_dt * (args.vde_push_steps + 5)) & (dz > 1e-3) & (dz < 0.5)
    if good.sum() > 3:
        g = np.polyfit(t[good], np.log(dz[good]), 1)[0]
        print(f"growth rate {g:.1f} per second, time {1e3 / g:.0f} ms, against {linear['growth_per_s']} from the linear model; {good.sum()} points, Z moved {dz[-1]:.3f} m in {t[-1] * 1e3:.0f} ms", flush=True)
    else:
        print("no clear growth", dz[-1], flush=True)
    c.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["iter_check", "iter_couple", "iter_vde"])
    ap.add_argument("--vde_dt", type=float, default=0.005)
    ap.add_argument("--vde_push", type=float, default=50.0)
    ap.add_argument("--vde_push_steps", type=int, default=4)
    ap.add_argument("--beta0", type=float, default=0.6)
    ap.add_argument("--alpha_n", type=float, default=1.4)
    ap.add_argument("--vde_steps", type=int, default=150)
    ap.add_argument("--settle", type=float, default=40.0)
    ap.add_argument("--relax", type=float, default=5.0)
    ap.add_argument("--window", type=float, default=15.0)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--extra_MW", type=float, default=33.0)
    ap.add_argument("--inverse_iter", type=int, default=300)
    ap.add_argument("--branches", nargs="*", default=["held", "free", "control"])
    ap.add_argument("--l2", type=float, default=1e-10)
    ap.add_argument("--frozen", nargs="*", default=["CS1", "CS2", "CS3", "VS3"])
    ap.add_argument("--control_tol", type=float, default=3e-4)
    ap.add_argument("--control_iter", type=int, default=40)
    ap.add_argument("--z0", type=float, default=0.6)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.command == "iter_check":
        iter_check(args)
    elif args.command == "iter_couple":
        iter_couple(args)
    elif args.command == "iter_vde":
        iter_vde(args)


if __name__ == "__main__":
    main()
