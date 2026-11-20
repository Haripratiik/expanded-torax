"""Fuelling by neutrals for the twin (H85): KN1D (kn1d_server.py, in its own venv) every `interval` of plasma time on the twin's own edge, driven by a wall molecular
pressure; its ionization inside the separatrix, uniform on each flux surface, is added to the particle source TORAX reads (ntm.EXTRA_PARTICLES) every step.
"kn1d_gauge" is an oracle: the pressure is MAST's measured fast ion gauge (FAIR-MAST level 1, gas_extract.py), so it measures what a perfect wall and vessel model would give,
not a hands-off prediction (H68). Geometry from MAST's EFIT limiter (outboard wall R 1.9 m, shadow beyond R 1.555 m, connection length 1.1 m there as KN1DPy's MAST-U example);
scrape-off layer e-folding lengths 2 cm (n_e) and 3 cm (T), assumed, the twin having no scrape-off layer of its own.
"""
import atexit
import os
import pickle
import subprocess
import threading

import numpy as np

ROOT = os.path.abspath(os.environ.get("XTORAX_HOME") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the workspace: runs/twin and external/ (README, "Where things live")
PYTHON = os.environ.get("XTORAX_KN1D_PYTHON") or os.path.join(ROOT, "external", "kn1d-venv", *(("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")))   # KN1DPy's own environment
SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kn1d_server.py")
GAS = os.path.join(ROOT, "runs", "twin", "gas")
REPLY_LIMIT_S = 180.0


class Server:
    def __deepcopy__(self, memo):
        return self                                    # a branch shares it: it holds a process, a pipe or fixed weights, and is never copied

    def __init__(self, log):
        self.lock = threading.Lock()
        self.log = open(log, "w")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        self.p = subprocess.Popen([PYTHON, "-I", "-u", SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, env=env)
        self.dead = False
        atexit.register(self.close)

    def __call__(self, name, **kw):
        if self.dead:
            raise RuntimeError("the KN1D server was stopped earlier in this run")
        with self.lock:
            box = {}

            def read():
                try:
                    box["reply"] = pickle.load(self.p.stdout)
                except BaseException as err:
                    box["error"] = err
            pickle.dump((name, kw), self.p.stdin, protocol=4)
            self.p.stdin.flush()
            th = threading.Thread(target=read, daemon=True)
            th.start()
            th.join(REPLY_LIMIT_S)
            if th.is_alive() or "error" in box:
                self.dead = True
                try:
                    self.p.kill()
                except Exception:
                    pass
                raise RuntimeError(f"{name}: KN1D server lost or too slow ({box.get('error')!r})")
        reply = box["reply"]
        if not reply["ok"]:
            raise RuntimeError(f"{name}: {reply['error']}\n{reply['trace']}")
        return reply["value"]

    def close(self):
        if self.dead:
            return
        try:
            pickle.dump(("quit", {}), self.p.stdin)
            self.p.stdin.flush()
            self.p.wait(timeout=10)
        except Exception:
            try:
                self.p.kill()
            except Exception:
                pass
        self.dead = True


class KN1DSource:
    def __init__(self, shot, interval=0.01, R_wall=1.9, R_lim=1.555, LC_shadow=1.1, lam_n=0.02, lam_T=0.03, rho_inner=0.5, folder=None, sol=None):
        f = os.path.join(GAS, f"{shot}.npz")
        if not os.path.exists(f):
            raise RuntimeError(f"no gas signals for {shot} (fairmast.py get aga, gas_extract.py)")
        g = dict(np.load(f))
        if "fig" not in g or not np.nanmax(np.nan_to_num(g["fig"])) > 0:
            raise RuntimeError(f"{shot}: no working fast ion gauge, so no oracle pressure")
        self.t_fig, self.fig = np.asarray(g["t_fig"], float), np.asarray(g["fig"], float)
        self.shot, self.interval = shot, interval
        self.geom = dict(R_wall=R_wall, R_lim=R_lim, LC_shadow=LC_shadow, lam_n=lam_n, lam_T=lam_T, rho_inner=rho_inner)
        self.folder = folder or os.path.join(ROOT, "runs", "twin", "kn1d", "runs", str(shot))
        os.makedirs(self.folder, exist_ok=True)
        self.server = Server(os.path.join(self.folder, "server.log"))
        self.S, self.rho, self.t_last, self.last = None, None, -1e9, {}
        self.sol = sol                     # a sol.SOL: its separatrix density, temperature and decay lengths make KN1D's scrape-off layer, and KN1D's ionization there feeds it back (H86)

    def gauge(self, t):
        w = (self.t_fig > t - 0.005) & (self.t_fig < t + 0.005) & np.isfinite(self.fig)
        v = float(np.median(self.fig[w])) if w.any() else float("nan")
        return v if np.isfinite(v) else 0.0

    def __call__(self, r):
        """Refresh the source when due; add it to this step's particle table. Returns the core source in particles per second."""
        import ntm
        t = r.t
        if self.S is None or t - self.t_last >= self.interval - 1e-9:
            geo, cp = r.state.geometry, r.state.core_profiles
            rho = np.asarray(geo.rho_norm)
            p = self.gauge(t)
            if not p > 1e14:                                   # no reading (the gauge is NaN or zero for stretches of some shots): no source until it reads again
                self.S, self.rho, self.t_last = np.zeros(len(rho)), rho, t
                self.last = dict(kn1d_source=0.0, kn1d_n0_sep=0.0, kn1d_inside=0.0, kn1d_gauge=p, kn1d_ly=0.0, kn1d_ion_all=0.0)
                ntm.EXTRA_PARTICLES[:] += 0.0
                r.carry.update(self.last)
                return 0.0
            geom = dict(self.geom)
            if self.sol is not None and self.sol.last:
                geom.update(lam_n=float(np.clip(self.sol.last["sol_lam_n"], 0.003, 0.1)), lam_T=float(np.clip(self.sol.last["sol_lam_T"], 0.003, 0.1)),
                            n_sep=float(self.sol.last["sol_n_sep"]), T_sep=float(self.sol.last["sol_T_sep"]))
            try:
                res = self.server("source", rho=rho, n_e=np.asarray(cp.n_e.value), T_e=np.asarray(cp.T_e.value) * 1e3, T_i=np.asarray(cp.T_i.value) * 1e3,
                                  R_out=np.asarray(geo.R_out), gauge=p, dir=os.path.join(self.folder, "scratch"), **geom)
                if self.sol is not None:
                    self.sol.S_sol = max(float(res.get("S_sol", 0.0)), 0.0)
                self.S, self.rho, self.t_last = np.clip(np.nan_to_num(np.asarray(res["S"], float)), 0.0, None), rho, t
                dv = np.asarray(geo.vpr) * float(np.asarray(geo.drho_norm).ravel()[0])
                self.last = dict(kn1d_source=float(np.sum(self.S * dv)), kn1d_n0_sep=float(res["n0_sep"]), kn1d_inside=float(res["f_inside"]), kn1d_gauge=p,
                                 kn1d_ly=float(res.get("ly", 0.0)), kn1d_ion_all=float(res.get("S_sol", 0.0)) + float(res.get("S_in", 0.0)))
            except RuntimeError as err:
                print(f"KN1D at {t:.3f}: {err}; the last source is kept", flush=True)
                self.t_last = t
        if self.S is not None:
            ntm.EXTRA_PARTICLES[:] += np.interp(ntm.TURB_RHO, self.rho, self.S)
        r.carry.update(self.last)
        return self.last.get("kn1d_source", 0.0)

    def close(self):
        self.server.close()
