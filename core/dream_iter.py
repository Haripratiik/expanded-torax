"""ITER after the onset (H76, docs/digital-twin.md "Judging a disruption chip"): DREAM in fluid mode on an ITER-like 15 MA DT plasma, set up as Vallhagen
et al. 2020 set up theirs (J. Plasma Phys., arXiv 2004.12861, GO code, DREAM's predecessor), so that it is checked against their numbers before it judges
anything; then the chip's question: what neon does when it arrives some time after a thermal quench that nothing mitigated.

Plasma (their section 3): Ip 15 MA, a = 2 m, R = 6 m, circular (toroidicity neglected, as theirs), a perfectly conducting wall at b = 2.15 m, n_e = 1e20
flat (deuterium and tritium half each), T = 20 keV (1 - (r/a)^2), j = j0 (1 - (r/a)^2)^0.41.
Thermal quench (their eq. 2.8): T(r, t) = 50 eV + (T0(r) - 50 eV) exp(-t / 1 ms) for 6 ms, then the energy balance (DREAM's self-consistent T) with a heat
diffusivity of 1 m^2/s. Runaways: Dreicer (neural network), tritium decay, Compton at ITER's photon flux (1e18 m^-2 s^-1), avalanche with partial
screening; no hot tail, as theirs.
Checks (their fig. 1 and Table 1), material spread evenly at t = 0: Case 1, 1e20 m^-3 of neon: Ohmic current gone at about 20 ms, a runaway beam of
6.7 MA; Case 2, 3e18 neon and 3e20 deuterium: no runaways, current quench far too slow; Case 4, 8e18 neon and 7e20 deuterium: 3.7 MA with an acceptable
current quench, the lowest they found.
The chip's question: a thermal quench nothing mitigated, after which the temperature is held at 6, 10 or 14 eV (a stated choice that sets how fast the
unmitigated current quench goes, as nothing measured fixes it for ITER; beryllium from the wall instead reheated the plasma or stopped the solver), and the
Case 1 or Case 4 material arriving t_a after the onset. ITER's pellets need more than 30 ms from trigger to plasma (Raman et al.).
Numerics: a stage DREAM fails on is redone with steps 4 times shorter over its first 20 ms (RETRY); with deuterium this happens in most stretches.

Usage (WSL, Aurora's environment, DREAMPATH=$HOME/build/DREAM):
  $XTORAX_DREAM_PYTHON core/dream_iter.py check [--ne 1e20] [--nd 0] [--tend 0.3]
  $XTORAX_DREAM_PYTHON core/dream_iter.py natural --tpost 10 [--tend 0.3]
  $XTORAX_DREAM_PYTHON core/dream_iter.py late --tpost 10 --arrive 0.030 [--ne 1e20] [--nd 0] [--tend 0.3] [--dtcq 5e-5]
Writes runs/twin/disruptions/iter/<name>/summary.json (DREAM's own files under ~/runs/dream_iter). Scans: runs/twin/_dream_iter_scan.sh, _scan2.sh.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
DREAMPATH = os.environ.get("DREAMPATH", os.path.expanduser("~/build/DREAM"))
sys.path.insert(0, os.path.join(DREAMPATH, "py"))
HERE = os.path.abspath(os.environ.get("XTORAX_HOME") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the workspace
OUT = os.path.join(HERE, "runs", "twin", "disruptions", "iter")
WORK = os.path.expanduser("~/runs/dream_iter")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import DREAM  # noqa: E402
import DREAM.Settings.CollisionHandler as Collisions  # noqa: E402
import DREAM.Settings.Equations.ColdElectronTemperature as Temperature  # noqa: E402
import DREAM.Settings.Equations.ElectricField as ElectricField  # noqa: E402
import DREAM.Settings.Equations.IonSpecies as Ions  # noqa: E402
import DREAM.Settings.Equations.RunawayElectrons as RE  # noqa: E402
import DREAM.Settings.RadialGrid as RadialGrid  # noqa: E402
import DREAM.Settings.Solver as Solver  # noqa: E402
import DREAM.Settings.TransportSettings as Transport  # noqa: E402
from DREAM import DREAMSettings  # noqa: E402

C = dict(R0=6.0, a=2.0, b=2.15, B0=5.3, Ip=15e6, T0=20e3, ne=1e20, nr=20, t0=1e-3, Tf=50.0, t_tq=6e-3, chi=1.0)
EXTRA = (("Ne", 10), ("Be", 4), ("Dinj", 1))   # species that may come in: neon and deuterium the chip injects, beryllium the wall gives off; present from the start at 1 m^-3
DEPOSIT = 1e-4                              # s over which injected or released material arrives (spread evenly over the radius)


def base(c):
    ds = DREAMSettings()
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED
    ds.collisions.lnlambda = Collisions.LNLAMBDA_ENERGY_DEPENDENT
    ds.radialgrid.setType(RadialGrid.TYPE_CYLINDRICAL)
    ds.radialgrid.setB0(c["B0"])
    ds.radialgrid.setMinorRadius(c["a"])
    ds.radialgrid.setWallRadius(c["b"])
    ds.radialgrid.setNr(c["nr"])
    ds.runawaygrid.setEnabled(False)
    ds.hottailgrid.setEnabled(False)
    ds.eqsys.n_re.setAvalanche(RE.AVALANCHE_MODE_FLUID_HESSLOW)
    ds.eqsys.n_re.setDreicer(RE.DREICER_RATE_NEURAL_NETWORK)
    ds.eqsys.n_re.setTritium(True)
    ds.eqsys.n_re.setCompton(RE.COMPTON_RATE_ITER_DMS)
    ds.eqsys.n_re.setEceff(RE.COLLQTY_ECEFF_MODE_FULL)
    ds.other.include("fluid", "scalar")
    ds.solver.setType(Solver.NONLINEAR)
    ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_LU)
    ds.solver.setMaxIterations(300)
    ds.solver.tolerance.set(reltol=1e-6)
    ds.solver.tolerance.set(unknown="n_re", reltol=1e-6, abstol=1e5)
    ds.solver.tolerance.set(unknown="j_re", reltol=1e-6, abstol=1e-5)
    return ds


def T0_of(c, r):
    return c["T0"] * (1 - (r / c["a"]) ** 2) + 1.0


def start(c, folder):
    """The plasma before the quench, carrying 15 MA in its profile: D and T fully ionized, the extra species at 1 m^-3."""
    ds = base(c)
    r = np.linspace(0, c["a"], 81)
    T0 = T0_of(c, r)
    j = (1 - (1 - 0.001 ** (1 / 0.41)) * (r / c["a"]) ** 2) ** 0.41
    ds.eqsys.T_cold.setPrescribedData(T0, radius=r)
    ds.eqsys.E_field.setPrescribedData(1e-4)
    # one temperature, as theirs: no separate ion heat (given an ion temperature, DREAM holds the ions' energy fixed through a prescribed thermal quench
    # and they reheat the electrons to about 560 eV once the energy balance takes over)
    ds.eqsys.n_i.addIon(name="D", Z=1, Z0=1, iontype=Ions.IONS_DYNAMIC, n=0.5 * c["ne"] * np.ones_like(r), r=r)
    ds.eqsys.n_i.addIon(name="T", Z=1, Z0=1, iontype=Ions.IONS_DYNAMIC, n=0.5 * c["ne"] * np.ones_like(r), r=r, tritium=True)
    for name, zz in EXTRA:
        ds.eqsys.n_i.addIon(name=name, Z=zz, Z0=0, iontype=Ions.IONS_DYNAMIC, n=np.ones_like(r), r=r)
    ds.eqsys.n_re.setTritium(False)                  # no runaways while the current settles
    ds.eqsys.n_re.setCompton(RE.COMPTON_MODE_NEGLECT)
    ds.eqsys.n_re.setDreicer(RE.DREICER_RATE_DISABLED)
    ds.eqsys.n_re.setAvalanche(RE.AVALANCHE_MODE_NEGLECT)
    ds.timestep.setTmax(1e-6)
    ds.timestep.setNt(1)
    do = DREAM.runiface(ds, os.path.join(folder, "cond.h5"), quiet=True)
    sigma = do.other.fluid.conductivity[-1, :]
    rg = do.grid.r[:]
    jj = np.interp(rg, r, j)
    j0 = c["Ip"] * 2 * np.pi / do.grid.integrate(jj)
    E = j0 * jj / sigma
    do.close()
    prev = "cond"
    for it in range(5):
        ds.eqsys.E_field.setPrescribedData(E * np.ones((1, rg.size)), radius=rg, times=[0])
        ds.timestep.setTmax(1e-4)
        ds.timestep.setNt(2)
        ds.fromOutput(os.path.join(folder, prev + ".h5"))
        do = DREAM.runiface(ds, os.path.join(folder, f"init{it}.h5"), quiet=True)
        ip = float(np.ravel(do.eqsys.I_p[-1])[0])
        do.close()
        prev = f"init{it}"
        if abs(ip / c["Ip"] - 1) < 0.003:
            break
        E = E * c["Ip"] / ip
    os.replace(os.path.join(folder, prev + ".h5"), os.path.join(folder, "init.h5"))
    return ds


def phase(ds_prev, c, folder, name, prev, tmax, nt, prescribed_T=None, sources=None):
    """One stage from the last one's end. prescribed_T: (times, radii, T) for the thermal quench; otherwise the energy balance. sources: {species: density
    added, m^-3}, arriving as neutral atoms evenly over the radius at the start of the stage, over DEPOSIT."""
    ds = DREAMSettings(ds_prev)
    ds.eqsys.n_re.setTritium(True)
    ds.eqsys.n_re.setCompton(RE.COMPTON_RATE_ITER_DMS)
    ds.eqsys.n_re.setDreicer(RE.DREICER_RATE_NEURAL_NETWORK)
    ds.eqsys.n_re.setAvalanche(RE.AVALANCHE_MODE_FLUID_HESSLOW)
    ds.eqsys.E_field.setType(ElectricField.TYPE_SELFCONSISTENT)
    ds.eqsys.E_field.setBoundaryCondition(ElectricField.BC_TYPE_PRESCRIBED, inverse_wall_time=0, V_loop_wall_R0=0)      # perfectly conducting wall at b
    if prescribed_T is not None:
        tt, rr, TT = prescribed_T
        ds.eqsys.T_cold.setPrescribedData(TT, radius=rr, times=tt)
    else:
        ds.eqsys.T_cold.setType(Temperature.TYPE_SELFCONSISTENT)
        ds.eqsys.T_cold.setRecombinationRadiation(Temperature.RECOMBINATION_RADIATION_NEGLECTED)
        ds.eqsys.T_cold.transport.prescribeDiffusion(drr=c["chi"])
        ds.eqsys.T_cold.transport.setBoundaryCondition(Transport.BC_F_0)
    tg = np.array([0.0, DEPOSIT, DEPOSIT * 1.001, max(tmax, DEPOSIT * 1.01)])
    on = np.array([1.0, 1.0, 0.0, 0.0])
    ds.eqsys.n_i.tSourceTerm = None
    ds.eqsys.n_i.rSourceTerm = None
    rs = np.linspace(0, c["a"], 20)
    for sp, zz in EXTRA:
        S = np.zeros((zz + 1, tg.size, rs.size))
        rate = (sources or {}).get(sp, 0.0) / DEPOSIT
        S[0] = (rate * on)[:, None] * np.ones((1, rs.size))
        ds.eqsys.n_i.addIonSource(sp, dNdt=S, t=tg, r=rs, source_type=Ions.ION_SOURCE_PRESCRIBED_VOLUMETRIC)
    ds.timestep.setTmax(tmax)
    ds.timestep.setNt(nt)
    ds.timestep.setNumberOfSaveSteps(min(nt, 200))
    ds.fromOutput(os.path.join(folder, prev + ".h5"))
    do = DREAM.runiface(ds, os.path.join(folder, name + ".h5"), quiet=True)
    return ds, do


def keep(out, do, t_off):
    t = do.grid.t[:] + t_off
    ip = np.array([float(np.ravel(v)[0]) for v in do.eqsys.I_p[:]])
    ire = np.asarray(do.eqsys.j_re.current()[:], float) if hasattr(do.eqsys.j_re, "current") else np.zeros_like(ip)
    iohm = np.asarray(do.eqsys.j_ohm.current()[:], float) if hasattr(do.eqsys.j_ohm, "current") else ip - ire
    for k, v in (("t", t), ("Ip", ip), ("Ire", ire), ("Iohm", iohm), ("Te0", do.eqsys.T_cold[:, 0]), ("Te_mid", do.eqsys.T_cold[:, do.grid.r.size // 2])):
        out[k] += [float(x) for x in np.ravel(v)]


def tq_profile(c, t_a, t_b, n=41, Tf=None):
    """The prescribed thermal quench between t_a and t_b after the onset, with times counted from t_a."""
    Tf = c["Tf"] if Tf is None else Tf
    r = np.linspace(0, c["a"], 81)
    t = np.linspace(t_a, t_b, max(n, int((t_b - t_a) / 2.5e-4) + 1))      # at least one point per 0.25 ms: the decay to Tf is a millisecond
    T = Tf + (T0_of(c, r)[None, :] - Tf) * np.exp(-t[:, None] / c["t0"])
    return t - t_a, r, T


FINE = (3 * DEPOSIT, 600)                   # after material arrives: 0.3 ms in 0.5 us steps, as DREAM's own ITER example does for its first 20 us
DT_TQ, DT_CQ = 5e-6, 5e-5                   # steps of the prescribed thermal quench and of the current quench
RETRY = (0.02, 4, 2)                        # a stage DREAM fails on is redone with steps 4 times shorter over its first 20 ms, at most twice: with injected
                                            # deuterium the solver can fail where the temperature collapses, and whether it does changes with the step's last bits


def run(c, tag, inject_at=None, ne_inj=1e20, be=0.0, t_end=0.15, tpost=None, nd_inj=0.0, dt_cq=DT_CQ):
    """inject_at: seconds after the onset at which the neon arrives (0: with the thermal quench, the check); be: beryllium the wall gives off at the
    thermal quench, m^-3; tpost: the natural disruption's temperature after its thermal quench, eV, held until the neon arrives (the stated choice that
    sets how fast an unmitigated current quench goes); without it, Vallhagen's switch to the energy balance at 6 ms."""
    folder = os.path.join(WORK, tag)
    os.makedirs(folder, exist_ok=True)
    ds = start(c, folder)
    out = dict(t=[], Ip=[], Ire=[], Iohm=[], Te0=[], Te_mid=[])
    if tpost is None:
        t_tq, Tf = c["t_tq"], c["Tf"]
    else:
        t_tq, Tf = (inject_at if inject_at is not None and inject_at > 0 else t_end), tpost
    arrivals = {}                                   # time after the onset -> material arriving then
    if be > 0:
        arrivals.setdefault(0.0, {})["Be"] = be
    if inject_at is not None:
        arrivals.setdefault(max(inject_at, 0.0), {})["Ne"] = ne_inj
        if nd_inj > 0:
            arrivals[max(inject_at, 0.0)]["Dinj"] = nd_inj
    starts = list(arrivals) + [t_tq]                # fine steps after material arrives and where the energy balance takes over
    cuts = sorted(set([0.0, c["t_tq"], t_tq, t_end] + starts + [min(a + FINE[0], t_end) for a in starts]))
    cuts = [x for x in cuts if x <= t_end]
    stages = []
    for ta, tb in zip(cuts[:-1], cuts[1:]):
        if tb - ta < 1e-9:
            continue
        fine = any(abs(ta - a) < 1e-12 or (a < ta < a + FINE[0] - 1e-12) for a in starts)
        stages.append((ta, tb, FINE[0] / FINE[1] if fine else (DT_TQ if tb <= c["t_tq"] + 1e-12 else dt_cq), 0))
    prev, k, retried = "init", 0, []
    while stages:
        ta, tb, dt, level = stages.pop(0)
        src = arrivals.get(ta, {})
        nt = max(int(round((tb - ta) / dt)), 5)
        pT = tq_profile(c, ta, tb, Tf=Tf) if tb <= t_tq + 1e-12 else None
        name = f"s{k:02d}"
        try:
            ds_new, do = phase(ds, c, folder, name, prev, tb - ta, nt, prescribed_T=pT, sources=src)
        except DREAM.DREAMException:          # the class (DREAM's __init__ imports it over the module's name)
            if level >= RETRY[2]:
                raise
            mid = min(tb, ta + RETRY[0])
            stages[:0] = [(ta, mid, dt / RETRY[1], level + 1)] + ([(mid, tb, dt, level)] if tb - mid > 1e-9 else [])
            retried.append([ta, mid, dt / RETRY[1]])
            continue
        ds = ds_new
        keep(out, do, ta - float(do.grid.t[0]))
        do.close()
        prev, k = name, k + 1
    out.update(summary(out, c))
    out.update(case=dict(c), inject_at=inject_at, ne_inj=ne_inj, be=be, nd_inj=nd_inj, retried=retried)
    keep_to = os.path.join(OUT, tag)
    os.makedirs(keep_to, exist_ok=True)
    json.dump(out, open(os.path.join(keep_to, "summary.json"), "w"), indent=0, default=float)
    return out


def summary(o, c):
    t, ip, ire, iohm = (np.array(o[k]) for k in ("t", "Ip", "Ire", "Iohm"))
    i0 = c["Ip"]
    res = dict(Ire_max_MA=float(np.max(ire)) / 1e6, Ire_end_MA=float(ire[-1]) / 1e6, Ip_end_MA=float(ip[-1]) / 1e6)
    k80, k20 = np.flatnonzero(iohm < 0.8 * i0), np.flatnonzero(iohm < 0.2 * i0)
    if k80.size and k20.size:                         # their definition: t_CQ = [t(I_Ohm = 0.2 I0) - t(I_Ohm = 0.8 I0)] / 0.6
        res["t_cq_ms"] = 1e3 * (t[k20[0]] - t[k80[0]]) / 0.6
        res["t_ohm20_ms"] = 1e3 * t[k20[0]]
    res["Te0_10ms_eV"] = float(np.interp(0.010, t, o["Te0"]))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("check", "late", "natural"))
    ap.add_argument("--ne", type=float, default=1e20, help="neon injected, m^-3")
    ap.add_argument("--nd", type=float, default=0.0, help="deuterium injected with the neon, m^-3")
    ap.add_argument("--be", type=float, default=0.0, help="beryllium the wall gives off at the thermal quench, m^-3")
    ap.add_argument("--arrive", type=float, default=0.03, help="s after the onset at which the neon arrives (late)")
    ap.add_argument("--tend", type=float, default=0.15)
    ap.add_argument("--nr", type=int, default=20)
    ap.add_argument("--tpost", type=float, default=None, help="the natural disruption's temperature after its thermal quench, eV, held until the neon arrives")
    ap.add_argument("--dtcq", type=float, default=DT_CQ, help="step of the current quench, s (a convergence check)")
    ap.add_argument("--set", nargs="*", default=[], help="override the plasma: key=value for any of C (R0, a, b, B0, Ip, T0, ne, ...), as the harness does from its run")
    ap.add_argument("--name", default=None, help="the run's folder name, in place of the one made from the settings")
    a = ap.parse_args()
    c = dict(C, nr=a.nr)
    for kv in a.set:
        k, v = kv.split("=", 1)
        if k not in C:
            raise SystemExit(f"--set: {k} is not one of {sorted(C)}")
        c[k] = float(v)
    tp = "" if a.tpost is None else f"_tpost{a.tpost:g}"
    dts = "" if a.dtcq == DT_CQ else f"_dt{a.dtcq:.1e}"
    if a.what == "check":
        o = run(c, a.name or f"check_ne{a.ne:.1e}_nr{a.nr}" + (f"_nd{a.nd:.1e}" if a.nd else "") + dts, inject_at=0.0, ne_inj=a.ne, t_end=a.tend, nd_inj=a.nd, dt_cq=a.dtcq)
    elif a.what == "natural":
        o = run(c, a.name or f"natural_be{a.be:.1e}{tp}{dts}", inject_at=None, be=a.be, t_end=a.tend, tpost=a.tpost, dt_cq=a.dtcq)
    else:
        o = run(c, a.name or f"late_be{a.be:.1e}_at{a.arrive * 1e3:.0f}ms_ne{a.ne:.1e}{tp}" + (f"_nd{a.nd:.1e}" if a.nd else "") + dts, inject_at=a.arrive, ne_inj=a.ne,
                be=a.be, t_end=a.tend, tpost=a.tpost, nd_inj=a.nd, dt_cq=a.dtcq)
    print(json.dumps({k: v for k, v in o.items() if k not in ("t", "Ip", "Ire", "Iohm", "Te0", "Te_mid", "case")}, default=float))
