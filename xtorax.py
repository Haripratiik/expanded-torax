"""xtorax: one entry point for the expanded TORAX twin and its harness. README.md and AGENTS.md beside this file say what each part is and how far to trust it.

Run with a Python that has TORAX 1.4.3 (README.md, "Installation guide"), from this folder:
  python xtorax.py <command> [options]

Commands
  env                                 check the environments, outside tools and data the twin needs, and what each enables
  versions                            the tested versions and the recommended one (natural.RECOMMENDED)
  shots [--all]                       MAST shots with every file a whole-shot run needs (--all: also those that cannot be scored)
  run SHOT ... | --scorable [--version V] [--tag TAG] [--seeds N] [--lanes N] [--extra JSON]
                                      whole MAST shots, hands-off from their first frame (listed shots only, when runs/twin/dev_shots.json exists); --scorable runs every scorable shot;
                                      --lanes splits them over N processes (tags TAGa, TAGb, ...); --extra adds override keys to the version's
  score TAG [--name NAME]             score runs against what MAST measured (whole_shot.py); TAG also takes in its lanes (TAGa, TAGb, ...) or a comma list
  ready NAME [--version V]            check a scored set against the written readiness criteria (readiness.py)
  compare NAME_A NAME_B               two scored sets on the same starts and times (whole_shot.py --common); version 12's reference set is v19
  script SCENARIO COMMANDS.json       a scenario driven by a list of harness commands, each answer printed
  serve SCENARIO [--port 8801]        the harness as a local server: POST commands as JSON, GET /observe, GET /help
  call JSON [--port 8801]             one command to a running server
  studio [--scenario NAME] [--port 8801]   the front end, in a browser at http://127.0.0.1:PORT
  smoke                               the harness's functional checks one after another; those needing FreeGSNKE or MAST shot files
                                      are skipped when they are missing (about 4 minutes on a clean workspace, 10 to 12 with everything)

SCENARIO is a file in scenarios/ (name with or without .json) or a path.
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True                           # no __pycache__ in core/ from this file's own imports
PACKAGE = "1.2"                                          # the package's release; the plasma's settings are natural.RECOMMENDED's version (CHANGELOG.md)
HERE = os.path.dirname(os.path.abspath(__file__))          # the repository: core/, scenarios/, studio/, data/
CORE = os.path.join(HERE, "core")                          # the simulator and harness
ROOT = os.path.abspath(os.environ.get("XTORAX_HOME") or HERE)   # the workspace: data and results in runs/twin, outside tools in external/
WIN = os.name == "nt"


def venv_python(name):
    return os.path.join(ROOT, "external", name, "Scripts" if WIN else "bin", "python.exe" if WIN else "python")


TOOLS = {"torax-venv": "XTORAX_PYTHON", "freegsnke-venv": "XTORAX_FREEGSNKE_PYTHON", "kn1d-venv": "XTORAX_KN1D_PYTHON", "fairmast-venv": "XTORAX_FAIRMAST_PYTHON"}


def tool_python(venv):
    """The Python for one environment: its variable if set, else external/<venv> if it exists; TORAX's falls back to this Python."""
    if os.environ.get(TOOLS[venv]):
        return os.environ[TOOLS[venv]]
    if os.path.exists(venv_python(venv)) or venv != "torax-venv":
        return venv_python(venv)
    return sys.executable


PY = tool_python("torax-venv")
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
NAT = os.path.join(ROOT, "runs", "twin", "mast_natural2")


def core(*parts):
    return os.path.join(CORE, *parts)


def run_py(args, timeout=None, isolated=False):
    """A script of core/ in TORAX's environment, from the repository, its output passed through."""
    cmd = [PY, "-u"] + (["-I"] if isolated else []) + args
    return subprocess.call(cmd, cwd=ROOT, env=ENV, timeout=timeout)


def natural():
    sys.path.insert(0, CORE)
    import natural as nat
    return nat


def scenario_path(name):
    if os.path.exists(name):
        return os.path.abspath(name)
    base = name if name.endswith(".json") else name + ".json"
    path = os.path.join(HERE, "scenarios", base)
    if not os.path.exists(path):
        raise SystemExit(f"no scenario {name}: see scenarios/")
    return path


# ------------------------------------------------------------------ commands

def cmd_env(a):
    rows = []

    def check(what, ok, enables, detail=""):
        rows.append((what, "ok" if ok else "MISSING", enables, detail))

    for venv, probe, enables in (("torax-venv", "import torax, jax; print('torax', torax.__version__, 'jax', jax.__version__)", "everything: TORAX, the harness, the front end, scoring"),
                                 ("freegsnke-venv", "import freegsnke; print(getattr(freegsnke, '__version__', 'ok'))", "the free boundary (free_boundary, every hands-off MAST run) and quench on MAST"),
                                 ("kn1d-venv", "import KN1DPy; print('ok')", "fuelling by neutrals (option fuelling kn1d_gauge, an oracle)"),
                                 ("fairmast-venv", "import zarr; print(zarr.__version__)", "downloading FAIR-MAST data (fairmast.py, gas_extract.py, beam_species.py)")):
        py = tool_python(venv)
        if not os.path.exists(py):
            check(venv, False, enables, f"not found: set {TOOLS[venv]} or make external/{venv}")
            continue
        try:
            out = subprocess.run([py, "-I", "-c", probe], capture_output=True, text=True, timeout=180)
            check(venv, out.returncode == 0, enables, (out.stdout.strip() or out.stderr.strip()[-120:]))
        except Exception as err:
            check(venv, False, enables, repr(err)[:120])
    check("data/epednn", os.path.exists(os.path.join(HERE, "data", "epednn", "EPED1NNensemble.json")), "EPED-NN's weights (pedestal_model, the coupled pedestal); shipped with the repository")
    files = (("runs/twin/mast_natural2/laws.json", "hands-off MAST runs (natural.py; read even by the general versions)"),
             ("runs/twin/freeboundary/mast_machine2/active.pickle", "FreeGSNKE's MAST machine"),
             ("runs/twin/beam_species.json", "each shot's beam energy split (beam_options species measured)"),
             ("runs/twin/sensors/geometry.json", "the synthetic magnetic sensors (quench mode full)"),
             ("runs/twin/dev_shots.json", "optional: a development-shot list; when present, runs and scores refuse any other shot"),
             ("solver/ntm.py", "the 2D island solver (scenario key plant2d); not part of this release"))
    for f, enables in files:
        check(f, os.path.exists(os.path.join(ROOT, f)), enables)
    nat = natural()
    usable = shot_table(nat.development_shots())
    check("MAST shot data", len(usable["runnable"]) > 0, "whole MAST shots (run, score, studio on a shot)", f"{len(usable['runnable'])} shots runnable, {len(usable['scorable'])} also scorable")
    wsl = shutil.which("wsl.exe") or shutil.which("wsl")
    dream_py = os.environ.get("XTORAX_DREAM_PYTHON", "python3")
    dream = os.environ.get("XTORAX_DREAMPATH", "$HOME/DREAM")
    if wsl or not WIN:
        try:
            probe = f"command -v {dream_py} >/dev/null 2>&1 && ls {dream}/py >/dev/null 2>&1 && echo yes || echo no"
            out = subprocess.run([wsl, "-e", "bash", "-lc", probe] if WIN else ["bash", "-lc", probe],
                                 capture_output=True, text=True, timeout=60, env=dict(os.environ, MSYS_NO_PATHCONV="1"))
            ok = "yes" in out.stdout
        except Exception:
            ok = False
        check("DREAM", ok, "quench off MAST (DREAM, in WSL on Windows); optional")
    else:
        check("DREAM", False, "quench off MAST (DREAM, in WSL on Windows); optional")
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", a.port))
        free = True
    except OSError:
        free = False
    finally:
        s.close()
    check(f"port {a.port}", free, "serve and studio", "" if free else "in use: something is already listening")
    w = max(len(r[0]) for r in rows)
    print(f"expanded TORAX package {PACKAGE}")
    for what, status, enables, detail in rows:
        print(f"{what:{w}s}  {status:7s}  {enables}" + (f"  [{detail}]" if detail else ""))
    print("Only torax-venv (this Python) and data/epednn are needed for the ITER and test scenarios that have no free boundary; the rest is optional (README.md, \"Optional pieces\").")
    return 0 if all(r[1] == "ok" for r in rows if r[0] in ("torax-venv", "data/epednn")) else 1


def shot_table(dev):
    R = os.path.join(ROOT, "runs", "twin")
    ids = lambda folder: {int(f[:-4]) for f in os.listdir(folder) if f.endswith(".npz") and f[:-4].isdigit()} if os.path.isdir(folder) else set()
    shots, coils, extras = ids(os.path.join(R, "mast_natural2", "shots")), ids(os.path.join(R, "mast_natural2", "coils")), ids(os.path.join(R, "mast_natural", "extras"))
    real_dir = os.path.join(R, "whole82", "real")
    real = {int(f[:-4]) for f in os.listdir(real_dir) if f[:-4].isdigit()} if os.path.isdir(real_dir) else set()
    runnable = sorted(shots & coils & extras & dev) if dev is not None else sorted(shots & coils & extras)
    return dict(runnable=runnable, scorable=sorted(set(runnable) & real))


def cmd_shots(a):
    t = shot_table(natural().development_shots())
    print(f"scorable shots ({len(t['scorable'])}): runs and whole_shot.py scores against MAST's measurements")
    print("  " + " ".join(str(s) for s in t["scorable"]))
    if a.all:
        rest = sorted(set(t["runnable"]) - set(t["scorable"]))
        print(f"runnable but not scorable ({len(rest)}): runs, but EFIT's signals for scoring were not read (core/whole_shot_efit.py)")
        print("  " + " ".join(str(s) for s in rest))
    return 0


def cmd_versions(a):
    nat = natural()
    print(f"package {PACKAGE}; plasma settings: version {nat.RECOMMENDED} recommended")
    for k in sorted(nat.VERSIONS, key=lambda v: float(v)):
        v = nat.VERSIONS[k]
        mark = "  <- recommended" if k == nat.RECOMMENDED else ""
        print(f"version {k}{mark}")
        print(f"   models {' '.join(v['models'])}; s_critical {v['s_critical']}, mixing {v['mixing']}")
        print("   overrides " + json.dumps(v["overrides"]))
    return 0


def cmd_run(a):
    nat = natural()
    shots = shot_table(nat.development_shots())["scorable"] if a.scorable else a.shots
    if not shots:
        raise SystemExit("name the shots, or --scorable for every scorable shot in the workspace")
    nat.check_development(shots)
    version = nat.RECOMMENDED if a.version in (None, "recommended") else a.version
    v = nat.VERSIONS[version]
    overrides = dict(v["overrides"], save_profiles=True, **(json.loads(a.extra) if a.extra else {}))
    tag = a.tag or f"v{version.replace('.', '_')}_{time.strftime('%m%d_%H%M')}"
    lanes = max(1, min(a.lanes, len(shots)))
    groups = [shots[i::lanes] for i in range(lanes)]
    tags = [tag] if lanes == 1 else [f"{tag}{chr(97 + i)}" for i in range(lanes)]
    print(f"version {version}{' with ' + a.extra if a.extra else ''}: {len(shots)} shot(s), {a.seeds} seed(s), {lanes} lane(s); results in runs/twin/mast_natural2/{', '.join(tags)}", flush=True)
    base = [PY, "-u", core("natural.py"), "run", "--models", *v["models"], "--seeds", str(a.seeds), "--s_critical", str(v["s_critical"]), "--mixing", str(v["mixing"]), "--overrides", json.dumps(overrides)]
    procs = []
    for t, group in zip(tags, groups):
        log = open(os.path.join(ROOT, "runs", "twin", f"xtorax_{t}.log"), "w") if lanes > 1 else None
        procs.append(subprocess.Popen(base + ["--tag", t, "--shots", *[str(s) for s in group]], cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT if log else None))
    codes = [p.wait() for p in procs]
    if lanes > 1:
        for t in tags:
            for line in open(os.path.join(ROOT, "runs", "twin", f"xtorax_{t}.log"), errors="replace"):
                if " steps " in line or " failed" in line:
                    print(line.rstrip()[:200])
    print(f"score with: python xtorax.py score {tag}")
    return max(codes)


def expand_tags(tag):
    if "," in tag:
        return tag
    found = [t for t in [tag] + [tag + chr(97 + i) for i in range(26)] if os.path.exists(os.path.join(NAT, t, "results.jsonl"))]
    if not found:
        raise SystemExit(f"no runs under runs/twin/mast_natural2/{tag} (or {tag}a, {tag}b, ...)")
    return ",".join(found)


def cmd_score(a):
    tags = expand_tags(a.tag)
    name = a.name or a.tag.split(",")[0]
    print(f"scoring {tags} as {name}", flush=True)
    return run_py([core("whole_shot.py"), "--tags", tags, "--name", name], isolated=True)


def cmd_ready(a):
    nat = natural()
    version = nat.RECOMMENDED if a.version in (None, "recommended") else a.version
    return run_py([core("readiness.py"), a.name, "--version", version], isolated=True)


def cmd_compare(a):
    return run_py([core("whole_shot.py"), "--common", f"{a.a},{a.b}"], isolated=True)


def cmd_script(a):
    return run_py([core("harness.py"), "script", scenario_path(a.scenario), os.path.abspath(a.commands)])


def cmd_serve(a):
    return run_py([core("harness.py"), "serve", scenario_path(a.scenario), "--port", str(a.port)] + (["--allow-python"] if a.allow_python else []))


def cmd_call(a):
    return run_py([core("harness.py"), "call", a.json, "--port", str(a.port)])


def cmd_studio(a):
    if a.scenario is None:                               # a MAST shot if the workspace has one, else ITER
        a.scenario = "studio_mast_handsoff_20689" if 20689 in shot_table(None)["runnable"] else "studio_iter"
    name = os.path.basename(a.scenario)[:-5] if a.scenario.endswith(".json") else a.scenario
    print(f"the front end at http://127.0.0.1:{a.port}  (scenario {name}; ready in about a minute while TORAX's step compiles, then press Run)", flush=True)
    return run_py([core("studio.py"), "--scenario", name, "--port", str(a.port)] + (["--allow-python"] if a.allow_python else []))


SMOKE = (("reactor", ["check_reactor.py"], ()), ("plant", ["check_plant.py"], ()), ("pellet", ["check_pellet.py"], ()), ("edge", ["check_edge.py"], ()),
         ("coupled", ["check_coupled.py"], ("freegsnke",)), ("coupled_mast", ["check_coupled.py", "--mast"], ("freegsnke", "mast")),
         ("branches", ["check_branches.py", "--case", "mast", "--steps", "30"], ("freegsnke", "mast")))


def have(need):
    """Whether the workspace has an optional piece a check needs: FreeGSNKE, or MAST shot 20689's files."""
    if need == "freegsnke":
        return os.path.exists(tool_python("freegsnke-venv"))
    return 20689 in shot_table(None)["runnable"]


def cmd_smoke(a):
    folder = os.path.join(ROOT, "runs", "twin", "smoke")
    os.makedirs(folder, exist_ok=True)
    summary = []
    for name, args, needs in SMOKE:
        missing = [n for n in needs if not have(n)]
        if missing:
            line = f"{name} skipped: needs {' and '.join('FreeGSNKE' if n == 'freegsnke' else 'MAST shot data' for n in missing)}"
            print(line, flush=True)
            summary.append(line)
            continue
        t0 = time.time()
        log = os.path.join(folder, f"{name}.log")
        with open(log, "w") as fh:
            try:
                code = subprocess.call([PY, "-u", core(args[0]), *args[1:]], cwd=ROOT, env=ENV, stdout=fh, stderr=subprocess.STDOUT, timeout=a.limit)
            except subprocess.TimeoutExpired:
                code = 124
        last = [x for x in open(log, errors="replace").read().splitlines() if x.strip()]
        line = f"{name} exit {code} in {time.time() - t0:.0f} s :: {(last[-1] if last else '')[:200]}"
        print(line, flush=True)
        summary.append(line)
    summary.append("smoke_done")
    open(os.path.join(folder, "summary.txt"), "w").write("\n".join(summary) + "\n")
    bad = [s for s in summary if " exit " in s and " exit 0 " not in s]
    print("all checks pass" if not bad else f"{len(bad)} check(s) failed: see runs/twin/smoke/<name>.log")
    return 0 if not bad else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)   # --help prints the command table above
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("env")
    p.add_argument("--port", type=int, default=8801)
    p.set_defaults(f=cmd_env)
    sub.add_parser("versions").set_defaults(f=cmd_versions)
    p = sub.add_parser("shots")
    p.add_argument("--all", action="store_true")
    p.set_defaults(f=cmd_shots)
    p = sub.add_parser("run")
    p.add_argument("shots", type=int, nargs="*")
    p.add_argument("--scorable", action="store_true", help="every scorable shot in the workspace (25 in the development workspace)")
    p.add_argument("--version", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--lanes", type=int, default=1, help="processes in parallel; at most 3 (memory)")
    p.add_argument("--extra", default=None, help="JSON: override keys added to the version's")
    p.set_defaults(f=cmd_run)
    p = sub.add_parser("score")
    p.add_argument("tag")
    p.add_argument("--name", default=None)
    p.set_defaults(f=cmd_score)
    p = sub.add_parser("ready")
    p.add_argument("name")
    p.add_argument("--version", default=None)
    p.set_defaults(f=cmd_ready)
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(f=cmd_compare)
    p = sub.add_parser("script")
    p.add_argument("scenario")
    p.add_argument("commands")
    p.set_defaults(f=cmd_script)
    p = sub.add_parser("serve")
    p.add_argument("scenario")
    p.add_argument("--port", type=int, default=8801)
    p.add_argument("--allow-python", action="store_true", help="accept decision {\"kind\": \"python\"}, which runs a named Python function")
    p.set_defaults(f=cmd_serve)
    p = sub.add_parser("call")
    p.add_argument("json")
    p.add_argument("--port", type=int, default=8801)
    p.set_defaults(f=cmd_call)
    p = sub.add_parser("studio")
    p.add_argument("--scenario", default=None, help="default: MAST shot 20689 hands-off if the workspace has it, else studio_iter")
    p.add_argument("--port", type=int, default=8801)
    p.add_argument("--allow-python", action="store_true", help="accept decision {\"kind\": \"python\"}, which runs a named Python function")
    p.set_defaults(f=cmd_studio)
    p = sub.add_parser("smoke")
    p.add_argument("--limit", type=int, default=3600, help="seconds per check")
    p.set_defaults(f=cmd_smoke)
    a = ap.parse_args()
    if a.command == "run" and a.lanes > 3:
        raise SystemExit("at most 3 lanes: each MAST run with FreeGSNKE takes 2 to 3 GB")
    sys.exit(a.f(a))


if __name__ == "__main__":
    main()
