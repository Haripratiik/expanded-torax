"""TORAX as a harness (H39, docs/digital-twin.md): a plasma that a script or an AI agent can edit while it runs.

A scenario (twin.py) sets the plasma; then named commands drive it: step, observe, island, kick, crash, elm, turbulence, turbulence_noise, heating, pedestal, brake, trip, sensor_bias, decision, snapshot, restore...
(twin.Harness.COMMANDS, or {"cmd": "help"}). Three ways in, with the same commands:
  script   a JSON list of commands, run in order, with each answer printed
  serve    a server on this machine only, for an agent or any program: POST a command as JSON to http://127.0.0.1:8801/, GET /observe, GET /help. A branch (snapshot, step, restore) is how an agent asks what-if
  call     one command to a running server, for an agent that works through a shell
The server runs one command at a time and takes no code, only the named commands.
Usage (TORAX's venv):
  python core/harness.py script scenarios/stress.json commands.json
  python core/harness.py serve scenarios/stress.json --port 8801
A scenario with a "hands_off" key (scenarios/studio_mast_handsoff_*.json) is a real MAST shot run hands-off (natural.hands_off), development shots only.
  python core/harness.py call '{"cmd": "step", "until": 25}' --port 8801
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer


def clean(x):
    """JSON has no NaN."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, float) and x != x:
        return None
    return x


def from_here(handler, port, post=False):
    """A request from this machine: the Host header names the loopback address and port (so a DNS name rebound to 127.0.0.1
    does not reach the server), and a POST that carries an Origin (every browser's does) comes from the server's own page."""
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    if handler.headers.get("Host", "") not in hosts:
        return False
    origin = handler.headers.get("Origin")
    return not (post and origin is not None and origin not in {"http://" + h for h in hosts})


def python_decision(cmd):
    """Whether a command asks for decision {"kind": "python"}, which imports and calls a function by name."""
    spec = cmd.get("spec") if isinstance(cmd, dict) else None
    return isinstance(cmd, dict) and cmd.get("cmd") == "decision" and isinstance(spec, dict) and spec.get("kind") == "python"


REFUSED_PYTHON = "decision kind python runs a named Python function; the server refuses it unless started with --allow-python"


def serve(h, port, allow_python=False):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, obj, code=200):
            body = json.dumps(clean(obj)).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not from_here(self, port):
                return self._send(dict(error="this server answers requests from this machine only"), 403)
            self._send(h.observe() if self.path.startswith("/observe") else h.COMMANDS)

        def do_POST(self):
            if not from_here(self, port, post=True):
                return self._send(dict(error="this server answers requests from this machine only"), 403)
            try:
                cmd = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if python_decision(cmd) and not allow_python:
                    return self._send(dict(error=REFUSED_PYTHON), 403)
                self._send(h.apply(cmd))
            except Exception as err:
                self._send(dict(error=str(err)), 400)

        def log_message(self, *args):
            pass
    print(f"harness ready at http://127.0.0.1:{port}/ (t = {h.reactor.t:.1f} s)", flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("how", choices=["script", "serve", "call"])
    ap.add_argument("target", help="scenario file; for call, the command as JSON")
    ap.add_argument("commands", nargs="?", default=None, help="script: a JSON list of commands")
    ap.add_argument("--port", type=int, default=8801)
    ap.add_argument("--allow-python", action="store_true", help="serve: accept decision {\"kind\": \"python\"}, which runs a named Python function")
    args = ap.parse_args()
    if args.how == "call":
        req = urllib.request.Request(f"http://127.0.0.1:{args.port}/", data=args.target.encode(), headers={"Content-Type": "application/json"})
        print(urllib.request.urlopen(req, timeout=600).read().decode())
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    scenario = json.load(open(args.target))
    if scenario.get("hands_off"):          # a real MAST shot from its start, by itself (natural.hands_off, as the front end builds it); without this the harness ran its default plasma (H88)
        import natural
        h = natural.hands_off(**scenario["hands_off"])
    else:
        import twin
        h = twin.Harness(scenario, keep_history=False)
    try:
        if args.how == "serve":
            serve(h, args.port, allow_python=args.allow_python)
        else:
            t0 = time.perf_counter()
            for c in json.load(open(args.commands)):
                print(json.dumps(clean(dict(command=c, answer=h.apply(c)))), flush=True)
            print(f"{time.perf_counter() - t0:.0f} s", flush=True)
    finally:
        if getattr(h, "free", None) is not None:      # FreeGSNKE's server, a process of its own
            try:
                h.free.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
