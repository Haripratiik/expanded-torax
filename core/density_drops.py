"""MAST's sudden density losses (H48): line density falling by a quarter or more within 30 ms with beam power steady. How many, what goes with them, whether the state warns of them, how the runs compare, and what was tried. Read-only on the shot files and the runs; the numbers are in docs/digital-twin.md.
Usage (TORAX's venv): python core/density_drops.py events|hazard|runs|split|h_end|balance|figure
"""
import glob
import json
import os
import sys

import numpy as np

os.chdir(os.path.abspath(os.environ.get("XTORAX_HOME") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))   # the workspace
D = "runs/twin/mast_natural2"
CAT = "runs/twin/mast_catalog"
EXTRAS = "runs/twin/mast_natural/extras"
LEADS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
mu0 = 4e-7 * np.pi


def catalog():
    return json.load(open(f"{CAT}/catalog.json"))["shots"]


def shots():
    return sorted(int(os.path.basename(p)[:-4]) for p in glob.glob(f"{D}/shots/*.npz"))


def load(s):
    return np.load(f"{D}/shots/{s}.npz", allow_pickle=False)


def sound(g):
    """No interferometer on some shots, and some read 100 times too high."""
    nel = g["nel"]
    return bool(np.isfinite(nel).any() and np.nanmax(nel) <= 8e19)


def drops(t, nel):
    dt = np.median(np.diff(t))
    lag = max(int(round(0.03 / dt)), 1)
    out, i, last = [], lag, -1.0
    while i < len(t):
        if nel[i - lag] > 0 and nel[i] / nel[i - lag] < 0.75 and t[i] - last > 0.05:
            out.append((i - lag, i))
            last = float(t[i])
            i += lag
        else:
            i += 1
    return out


def find_events():
    cat = catalog()
    out = []
    n_sound = 0
    for s in shots():
        g = load(s)
        if not sound(g):
            continue
        n_sound += 1
        t, nel = g["t"], np.nan_to_num(g["nel"])
        P = np.nan_to_num(g["P_heat"])
        x = cat.get(str(s), {})
        sw = x.get("sawteeth") or {}
        ct, cd = np.array(sw.get("crash_t") or []), np.array(sw.get("crash_drop") or [])
        windows = x.get("phases", {}).get("hmode", {}).get("windows") or []
        modes = x.get("modes") or []
        for a, b in drops(t, nel):
            if not (P[a] > 0.5 and abs(P[b] / max(P[a], 0.05) - 1) < 0.25) or t[-1] - t[b] < 0.02:      # beam power steady, and not the last moment of the data
                continue
            near = (ct >= t[a] - 0.005) & (ct <= t[b] + 0.005)
            late = min(b + 2, len(t) - 1)
            n_core, T_core = g["n_e"][:, 0], g["T_e"][:, 0]
            out.append(dict(shot=s, t=float(t[b]), ratio=float(nel[b] / nel[a]), W_th=float(g["W_th"][late] / g["W_th"][a]), W_dia=float(g["W_dia"][late] / g["W_dia"][a]), T_e0=float(T_core[late] / T_core[a]), n_e0=float(n_core[late] / n_core[a]), V_up=float(np.nan_to_num(g["V_loop"][b]) - np.nan_to_num(g["V_loop"][a])), dW=float(g["W_dia"][b] / g["W_dia"][a] - 1) if g["W_dia"][a] > 0 else np.nan,
                            crash=float(cd[near].max()) if near.any() else 0.0, h_end=any(t[a] - 0.01 <= w["t1"] <= t[b] + 0.01 for w in windows), mode=sum(t[a] - 0.02 <= m["t_on"] <= t[b] + 0.005 for m in modes),
                            lock=sum(bool(m.get("locked")) and t[a] - 0.02 <= (m.get("t_lock") or -1) <= t[b] + 0.005 for m in modes)))
    return out, n_sound


def events():
    ev, n_sound = find_events()
    print(f"{len(ev)} drops on {len({e['shot'] for e in ev})} of {n_sound} shots with sound line density ({len(shots())} converted)")
    print("time, median", round(float(np.median([e["t"] for e in ev])), 3), "IQR", np.percentile([e["t"] for e in ev], [25, 75]).round(3).tolist(), "; to, median", round(float(np.median([e["ratio"] for e in ev])), 2), "of the level before")
    for key, name in (("W_th", "thermal energy, from Thomson and charge exchange"), ("W_dia", "EFIT's diamagnetic energy"), ("T_e0", "T_e on axis"), ("n_e0", "n_e on axis")):
        v = np.array([e[key] for e in ev], float)
        print(f"  {name}, from the start of the fall to 10 ms after it: median {np.nanmedian(v):.2f} of its level, IQR {np.nanpercentile(v, 25):.2f} to {np.nanpercentile(v, 75):.2f}")
    for name, f in (("loop voltage up 0.3 V or more", lambda e: e["V_up"] > 0.3), ("large core crash (over 30%)", lambda e: e["crash"] > 0.3), ("any core crash", lambda e: e["crash"] > 0), ("end of an H phase", lambda e: e["h_end"]), ("mode onset", lambda e: e["mode"] > 0), ("lock", lambda e: e["lock"] > 0), ("stored energy down 5% or more", lambda e: e["dW"] < -0.05), ("none of crash, H end, mode, lock", lambda e: e["crash"] == 0 and not e["h_end"] and e["mode"] == 0 and e["lock"] == 0)):
        print(f"  {name}: {sum(f(e) for e in ev)} of {len(ev)}")


def hazard():
    ev, _ = find_events()
    rows = []
    for s in shots():
        g = load(s)
        if not sound(g):
            continue
        t = g["t"]
        V = g["V_rho"][:, -1] if g["V_rho"].ndim == 2 else g["V_rho"]
        a, B, Ip = g["a"], np.nan_to_num(g["B0"]), g["Ip"]
        beta_T = 100 * (2 / 3) * g["W_dia"] / np.maximum(V, 0.1) / np.maximum(B ** 2 / (2 * mu0), 1.0)
        beta_N = beta_T * a * B / np.maximum(Ip / 1e6, 0.05)
        fG = g["nel"] / np.maximum((Ip / 1e6) / (np.pi * a ** 2) * 1e20, 1e18)
        on = np.nan_to_num(g["P_heat"]) > 0.5
        times = [e["t"] for e in ev if e["shot"] == s]
        for i in np.flatnonzero(on):
            rows.append((t[i], beta_N[i], fG[i], g["q95"][i], g["q0"][i], g["nel"][i] / 1e19, Ip[i] / 1e6, any(0.02 <= x - t[i] <= 0.05 for x in times), any(x - 0.02 < t[i] for x in times)))
    R = np.array(rows, float)
    before, after = R[:, 7] > 0, R[:, 8] > 0
    use = ~after | before             # at risk: before the first drop of the shot, or just before any drop
    X, Y = R[use], before[use]

    def rank(x, y):
        good = np.isfinite(x)
        x, y = x[good], y[good]
        r = np.empty(len(x))
        r[np.argsort(x)] = np.arange(len(x))
        n1, n0 = y.sum(), (~y).sum()
        return (r[y].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)

    print("the 20 to 50 ms before a drop, ranked against other moments at risk (0.5 is no warning)")
    for k, c in dict(time=0, beta_N=1, f_Greenwald=2, q95=3, q0=4, line_density=5, Ip=6).items():
        v = rank(X[:, c], Y)
        print(f"  {k:13s} {max(v, 1 - v):.2f} ({'high' if v > 0.5 else 'low'} before a drop); median before {np.nanmedian(X[Y, c]):.3f}, otherwise {np.nanmedian(X[~Y, c]):.3f}")
    dt = 0.005
    print("hazard, per s, by beta_N from EFIT's stored energy, fast ions included:")
    for lo, hi in ((0, 3), (3, 4), (4, 5), (5, 99)):
        m = (X[:, 1] >= lo) & (X[:, 1] < hi)
        print(f"  {lo} to {hi}: {m.sum() * dt:5.2f} s of plasma, {Y[m].sum() / 6:5.1f} events, {(Y[m].sum() / 6) / max(m.sum() * dt, 1e-3):5.1f} per s")
    print("by time in the shot:")
    for lo, hi in ((0, .15), (.15, .2), (.2, .25), (.25, .3), (.3, .35), (.35, .4), (.4, .6)):
        m = (X[:, 0] >= lo) & (X[:, 0] < hi)
        print(f"  {lo} to {hi}: {m.sum() * dt:5.2f} s of plasma, {Y[m].sum() / 6:5.1f} events, {(Y[m].sum() / 6) / max(m.sum() * dt, 1e-3):5.1f} per s")


def runs():
    for tag in ("v22h", "v1h", "v21f", "v1f"):
        rows = [json.loads(line) for line in open(f"{D}/{tag}/results.jsonl")]
        n_sim, n_real, after, seen, seeds = 0, 0, [], set(), 2
        for r in rows:
            p = f"{D}/{tag}/{r['shot']}_{r['seed']}.npz"
            if r["steps"] < 5 or not os.path.exists(p):
                continue
            d = np.load(p, allow_pickle=False)
            t, nl = d["t"], np.nan_to_num(d["n_line"])
            n_sim += len(drops(t, nl)) / seeds
            for x, on in r["h_log"]:
                if not on and x + 0.03 <= t[-1] and x - 0.002 >= t[0] and nl[np.argmin(np.abs(t - x))] > 0:
                    after.append(nl[np.argmin(np.abs(t - (x + 0.03)))] / nl[np.argmin(np.abs(t - x))])
            if r["shot"] not in seen:
                seen.add(r["shot"])
                g = load(r["shot"])
                m = (g["t"] >= t[0]) & (g["t"] <= t[-1])
                if sound(g) and m.sum() > 5:
                    n_real += len(drops(g["t"][m], np.nan_to_num(g["nel"][m])))
        print(f"{tag}: {len(seen)} shots; drops per shot, runs {n_sim / len(seen):.2f}, real {n_real / len(seen):.2f}; after a run's own end of H, line density at 30 ms: {len(after)} cases, median {np.median(after):.2f}")
    cat = catalog()
    ratio, ratio_w = [], []
    for s in shots():
        g = load(s)
        if not sound(g):
            continue
        for w in cat.get(str(s), {}).get("phases", {}).get("hmode", {}).get("windows") or []:
            if g["t"][0] < w["t1"] and w["t1"] + 0.03 < g["t"][-1]:
                a, b = np.interp([w["t1"], w["t1"] + 0.03], g["t"], g["nel"])
                c, e = np.interp([w["t1"], w["t1"] + 0.03], g["t"], g["W_dia"])
                if a > 0:
                    ratio.append(b / a)
                    ratio_w.append(e / c)
    print(f"real ends of H with 30 ms of data after: {len(ratio)}; line density at {np.median(ratio):.2f}, IQR {np.percentile(ratio, 25):.2f} to {np.percentile(ratio, 75):.2f}; stored energy at {np.median(ratio_w):.2f}")


def split():
    tables = {}
    for tag in ("v22h", "v1h"):
        per = {}
        for line in open(f"{D}/{tag}/results.jsonl"):
            r = json.loads(line)
            if "profile" in r:
                per.setdefault(r["shot"], []).append(r)
        tables[tag] = per
    when = {}
    for s, g in tables["v22h"].items():
        z = load(s)
        when[s] = None if not sound(z) else [z["t"][b] - g[0]["t_cut"] for a, b in drops(z["t"], np.nan_to_num(z["nel"])) if z["t"][b] > g[0]["t_cut"]]
    for lead in (0.1, 0.15, 0.2):
        i = LEADS.index(lead)
        groups = {"with a drop before this time": [], "without": [], "line density not usable": []}
        for s, evs in when.items():
            a = [r["profile"]["err_at_s"]["n_e"][i] for r in tables["v22h"][s] if r["profile"]["err_at_s"]["n_e"][i] is not None]
            b = [r["profile"]["err_at_s"]["n_e"][i] for r in tables["v1h"].get(s, []) if r["profile"]["err_at_s"]["n_e"][i] is not None]
            c = [r["held"]["err_at_s"]["n_e"][i] for r in tables["v22h"][s] if r["held"]["err_at_s"]["n_e"][i] is not None]
            if a and b and c:
                groups["line density not usable" if evs is None else "with a drop before this time" if any(e <= lead for e in evs) else "without"].append((s, 100 * np.mean(a), 100 * np.mean(b), 100 * np.mean(c)))
        print(f"n_e at {int(1e3 * lead)} ms, percent, mean (median):")
        for k, v in groups.items():
            if v:
                A = np.array([x[1:] for x in v])
                print(f"  {k}, {len(v)} shots {[x[0] for x in v]}: version 2.2 {A[:, 0].mean():.1f} ({np.median(A[:, 0]):.1f}); version 1 {A[:, 1].mean():.1f} ({np.median(A[:, 1]):.1f}); holding {A[:, 2].mean():.1f} ({np.median(A[:, 2]):.1f})")


def h_end():
    """With H given as measured, on the fit shots: tag hs0, the fuelling law only adds; hs1, it also takes out."""
    cat = catalog()
    for tag in ("hs0", "hs1"):
        real, sim = [], []
        for line in open(f"{D}/{tag}/results.jsonl"):
            r = json.loads(line)
            s = r["shot"]
            d, g = np.load(f"{D}/{tag}/{s}_{r['seed']}.npz", allow_pickle=False), load(s)
            if not sound(g):
                continue
            t, nl = d["t"], np.nan_to_num(d["n_line"])
            for w in cat[str(s)]["phases"]["hmode"]["windows"]:
                e = w["t1"]
                if e - 0.002 >= t[0] and e + 0.03 <= min(t[-1], g["t"][-1]):
                    a, b = np.interp([e, e + 0.03], g["t"], g["nel"])
                    c, f = np.interp([e, e + 0.03], t, nl)
                    real.append(b / a)
                    sim.append(f / c)
        print(f"{tag}: {len(real)} ends of H; line density 30 ms later, real {np.median(real):.2f}, run {np.median(sim):.2f} (IQR {np.percentile(sim, 25):.2f} to {np.percentile(sim, 75):.2f})")


def balance():
    """0-D particle balance against the fitted density law: dN/dt = eta * gas + beams - N / tau(H). Fit eta, tau_L, tau_H on the 19 fit shots only; H given as measured to both, so this asks only about the density's own physics."""
    cat = catalog()
    nat = json.load(open(f"{CAT}/natural_test_shots.json"))
    fresh1 = json.load(open(f"{CAT}/natural_fresh_shots.json"))["shots"]
    fresh2 = json.load(open(f"{CAT}/natural_fresh2_shots.json"))["shots"]
    fit = [x["shot"] for x in nat["shots"] if x["split"] == "fit"]
    held = [x["shot"] for x in nat["shots"] if x["split"] != "fit"]
    law_N = json.load(open(f"{D}/laws.json"))["pedestal"]["N"]
    E_b = 65e3 * 1.602e-19
    picks = (0.02, 0.05, 0.1, 0.15, 0.2)
    cut_of = {}
    for tag in ("v22h", "v21f", "v2held", "src_a"):
        p = f"{D}/{tag}/results.jsonl"
        if os.path.exists(p):
            for line in open(p):
                r = json.loads(line)
                cut_of.setdefault(r["shot"], r["t_cut"])

    def prepare(s):
        if s not in cut_of:                    # two drawn shots could not be converted
            return None
        g = load(s)
        if not sound(g):
            return None
        e = dict(np.load(f"{EXTRAS}/{s}.npz", allow_pickle=False))
        t = np.arange(cut_of[s], g["t"][-1], 0.001)
        V = g["V_rho"][:, -1] if g["V_rho"].ndim == 2 else g["V_rho"]
        gas = np.interp(t, e["t_gas"], np.convolve(np.nan_to_num(e["gas_total_injected"]), np.ones(11) / 11, mode="same"))
        H = np.zeros_like(t, bool)
        for w in cat.get(str(s), {}).get("phases", {}).get("hmode", {}).get("windows") or []:
            H |= (t >= w["t0"]) & (t <= w["t1"])
        beam = np.interp(t, e["t_sum"], np.nan_to_num(e["power_nbi"])) / E_b
        return dict(t=t, real=np.interp(t, g["t"], np.nan_to_num(g["nel"])) * np.interp(t, g["t"], np.nan_to_num(V)), gas=np.gradient(gas, t).clip(0), beam=beam, H=H, Ip=np.interp(t, e["t_prog"], np.nan_to_num(e["prog_i_plasma"])), prog=np.interp(t, e["t_prog"], np.nan_to_num(e["prog_n_e_line"])) if "prog_n_e_line" in e else None, P=np.maximum(np.interp(t, g["t"], np.nan_to_num(g["P_heat"])), 0.05))

    def by_balance(x, eta, tL, tH):
        N = np.zeros_like(x["t"])
        N[0] = x["real"][0]
        for i in range(1, len(N)):
            N[i] = N[i - 1] + 0.001 * (eta * x["gas"][i - 1] + x["beam"][i - 1] - N[i - 1] / (tH if x["H"][i - 1] else tL))
        return N

    def by_law(x, signed=False, tau=0.03):
        kind = "programme" if x["prog"] is not None and x["prog"][0] > 0 else "no_programme"
        L, N = law_N[kind], np.zeros_like(x["t"])
        N[0] = x["real"][0]
        for i in range(1, len(N)):
            d = dict(Ip=np.log(max(x["Ip"][i], 1e4) / max(x["Ip"][0], 1e4)), P=np.log(x["P"][i] / x["P"][0]), H=float(x["H"][i]) - float(x["H"][0]))
            if kind == "programme":
                d["programme"] = np.log(max(x["prog"][i], 1e-3) / x["prog"][0])
            rate = N[i - 1] * (np.clip(sum(L[a] * d[a] for a in L), -1.5, 1.5) + np.log(N[0] / N[i - 1])) / tau
            N[i] = N[i - 1] + 0.001 * (rate if signed else max(rate, 0.0))
        return N

    def err(x, N):
        out = []
        for lead in picks:
            i = int(round(lead / 0.001))
            out.append(abs(np.log(N[i] / x["real"][i])) if i < len(N) and x["real"][i] > 0 and N[i] > 0 else np.nan)
        return out

    sets = {name: [x for x in (prepare(s) for s in group) if x is not None] for name, group in (("fit", fit), ("held out", held), ("fresh 1", [x["shot"] for x in fresh1]), ("fresh 2", [x["shot"] for x in fresh2]))}
    best = None
    for eta in (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4):
        for tL in (0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3):
            for tH in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0):
                e = float(np.nanmean([np.nanmean(err(x, by_balance(x, eta, tL, tH))) for x in sets["fit"]]))
                if best is None or e < best[0]:
                    best = (e, eta, tL, tH)
    print("best on the fit shots: eta, tau_L, tau_H", best[1:])
    for name, X in sets.items():
        print(f"{name}, {len(X)} shots: mean |ln N|, percent, at {[int(1e3 * a) for a in picks]} ms")
        for label, fn in (("hold", lambda x: np.full_like(x["t"], x["real"][0])), ("fitted law, adds only", by_law), ("fitted law, both ways", lambda x: by_law(x, signed=True)), ("balance", lambda x: by_balance(x, *best[1:]))):
            print(f"  {label:24s}", np.round(100 * np.nanmean([err(x, fn(x)) for x in X], axis=0), 1).tolist())


def figure():
    """Six shots of the second clean test: three whose real density falls suddenly, three whose does not. Writes docs/figures/h48_density_drops.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, NEW, REAL = "#1f2328", "#6a7178", "#c8553d", "#1f5fa8"
    picks = (23906, 24801, 28050, 28825, 22115, 23867)
    fig, axes = plt.subplots(2, len(picks), figsize=(13, 5), sharex="col", constrained_layout=True)
    for j, s in enumerate(picks):
        g = load(s)
        for seed in (0, 1):
            new, old = np.load(f"{D}/v22h/{s}_{seed}.npz", allow_pickle=False), np.load(f"{D}/v1h/{s}_{seed}.npz", allow_pickle=False)
            m = (g["t"] >= new["t"][0]) & (g["t"] <= new["t"][-1])
            for a, real, key, scale, name in ((axes[0, j], g["nel"], "n_line", 1e19, "Line density (1e19 m^-3)"), (axes[1, j], g["W_th"], "W_thermal", 1e3, "Thermal energy (kJ)")):
                a.plot(1e3 * old["t"], old[key] / scale, color=MUTED, lw=1.2, ls="--", label="version 1" if seed == 0 else None)
                a.plot(1e3 * new["t"], new[key] / scale, color=NEW, lw=1.8, label="version 2.2, hands-off" if seed == 0 else None)
                if seed == 0:
                    a.plot(1e3 * g["t"][m], real[m] / scale, color=REAL, lw=2, label="measured")
                    a.spines[["top", "right"]].set_visible(False)
                    a.tick_params(colors=MUTED, labelsize=8)
                    a.grid(axis="y", color="#e6e8eb", lw=0.7)
                    a.set_ylim(bottom=0)
                    if j == 0:
                        a.set_ylabel(name, color=INK, fontsize=9)
        axes[0, j].set_title(f"shot {s}" + (", sudden drop" if j < 3 else ", none"), color=INK, fontsize=10)
        axes[1, j].set_xlabel("time (ms)", color=MUTED, fontsize=8)
    axes[0, 3].legend(frameon=False, fontsize=8, loc="upper left")
    path = "docs/figures/h48_density_drops.png"
    fig.savefig(path, dpi=130)
    print("written", path)


if __name__ == "__main__":
    dict(events=events, hazard=hazard, runs=runs, split=split, h_end=h_end, balance=balance, figure=figure)[sys.argv[1]]()
