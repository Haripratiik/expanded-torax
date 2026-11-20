// The harness's front end: polls the simulation thread (studio.py), draws the shot, sends commands. No framework, no build step; everything here is canvases and a few lists.
"use strict";
const $ = id => document.getElementById(id);
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const MONO = '9.5px "B612 Mono", Consolas, monospace';
const fmt = (v, d = 3) => v == null || !isFinite(v) ? "-" : Math.abs(v) >= 1e4 || (v !== 0 && Math.abs(v) < 1e-2) ? v.toExponential(2) : +v.toPrecision(d) + "";

const S = { rev: -1, n: 0, nf: 0, t: [], data: {}, frames: [], meta: null, st: {}, cursor: null, pinned: false, window: 30, zoom: null, active: [], heat: "T_e", tab: "Instabilities", ghost: null, dirty: true, pf4: "chi", measured: null, measuredAt: null, flash: { elm: -1e9, saw: -1e9 }, hits: {}, counts: {} };

// what can be drawn: a key, its label, how to scale it, and its pen, which stays the key's whatever else is on the strip
const SIG = {
  P_fusion: ["fusion", 1e-6, 1], P_SOL: ["exhaust", 1e-6, 2], P_rad: ["radiated", 1e-6, 3], P_alpha: ["alpha", 1e-6, 4], P_LH: ["L-H threshold", 1e-6, 7], P_ohmic: ["ohmic", 1e-6, 5],
  W_thermal: ["thermal", 1e-6, 1], H: ["H-mode", 1, 1], H98: ["H98", 1, 1], Q_fusion: ["Q", 1, 2], P_cmd: ["ECCD delivered", 1, 2], aim_cmd: ["aim", 1, 2], T_e0: ["T_e axis", 1, 1], T_i0: ["T_i axis", 1, 2], n_line: ["line density", 1e-20, 1], f_GW: ["Greenwald fraction", 1, 3],
  beta_N: ["beta_N", 1, 1], q95: ["q95", 1, 2], li3: ["li3", 1, 7], f_bs: ["bootstrap fraction", 1, 3], Z_eff_avg: ["Z_eff mean", 1, 7], T_e_target: ["target T_e", 1, 2], q_target: ["target heat flux", 1, 8], T_e_sep: ["separatrix T_e", 1, 1], gas_rate: ["gas", 1e-21, 1], seed_rate: ["seeding", 1e-21, 3], A_rwm: ["RWM amplitude", 1, 8], z_cm: ["vertical position", 1, 5], flux_2d: ["2D island flux", 1, 1], a_2d: ["2D drive", 1, 2], beam_neutrons: ["beam-target", 1e-13, 1], beam_W_fast: ["held by fast ions", 1e-6, 1], beam_through: ["share lost through", 1, 2],
};
const STRIPS = {
  power: { title: "Power", unit: "MW", keys: ["P_fusion", "P_alpha", "P_SOL", "P_rad"], on: true },
  energy: { title: "Stored energy", unit: "MJ", keys: ["W_thermal"], on: true, small: true },
  islands: { title: "Island width", unit: "cm", dyn: "w_", on: true },
  drive: { title: "Gyrotrons", unit: "MW", keys: ["P_cmd"], on: true, small: true, step: true },
  core: { title: "Core temperature", unit: "keV", keys: ["T_e0", "T_i0"], on: true, small: true },
  density: { title: "Density", unit: "1e20 m^-3", keys: ["n_line"], on: true, small: true },
  hmode: { title: "H-mode", unit: "1 = on", keys: ["H"], small: true, step: true },
  rotation: { title: "Island rotation", unit: "Hz", dyn: "f_", small: true, freeY: true },
  aim: { title: "Mirror aim", unit: "rho", keys: ["aim_cmd"], small: true, freeY: true },
  limits: { title: "Greenwald fraction", unit: "", keys: ["f_GW"], small: true },
  beta: { title: "Normalized beta", unit: "", keys: ["beta_N"], small: true },
  conf: { title: "Confinement", unit: "", keys: ["H98"], small: true },
  gain: { title: "Fusion gain", unit: "", keys: ["Q_fusion"], small: true },
  q: { title: "Edge safety factor", unit: "", keys: ["q95"], small: true, freeY: true },
  zeff: { title: "Z_eff", unit: "", keys: ["Z_eff_avg"], small: true, freeY: true },
  target: { title: "Divertor target", unit: "eV, MW/m^2", keys: ["T_e_target", "q_target"], small: true },
  gas: { title: "Gas and seeding", unit: "1e21 /s", keys: ["gas_rate", "seed_rate"], small: true },
  stab: { title: "Wall mode", unit: "", keys: ["A_rwm"], small: true },
  vert: { title: "Vertical", unit: "cm", keys: ["z_cm"], small: true, freeY: true },
  flux2d: { title: "2D island", unit: "", keys: ["flux_2d", "a_2d"], small: true },
  neutrons: { title: "Neutrons", unit: "1e13 /s", keys: ["beam_neutrons"], small: true },
  fast: { title: "Fast ions", unit: "MJ", keys: ["beam_W_fast"], small: true },
  through: { title: "Beam shine-through", unit: "", keys: ["beam_through"], small: true },
};
const HEAT = { T_e: ["T_e", "keV"], T_i: ["T_i", "keV"], n_e: ["n_e", "1e20 m^-3"], q: ["q", ""], chi_e: ["chi_e", "m^2/s"], chi_i: ["chi_i", "m^2/s"], Z_eff: ["Z_eff", ""], j: ["current density", "MA/m^2"], rad: ["line radiation", "MW/m^3"] };
const LEVELS = { coupled: ["open-source tool", "An open-source code coupled to TORAX does the work: EPED-NN, FreeGSNKE."], torax: ["TORAX's own", "TORAX's own model does the work; this only changes its inputs."], reduced: ["reduced equation", "A published reduced equation, integrated here."], prescribed: ["prescribed effect", "An effect with assumed size and timing: asks what if, does not predict."], margin: ["0D margin", "A 0D model on TORAX's scalars, with no effect on the profiles until it ends the shot."] };

// ---------- talking to the server
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return r.json();
}
function toast(text) {
  const el = $("toast");
  el.textContent = text;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (el.hidden = true), 5000);
}
async function send(c) {
  const a = await api("/api/command", c);
  if (a.error) toast(a.error);
  else if (a.scheduled !== undefined) toast(`${c.cmd} scheduled at ${fmt(a.scheduled, 5)} s`);
  S.dirty = true;
  return a;
}
function reset() {
  S.n = 0; S.nf = 0; S.t = []; S.data = {}; S.frames = []; S.cursor = null; S.pinned = false; S.zoom = null; S.measured = null;
  $("pin-clear").hidden = true;
}
async function poll() {
  try {
    const q = await api(`/api/trace?since=${S.n}&frames=${S.nf}&rev=${S.rev}`);
    if (typeof q.rev !== "number") throw new Error(q.error || "no trace");   // an error reply, or a proxy's page while the server starts: keep our place
    if (q.rev !== S.rev) { reset(); S.rev = q.rev; if (S.meta && q.scenario !== S.meta.scenario) await loadMeta(); }
    merge(q);
    const ready = S.st.state === "loading" && q.state !== "loading";
    S.st = q;
    S.dirty = true;
    header(q);
    if (ready || (S.meta && !S.meta.machine && q.has_machine)) await loadMeta();          // a new plasma's geometry and channels, and the machine's drawing once FreeGSNKE is on
    if (S.meta && S.meta.measured && S.frames.length && !S.pinned) { const tt = S.frames[S.frames.length - 1].t; if (S.measuredAt == null || Math.abs(S.measuredAt - tt) > 0.004) { S.measuredAt = tt; S.measured = await api(`/api/measured?t=${tt}`); } }
  } catch (e) {
    $("state").dataset.s = "loading";
    $("state").lastElementChild.textContent = "no server";
  }
  setTimeout(poll, S.st.running ? 300 : 800);
}
function merge(q) {
  for (const row of q.rows || []) {
    const i = S.t.length;
    S.t.push(row.t);
    for (const k in row) {
      if (k === "t") continue;
      if (!S.data[k]) S.data[k] = new Array(i).fill(NaN);
      S.data[k].push(row[k] === null ? NaN : row[k]);
    }
    for (const k in S.data) if (S.data[k].length < i + 1) S.data[k].push(NaN);
    if (row.elm > 0) S.flash.elm = performance.now();
    if (row.sawtooth > 0) S.flash.saw = performance.now();
  }
  S.n = q.n ?? S.n;
  for (const f of q.frames || []) S.frames.push(f);
  S.nf = q.n_frames ?? S.nf;
}
async function loadMeta() {
  S.meta = await api("/api/meta");
  const sel = $("scenario");
  sel.replaceChildren(...S.meta.scenarios.map(n => { const o = document.createElement("option"); o.value = o.textContent = n; return o; }));
  sel.value = S.meta.scenario;
  $("decision").replaceChildren(...Object.keys(S.meta.decisions).map(n => { const o = document.createElement("option"); o.value = o.textContent = n; return o; }));
  buildConsole();
  buildStrips();
  S.real = S.meta.measured ? await api("/api/measured_traces") : null;
  if (S.meta.measured) { for (const k of ["neutrons", "fast", ...(S.real && S.real.H ? ["hmode"] : [])]) if (!S.active.includes(k)) S.active.push(k); layoutStrips(); S.window = 0; $("window").value = "0"; }
}

// ---------- the header
function header(q) {
  $("time").textContent = q.t == null ? "-" : q.t.toFixed(q.t < 5 ? 3 : 2);
  const s = q.state || "loading";
  $("state").dataset.s = s;
  $("state").lastElementChild.textContent = s === "ended" ? (q.message || "ended") : s;
  $("run").textContent = q.running ? "Pause" : "Run";
  $("run").dataset.running = !!q.running;
  $("overlay").hidden = s !== "loading";
  $("overlay-msg").textContent = q.message || "";
  const br = $("branches"), have = [...br.options].map(o => o.value).join();
  const now = (q.branches || []).map(b => b.name).join();
  if (have !== (now ? "" + now : "")) {
    br.replaceChildren(...(q.branches || []).map(b => { const o = document.createElement("option"); o.value = b.name; o.textContent = `${b.name} (${fmt(b.t, 4)} s)`; return o; }));
    if (!q.branches || !q.branches.length) { const o = document.createElement("option"); o.value = ""; o.textContent = "no marks"; br.append(o); }
  }
  loopPanel(q);
  paintPlane(q);
}

// ---------- the decision loop
function frameAt(t) {
  const F = S.frames;
  if (!F.length) return null;
  if (t == null) return F[F.length - 1];
  let a = 0, b = F.length - 1;
  while (b - a > 1) { const m = (a + b) >> 1; if (F[m].t <= t) a = m; else b = m; }
  return Math.abs(F[a].t - t) < Math.abs(F[b].t - t) ? F[a] : F[b];
}
function loopPanel(q) {
  const f = frameAt(S.pinned ? S.cursor : null);
  if (!f) return;
  const isl = (f.islands || []).filter(i => i.w > 0);
  $("loop-plasma").textContent = isl.length ? isl.map(i => `${i.m}/${i.n}: ${fmt(100 * i.w, 3)} cm at rho ${fmt(i.rho, 3)}` + (i.f != null ? `, ${fmt(i.f, 3)} Hz` : "")).join("; ") : "no island";
  const n = f.sensors || {};
  $("loop-sensors").textContent = `noise ${fmt(n.sigma_w_cm, 2)} cm, delay ${n.latency} step${n.latency === 1 ? "" : "s"}` + (n.faults ? `, ${n.faults} FAULT${n.faults > 1 ? "S" : ""} active` : "");
  $("loop-decision").textContent = q.decision_name || q.decision || "-";
  if (q.decision_name && q.decision_name in S.meta.decisions && document.activeElement !== $("decision")) $("decision").value = q.decision_name;
  const g = f.gyro || {};
  const on = g.unit_MW ? Math.round((g.P || 0) / g.unit_MW) : 0;
  $("loop-gyro").textContent = `${fmt(g.P, 3)} MW` + (g.aim != null ? ` at rho ${fmt(g.aim, 3)}` : "") + (g.lost ? `, ${g.lost} tripped` : "") + (g.stuck ? ", mirror STUCK" : "");
  const L = $("leds");
  if (L.children.length !== g.units) L.replaceChildren(...Array.from({ length: g.units || 0 }, () => document.createElement("i")));
  [...L.children].forEach((el, i) => { el.className = i >= g.units - g.lost ? "lost" : i < on ? "on" : ""; });
  const chips = [];
  const p = q.plant || {}, st = p.settings || {};
  if (st.density_control && st.density_control.target) chips.push(`density feedback ${st.density_control.target}`);
  if (st.radiation_control && st.radiation_control.target_MW) chips.push(`radiation feedback ${st.radiation_control.target_MW} MW`);
  if (st.mitigation && st.mitigation.armed) chips.push("mitigation armed");
  if (st.limits && st.limits.on) chips.push("limits watched");
  if (st.rmp && st.rmp.level) chips.push(`RMP ${st.rmp.level}`);
  for (const e of p.effects || []) chips.push(e);
  if (q.bursts) chips.push(`${q.bursts} turbulence burst${q.bursts > 1 ? "s" : ""}`);
  if (q.fields) chips.push("turbulence field");
  if (p.shards) chips.push("shards in flight");
  const lastFrame = S.frames && S.frames.length ? S.frames[S.frames.length - 1] : null;
  if (lastFrame && lastFrame.eped && lastFrame.eped.extrapolation > 0) chips.push(`EPED-NN is ${fmt(lastFrame.eped.extrapolation, 2)} of a range outside its training (${lastFrame.eped.worst})`);
  for (const [s, v] of Object.entries(p.impurity_atoms || {})) chips.push(`${s} ${fmt(v, 2)} atoms`);
  $("loops-on").replaceChildren(...(chips.length ? chips : ["nothing"]).map(c => { const li = document.createElement("li"); li.textContent = c; return li; }));
  card(p.card);
}
function card(c) {
  const el = $("card");
  el.hidden = !c;
  if (!c) return;
  el.replaceChildren();
  const h = document.createElement("h3"); h.textContent = `Shot ended: ${c.kind}`; el.append(h);
  const line = t => { const d = document.createElement("div"); d.textContent = t; el.append(d); };
  line(`thermal energy ${fmt(c.W_thermal_MJ)} MJ, magnetic ${fmt(c.W_magnetic_MJ)} MJ, ${fmt(c.I_p_MA)} MA`);
  if (c.radiated_share != null) line(`radiated ${fmt(100 * c.radiated_share, 3)}%, conducted ${fmt(c.E_conducted_MJ)} MJ`);
  if (c.quench) { const q = c.quench; line("current quench if the plasma settles at 10 / 20 / 40 eV: " + ["T_10eV", "T_20eV", "T_40eV"].map(k => fmt(q[k].tau_CQ_ms, 3)).join(" / ") + " ms (ITER wants 50 to 150)"); line("E over Connor-Hastie field: " + ["T_10eV", "T_20eV", "T_40eV"].map(k => fmt(q[k].E_over_Ec, 3)).join(" / ")); }
  const em = document.createElement("em"); em.textContent = c.note || ""; el.append(em);
}

// ---------- canvases
function fit(cv) {
  const r = cv.getBoundingClientRect(), d = devicePixelRatio || 1;
  const w = Math.max(10, Math.round(r.width)), h = Math.max(10, Math.round(r.height));
  if (cv.width !== w * d || cv.height !== h * d) { cv.width = w * d; cv.height = h * d; }
  const g = cv.getContext("2d");
  g.setTransform(d, 0, 0, d, 0, 0);
  g.clearRect(0, 0, w, h);
  return [g, w, h];
}
const PAD = { l: 42, r: 8 };
function range() {
  if (S.zoom) return S.zoom;
  const t = S.t;
  if (!t.length) return [0, 1];
  const b = t[t.length - 1];
  const a = S.window > 0 ? Math.max(t[0], b - S.window) : t[0];
  return a === b ? [a, a + 1e-3] : [a, b];
}
const X = (t, [a, b], w) => PAD.l + (t - a) / (b - a) * (w - PAD.l - PAD.r);
const tOf = (x, [a, b], w) => a + (x - PAD.l) / (w - PAD.l - PAD.r) * (b - a);
function lower(t, v) { let a = 0, b = t.length; while (a < b) { const m = (a + b) >> 1; if (t[m] < v) a = m + 1; else b = m; } return a; }
function ticks(a, b, n = 4) {
  if (!(b > a)) return [a];
  const raw = (b - a) / n, mag = 10 ** Math.floor(Math.log10(raw)), f = raw / mag, step = (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * mag, out = [];
  for (let x = Math.ceil(a / step) * step; x <= b + step * 1e-6; x += step) out.push(+x.toFixed(10));
  return out;
}
function hline(g, x0, x1, y) { g.beginPath(); g.moveTo(x0, Math.round(y) + .5); g.lineTo(x1, Math.round(y) + .5); g.stroke(); }
function vline(g, x, y0, y1) { g.beginPath(); g.moveTo(Math.round(x) + .5, y0); g.lineTo(Math.round(x) + .5, y1); g.stroke(); }

// draw one series, thinned to one low-high pair per pixel where the points outnumber the pixels
function pen(g, t, v, scale, R, w, Y, step) {
  const i0 = Math.max(lower(t, R[0]) - 1, 0), i1 = Math.min(lower(t, R[1]) + 1, t.length);
  g.beginPath();
  let down = false, last = null, lo = Infinity, hi = -Infinity, px = null;
  const flush = () => { if (px == null || lo === Infinity) return; if (!down) { g.moveTo(px, Y(lo)); down = true; } else g.lineTo(px, Y(lo)); if (hi !== lo) g.lineTo(px, Y(hi)); };
  for (let i = i0; i < i1; i++) {
    const y = v[i] * scale;
    if (!isFinite(y)) { flush(); px = null; lo = Infinity; hi = -Infinity; down = false; continue; }
    const x = Math.round(X(t[i], R, w) * 2) / 2;
    if (x !== px) { flush(); if (step && last != null && down) g.lineTo(x, Y(last)); px = x; lo = hi = y; } else { lo = Math.min(lo, y); hi = Math.max(hi, y); }
    last = y;
  }
  flush();
  g.stroke();
}

function buildStrips() {
  S.active = Object.entries(STRIPS).filter(([, d]) => d.on).map(([k]) => k);
  const menu = $("strip-menu");
  menu.replaceChildren(...Object.entries(STRIPS).map(([k, d]) => {
    const l = document.createElement("label"), c = document.createElement("input");
    c.type = "checkbox"; c.checked = !!d.on;
    c.onchange = () => { S.active = c.checked ? [...S.active, k] : S.active.filter(x => x !== k); layoutStrips(); };
    l.append(c, document.createTextNode(d.title));
    return l;
  }));
  $("heat-field").replaceChildren(...Object.entries(HEAT).map(([k, v]) => { const o = document.createElement("option"); o.value = k; o.textContent = v[0]; return o; }));
  $("heat-field").value = S.heat;
  layoutStrips();
}
function layoutStrips() {
  $("strips").replaceChildren(...S.active.map(k => {
    const d = STRIPS[k], box = document.createElement("div");
    box.className = "track strip" + (d.small ? " small" : "");
    box.dataset.k = k;
    const cv = document.createElement("canvas"), head = document.createElement("div");
    head.className = "track-head";
    box.append(head, cv);
    wire(cv);
    return box;
  }));
  S.dirty = true;
}
const keysOf = d => d.dyn ? Object.keys(S.data).filter(k => d.dyn === "w_" ? /^w_\d+_\d+$/.test(k) : /^f_\d+_\d+$/.test(k)).sort() : d.keys.filter(k => k in S.data);
function labelOf(k) { const m = k.match(/^[wf]_(\d+)_(\d+)$/); return m ? `${m[1]}/${m[2]}` : SIG[k][0]; }
function scaleOf(k) { return /^w_/.test(k) ? 100 : /^f_\d/.test(k) ? 1 : SIG[k][1]; }
function slotOf(k) { const m = k.match(/^[wf]_(\d+)_(\d+)$/); if (m) return { "2_1": 1, "3_2": 2, "4_3": 3, "5_4": 4, "3_1": 7 }[`${m[1]}_${m[2]}`] || 6; return SIG[k][2]; }

function drawStrips(R, cursorT) {
  const col = { grid: css("--grid"), ink3: css("--ink-3"), ink: css("--ink"), crit: css("--critical"), ghost: css("--ghost"), surf: css("--void"), measured: css("--measured") };
  for (const box of $("strips").children) {
    const d = STRIPS[box.dataset.k], cv = box.lastChild, head = box.firstChild;
    const [g, w, h] = fit(cv);
    const keys = keysOf(d), t = S.t, i0 = Math.max(lower(t, R[0]) - 1, 0), i1 = Math.min(lower(t, R[1]) + 1, t.length);
    let lo = Infinity, hi = -Infinity;
    for (const k of keys) { const v = S.data[k], s = scaleOf(k); for (let i = i0; i < i1; i++) { const y = v[i] * s; if (isFinite(y)) { if (y < lo) lo = y; if (y > hi) hi = y; } } }
    // what really happened, when this is a real shot: grey dots, in the units of the trace
    const real = S.real || {}, mine = [];
    for (const k of keys) {
      const spec = { W_thermal: ["t", "W_thermal"], T_e0: ["t", "T_e0"], T_i0: ["t", "T_i0"], n_line: ["t", "n_line"], beam_neutrons: ["t_sum", "neutrons"], P_rad: ["t_sum", "P_rad"], H: ["t", "H"] }[k];
      if (spec && real[spec[1]]) mine.push([real[spec[0]], real[spec[1]], scaleOf(k)]);
    }
    for (const [rt, rv, sc] of mine) for (let i = 0; i < rv.length; i++) if (rv[i] != null && rt[i] >= R[0] && rt[i] <= R[1]) { const y = rv[i] * sc; if (y < lo) lo = y; if (y > hi) hi = y; }
    const gh = S.ghost ? S.ghost.rows : null;
    if (gh) for (const row of gh) for (const k of keys) if (row[k] != null && row.t >= R[0] && row.t <= R[1]) { const y = row[k] * scaleOf(k); if (y < lo) lo = y; if (y > hi) hi = y; }
    if (lo === Infinity) { lo = 0; hi = 1; }
    if (!d.freeY) lo = Math.min(lo, 0);
    if (hi - lo < 1e-9) hi = lo + (Math.abs(lo) * .1 || 1);
    const pad = (hi - lo) * .08;
    hi += pad; if (d.freeY) lo -= pad;
    const top = 5, bot = h - 3;
    const Y = y => bot - (y - lo) / (hi - lo) * (bot - top);
    g.font = MONO; g.lineWidth = 1; g.strokeStyle = col.grid; g.fillStyle = col.ink3; g.textBaseline = "middle"; g.textAlign = "right";
    for (const y of ticks(lo, hi, d.small ? 2 : 3)) { hline(g, PAD.l, w - PAD.r, Y(y)); g.fillText(Math.abs(y) >= 1000 ? (y / 1000) + "K" : +y.toPrecision(3), PAD.l - 4, Y(y)); }
    for (const x of ticks(R[0], R[1], 6)) vline(g, X(x, R, w), top, bot);
    if (d.dyn === "w_") {          // locks shaded
      g.fillStyle = col.crit; g.globalAlpha = .12;
      for (const lk of Object.keys(S.data).filter(k => k.startsWith("locked_"))) { const v = S.data[lk]; for (let i = i0; i < i1 - 1; i++) if (v[i] > 0) g.fillRect(X(t[i], R, w), top, Math.max(1, X(t[i + 1], R, w) - X(t[i], R, w)), bot - top); }
      g.globalAlpha = 1;
    }
    g.lineJoin = g.lineCap = "round";
    if (gh) {
      g.strokeStyle = col.ghost; g.lineWidth = 1;
      const tg = gh.map(r => r.t);
      for (const k of keys) { const v = gh.map(r => r[k] == null ? NaN : r[k]); pen(g, tg, v, scaleOf(k), R, w, Y, d.step); }
    }
    g.lineWidth = 1.5;
    for (const k of keys) { g.strokeStyle = css(`--s${slotOf(k)}`); pen(g, t, S.data[k], scaleOf(k), R, w, Y, d.step); }
    g.fillStyle = col.measured;
    for (const [rt, rv, sc] of mine) for (let i = 0; i < rv.length; i++) if (rv[i] != null && rt[i] >= R[0] && rt[i] <= R[1]) g.fillRect(X(rt[i], R, w) - 1, Y(rv[i] * sc) - 1, 2, 2);
    // the track's header: the reading, and the key that goes with it. The colour sits on the key, the words stay in ink
    const at = cursorT == null ? t.length - 1 : Math.min(lower(t, cursorT), t.length - 1);
    head.replaceChildren();
    const title = document.createElement("b"); title.textContent = d.title;
    if (d.unit) { const u = document.createElement("small"); u.textContent = d.unit; title.append(u); }
    head.append(title);
    for (const k of keys) {
      const row = document.createElement("div"), s = document.createElement("span"), i = document.createElement("i"), v = document.createElement("em");
      row.className = "ch";
      i.style.background = css(`--s${slotOf(k)}`);
      s.append(i, document.createTextNode(labelOf(k)));
      v.textContent = fmt(S.data[k][at] * scaleOf(k));
      row.append(s, v);
      head.append(row);
    }
    if (mine.length) { const s = document.createElement("div"); s.className = "real"; s.textContent = "white: measured"; head.append(s); }
    if (cursorT != null) {
      const x = X(t[at], R, w);
      g.strokeStyle = col.ink; g.lineWidth = 1; vline(g, x, top - 2, bot);
      for (const k of keys) {
        const y = S.data[k][at] * scaleOf(k);
        if (!isFinite(y)) continue;
        g.beginPath(); g.arc(x, Y(y), 4, 0, 7); g.fillStyle = css(`--s${slotOf(k)}`); g.fill(); g.lineWidth = 2; g.strokeStyle = col.surf; g.stroke();
      }
    }
  }
}
function wire(cv) {
  let startX = null;
  cv.addEventListener("pointermove", e => {
    const r = cv.getBoundingClientRect();
    if (!S.pinned) S.cursor = tOf(e.clientX - r.left, range(), r.width);
    S.hoverY = e.clientY; S.hoverX = e.clientX; S.hoverCanvas = cv;
    S.dirty = true;
    tip(e);
  });
  cv.addEventListener("pointerleave", () => { if (!S.pinned) S.cursor = null; $("tooltip").hidden = true; S.dirty = true; });
  cv.addEventListener("pointerdown", e => { startX = e.clientX; });
  cv.addEventListener("pointerup", e => {
    const r = cv.getBoundingClientRect();
    if (startX != null && Math.abs(e.clientX - startX) > 8) {
      const R = range(), a = tOf(Math.min(startX, e.clientX) - r.left, R, r.width), b = tOf(Math.max(startX, e.clientX) - r.left, R, r.width);
      S.zoom = [a, b];
      $("pin-clear").hidden = false; $("pin-clear").textContent = "Follow live";
    } else {
      S.pinned = true;
      S.cursor = tOf(e.clientX - r.left, range(), r.width);
      $("pin-clear").hidden = false;
      fetchMeasured();
    }
    startX = null;
    S.dirty = true;
  });
  cv.addEventListener("dblclick", () => { S.zoom = null; S.pinned = false; S.cursor = null; $("pin-clear").hidden = true; S.dirty = true; });
}
function tip(e) {
  const box = e.target.parentElement, d = box && box.classList && box.classList.contains("strip") ? STRIPS[box.dataset.k] : null, el = $("tooltip");
  if (!d || !S.t.length) { el.hidden = true; return; }
  const r = e.target.getBoundingClientRect(), t = tOf(e.clientX - r.left, range(), r.width), i = Math.min(lower(S.t, t), S.t.length - 1);
  el.replaceChildren();
  const h = document.createElement("div"); h.className = "t"; h.textContent = `${fmt(S.t[i], 5)} s`; el.append(h);
  for (const k of keysOf(d)) {
    const row = document.createElement("div"), key = document.createElement("i"), val = document.createElement("b");
    key.style.background = css(`--s${slotOf(k)}`);
    val.textContent = fmt(S.data[k][i] * scaleOf(k), 4);
    row.append(key, val, document.createTextNode(` ${labelOf(k)}`));
    el.append(row);
  }
  el.hidden = false;
  el.style.left = Math.min(e.clientX + 14, innerWidth - 240) + "px";
  el.style.top = (e.clientY + 12) + "px";
}

// the events above the chart
function drawRug(R, cursorT) {
  const [g, w, h] = fit($("rug"));
  const items = [], t = S.t;
  const i0 = Math.max(lower(t, R[0]), 0), i1 = lower(t, R[1]) + 1;
  g.lineWidth = 1;
  const marks = (key, color, top) => { const v = S.data[key]; if (!v) return; g.strokeStyle = css(color); for (let i = i0; i < Math.min(i1, t.length); i++) if (v[i] > 0) vline(g, X(t[i], R, w), top, h - 1); };
  marks("elm", "--s4", h - 7);
  marks("sawtooth", "--s7", h - 12);
  g.font = MONO; g.textBaseline = "top"; g.textAlign = "left"; g.fillStyle = css("--ink");
  let lastX = -1e9;
  const st = S.st;
  for (const c of (st.commands || [])) {
    if (c.t < R[0] || c.t > R[1]) continue;
    const x = X(c.t, R, w);
    g.strokeStyle = css("--ink"); vline(g, x, 1, h - 1);
    if (x - lastX > 34) { g.fillText(c.command.cmd, x + 3, 1); lastX = x + 3 + g.measureText(c.command.cmd).width; }
    items.push([x, `${fmt(c.t, 5)} s: ${JSON.stringify(c.command)}`]);
  }
  for (const c of (st.scheduled || [])) { if (c.t >= R[0] && c.t <= R[1]) { const x = X(c.t, R, w); g.strokeStyle = css("--ink-3"); g.setLineDash([2, 2]); vline(g, x, 1, h - 1); g.setLineDash([]); items.push([x, `scheduled ${fmt(c.t, 5)} s: ${c.cmd}`]); } }
  const dis = S.data.disrupted;
  if (dis) for (let i = i0; i < Math.min(i1, t.length); i++) if (dis[i] > 0) { const x = X(t[i], R, w); g.fillStyle = css("--critical"); g.beginPath(); g.moveTo(x, 12); g.lineTo(x + 5, 18); g.lineTo(x, 24); g.lineTo(x - 5, 18); g.fill(); items.push([x, `${fmt(t[i], 5)} s: the shot ended`]); }
  $("rug").onpointermove = e => { const r = $("rug").getBoundingClientRect(), x = e.clientX - r.left; let best = null, bd = 6; for (const it of items) { const d = Math.abs(it[0] - x); if (d < bd) { bd = d; best = it; } } const el = $("tooltip"); if (!best) { el.hidden = true; return; } el.replaceChildren(document.createTextNode(best[1])); el.hidden = false; el.style.left = Math.min(e.clientX + 14, innerWidth - 300) + "px"; el.style.top = e.clientY + 12 + "px"; };
  $("rug").onpointerleave = () => ($("tooltip").hidden = true);
}
function drawAxis(R) {
  const [g, w, h] = fit($("axis"));
  g.font = MONO; g.fillStyle = css("--ink-2"); g.textAlign = "center"; g.textBaseline = "top"; g.strokeStyle = css("--ink"); g.lineWidth = 1;
  hline(g, PAD.l, w - PAD.r, 0);
  for (const x of ticks(R[0], R[1], 6)) { const px = X(x, R, w); vline(g, px, 0, 4); g.fillText(+x.toPrecision(6), px, 6); }
  g.textAlign = "right"; g.fillText("s", PAD.l - 6, 6);
}

// ---------- heat map and colours
const RAMP = ["#0d0604", "#2c0e08", "#5e170c", "#9a2b0f", "#d24b14", "#f2801f", "#f9b45a", "#fde3b0", "#fff8ea"];
const hex = c => [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)];
function rampAt(u) {
  u = Math.max(0, Math.min(1, u)) * (RAMP.length - 1);
  const i = Math.min(Math.floor(u), RAMP.length - 2), f = u - i, a = hex(RAMP[i]), b = hex(RAMP[i + 1]);
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
function fieldOf(f, k) { return f[k] || null; }
function drawHeat(R) {
  const cv = $("heat"), [g, w, h] = fit(cv), F = S.frames, top = 2, bot = h - 2;
  const k = S.heat;
  if (!F.length) return;
  const i0 = Math.max(lower(F.map(f => f.t), R[0]) - 1, 0), i1 = Math.min(lower(F.map(f => f.t), R[1]) + 1, F.length);
  let lo = Infinity, hi = -Infinity;
  for (let i = i0; i < i1; i++) { const v = fieldOf(F[i], k); if (v) for (const y of v) if (isFinite(y)) { if (y < lo) lo = y; if (y > hi) hi = y; } }
  if (lo === Infinity) return;
  if (k === "q") hi = Math.min(hi, 8);
  const pw = Math.max(1, Math.round(w - PAD.l - PAD.r));
  const off = heatBuf(pw, 41), o = off.getContext("2d"), img = o.createImageData(pw, 41);
  for (let x = 0; x < pw; x++) {
    const f = frameAt(R[0] + (x + .5) / pw * (R[1] - R[0])), v = f && fieldOf(f, k);
    const inside = f && f.t >= S.frames[0].t && f.t <= S.frames[F.length - 1].t + 1e-6;
    for (let r = 0; r < 41; r++) {
      const j = (x + (40 - r) * pw) * 4;
      if (!v || !inside) { img.data[j + 3] = 0; continue; }
      const [a, b, c] = rampAt((v[r] - lo) / (hi - lo || 1));
      img.data[j] = a; img.data[j + 1] = b; img.data[j + 2] = c; img.data[j + 3] = 255;
    }
  }
  o.putImageData(img, 0, 0);
  g.imageSmoothingEnabled = true;
  g.drawImage(off, PAD.l, top, pw, bot - top);
  const Y = rho => bot - rho * (bot - top);
  g.font = MONO; g.fillStyle = css("--ink-3"); g.textAlign = "right"; g.textBaseline = "middle";
  for (const y of [0, .5, 1]) g.fillText(y, PAD.l - 4, Y(y));
  // islands on top of it
  const a = S.meta && S.meta.geometry ? S.meta.geometry.a : 2;
  g.strokeStyle = "#ffffff"; g.lineWidth = 1;
  const tracks = {};
  for (let i = i0; i < i1; i += Math.max(1, Math.floor((i1 - i0) / pw))) for (const isl of F[i].islands || []) { if (isl.rho == null) continue; (tracks[`${isl.m}/${isl.n}`] ||= []).push([F[i].t, isl.rho, isl.w]); }
  for (const pts of Object.values(tracks)) {
    g.setLineDash([3, 3]);
    g.beginPath(); pts.forEach(([t, r], i) => { const x = X(t, R, w); i ? g.lineTo(x, Y(r)) : g.moveTo(x, Y(r)); }); g.stroke();
    g.setLineDash([]);
    for (const [t, r, wd] of pts) if (wd > 0) { const x = X(t, R, w), half = wd / (2 * a); g.strokeStyle = "#ffffff"; vline(g, x, Y(r + half), Y(r - half)); }
  }
  $("heat-note").textContent = `${fmt(lo)} to ${fmt(hi)} ${HEAT[k][1]}, bright is high. White: rational surfaces and islands`;
  if (S.cursor != null) { g.strokeStyle = css("--ink"); vline(g, X(S.cursor, R, w), top, bot); }
}
let _hb = null;
function heatBuf(w, h) { if (!_hb) _hb = document.createElement("canvas"); if (_hb.width !== w || _hb.height !== h) { _hb.width = w; _hb.height = h; } return _hb; }

// ---------- profiles
const PROFILES = [
  { id: "T", title: "Temperature (keV)", series: [["T_e", "T_e", 1], ["T_i", "T_i", 2]], measured: [["T_e", 1], ["T_i", 2]] },
  { id: "n", title: "Density (1e20 m^-3)", series: [["n_e", "n_e", 1]], measured: [["n_e", 1]] },
  { id: "q", title: "Safety factor", series: [["q", "q", 1]], rules: [1, 1.5, 2, 3], max: 7 },
  { id: "x", title: "", select: true },
];
const EXTRA = { chi: ["Transport (m^2/s)", [["chi_e", "chi_e", 1], ["chi_i", "chi_i", 2]]], Z_eff: ["Z_eff", [["Z_eff", "Z_eff", 7]]], j: ["Current density (MA/m^2)", [["j", "j", 1]]], rad: ["Line radiation (MW/m^3)", [["rad", "rad", 3]]], imp: ["Injected impurity (1e19 m^-3)", null] };
function buildProfiles() {
  $("profiles").replaceChildren(...PROFILES.map(p => {
    const box = document.createElement("div"), h = document.createElement("h4"), cv = document.createElement("canvas");
    box.className = "profile"; box.dataset.id = p.id;
    if (p.select) {
      const sel = document.createElement("select");
      sel.append(...Object.entries(EXTRA).map(([k, v]) => { const o = document.createElement("option"); o.value = k; o.textContent = v[0]; return o; }));
      sel.onchange = () => { S.pf4 = sel.value; S.dirty = true; };
      sel.setAttribute("aria-label", "What the fourth profile shows");
      box.append(sel);
    }
    box.append(h, cv);
    cv.addEventListener("pointermove", e => profTip(e, box));
    cv.addEventListener("pointerleave", () => ($("tooltip").hidden = true));
    return box;
  }));
}
function profSeries(p, f) {
  if (!p.select) return p.series.filter(s => f[s[0]]).map(s => [s[1], f[s[0]], s[2]]);
  if (S.pf4 === "imp") return Object.entries(f.imp || {}).map(([k, v], i) => [k, v, [1, 2, 3, 4, 7, 8][i % 6]]);
  return EXTRA[S.pf4][1].filter(s => f[s[0]]).map(s => [s[1], f[s[0]], s[2]]);
}
function drawProfiles(t) {
  const f = frameAt(t), knots = S.meta ? S.meta.knots : [];
  if (!f) return;
  for (const box of $("profiles").children) {
    const p = PROFILES.find(x => x.id === box.dataset.id), cv = box.querySelector("canvas"), h = box.querySelector("h4");
    const [g, w, H] = fit(cv), l = 30, r = 6, top = 6, bot = H - 16;
    const series = profSeries(p, f);
    let hi = -Infinity, lo = 0;
    for (const s of series) for (const y of s[1]) if (isFinite(y)) { hi = Math.max(hi, y); lo = Math.min(lo, y); }
    const pts = p.measured && S.measured ? p.measured.flatMap(([k, slot]) => (S.measured[k] || []).map(q => [q[0], q[1], slot])) : [];
    for (const q of pts) if (isFinite(q[1])) hi = Math.max(hi, Math.min(q[1], hi * 2.5));
    if (p.max) hi = Math.min(hi, p.max);
    if (!(hi > lo)) hi = lo + 1;
    hi *= 1.05;
    const Xr = x => l + x * (w - l - r), Y = y => bot - (y - lo) / (hi - lo) * (bot - top);
    g.font = MONO; g.strokeStyle = css("--grid"); g.fillStyle = css("--ink-3"); g.lineWidth = 1; g.textAlign = "right"; g.textBaseline = "middle";
    for (const y of ticks(lo, hi, 3)) { hline(g, l, w - r, Y(y)); g.fillText(+y.toPrecision(3), l - 3, Y(y)); }
    g.textAlign = "center"; g.textBaseline = "top";
    for (const x of [0, .5, 1]) { vline(g, Xr(x), top, bot); g.fillText(x, Xr(x), bot + 3); }
    if (p.rules) { g.strokeStyle = css("--ink-3"); g.setLineDash([2, 3]); for (const y of p.rules) if (y < hi) hline(g, l, w - r, Y(y)); g.setLineDash([]); }
    for (const q of pts) if (isFinite(q[1]) && q[1] < hi) { g.fillStyle = css("--measured"); g.fillRect(Xr(Math.min(q[0], 1)) - 1, Y(q[1]) - 1, 2, 2); }
    g.lineJoin = g.lineCap = "round";
    for (const s of series) {
      g.strokeStyle = css(`--s${s[2]}`); g.lineWidth = 2; g.beginPath();
      s[1].forEach((y, i) => { const px = Xr(knots[i]), py = Y(Math.min(y, hi)); i ? g.lineTo(px, py) : g.moveTo(px, py); });
      g.stroke();
    }
    for (const isl of f.islands || []) if (isl.w > 0 && isl.rho != null && (S.meta.geometry)) { const half = isl.w / (2 * S.meta.geometry.a); g.fillStyle = css("--critical"); g.globalAlpha = .13; g.fillRect(Xr(isl.rho - half), top, Xr(isl.rho + half) - Xr(isl.rho - half), bot - top); g.globalAlpha = 1; }
    h.replaceChildren(document.createTextNode(p.select ? "" : p.title));
    for (const s of series) { const sp = document.createElement("span"), i = document.createElement("i"); i.style.background = css(`--s${s[2]}`); sp.append(i, document.createTextNode(s[0])); h.append(sp); }
    if (pts.length) { const sp = document.createElement("span"); sp.textContent = "white: measured"; h.append(sp); }
  }
}
function profTip(e, box) {
  const p = PROFILES.find(x => x.id === box.dataset.id), f = frameAt(S.pinned ? S.cursor : S.cursor), el = $("tooltip"), cv = e.target;
  if (!f) return;
  const r = cv.getBoundingClientRect(), rho = Math.max(0, Math.min(1, (e.clientX - r.left - 30) / (r.width - 36))), i = Math.round(rho * 40);
  el.replaceChildren();
  const h = document.createElement("div"); h.className = "t"; h.textContent = `rho ${rho.toFixed(2)}`; el.append(h);
  for (const s of profSeries(p, f)) { const row = document.createElement("div"), k = document.createElement("i"), b = document.createElement("b"); k.style.background = css(`--s${s[2]}`); b.textContent = fmt(s[1][i], 4); row.append(k, b, document.createTextNode(` ${s[0]}`)); el.append(row); }
  el.hidden = false; el.style.left = Math.min(e.clientX + 14, innerWidth - 200) + "px"; el.style.top = e.clientY + 12 + "px";
}
async function fetchMeasured() {
  if (!S.meta || !S.meta.measured) return;
  const f = frameAt(S.cursor);
  if (!f) return;
  S.measured = await api(`/api/measured?t=${f.t}`);
  S.dirty = true;
}

// ---------- the cross-section
function drawXsec(t, now) {
  const f = frameAt(t), M = S.meta && S.meta.machine, G = (f && f.shape) || (S.meta && S.meta.geometry);
  const cv = $("xsec"), [g, w, h] = fit(cv);
  if (!G || !f) return;
  const shape = f.shape, measured = shape && shape.measured_lcfs && shape.measured_lcfs.length;
  $("xs-note").textContent = (shape ? "FreeGSNKE's boundary dashed" + (measured ? ", EFIT's measured boundary in blue" : "") + ", TORAX's surfaces inside" : "TORAX's flux surfaces") + (M ? "; the machine's own wall and coils" : "; vessel not to scale");
  const k = S.heat, v = f[k] || f.T_e, last = G.rho.length - 1, z0 = (shape && shape.Z_axis) || 0;
  const Rc = i => (G.R_in[i] + G.R_out[i]) / 2, rad = i => (G.R_out[i] - G.R_in[i]) / 2;
  const edgeK = G.kappa[last], edgeD = G.delta[last];
  const box = M ? M.box : [Rc(last) - G.a * 1.34, Rc(last) + G.a * 1.34, -G.a * edgeK * 1.3, G.a * edgeK * 1.3];
  const sc = Math.min((w - 14) / (box[1] - box[0]), (h - 14) / (box[3] - box[2])), cx = w / 2 - (box[0] + box[1]) / 2 * sc, cy = h / 2 + (box[2] + box[3]) / 2 * sc;
  const P = (R, Z) => [cx + R * sc, cy - Z * sc];
  const point = (i, th, extra = 0) => { const r = rad(i) + extra; return P(Rc(i) + r * Math.cos(th + G.delta[i] * Math.sin(th)), z0 + G.kappa[i] * r * Math.sin(th)); };
  const outline = (pts, close = true) => { g.beginPath(); pts.forEach(([R, Z], j) => { const [x, y] = P(R, Z); j ? g.lineTo(x, y) : g.moveTo(x, y); }); if (close) g.closePath(); };
  // the machine
  if (M) {
    g.fillStyle = "#16140f"; for (const [R, Z, dR, dZ] of M.passive || []) { const [x, y] = P(R - dR / 2, Z + dZ / 2); g.fillRect(x, y, Math.max(dR * sc, 1), Math.max(dZ * sc, 1)); }
    g.strokeStyle = css("--rule-2"); g.lineWidth = 1; if (M.wall) { outline(M.wall); g.stroke(); }
    const now_I = (shape && shape.coils_A) || {}, Imax = Math.max(1, ...Object.values(now_I).map(Math.abs));
    for (const c of M.coils || []) {
      const [x, y] = P(c.R - c.dR / 2, c.Z + c.dZ / 2), I = now_I[c.name], cw = Math.max(c.dR * sc, 2), ch = Math.max(c.dZ * sc, 2);
      g.fillStyle = "#1d1b18";
      if (I != null) { const a = Math.min(1, Math.abs(I) / Imax); g.fillStyle = I >= 0 ? `rgba(57,135,229,${.18 + .72 * a})` : `rgba(217,89,38,${.18 + .72 * a})`; }
      g.fillRect(x, y, cw, ch);
    }
  } else {
    const wallR = G.a * 1.2, wallZ = G.a * edgeK * 1.2;
    g.strokeStyle = css("--rule-2"); g.lineWidth = 1; g.setLineDash([3, 4]); g.beginPath();
    for (let i = 0; i <= 72; i++) { const th = i / 72 * 2 * Math.PI, [x, y] = P(Rc(last) + wallR * Math.cos(th + edgeD * Math.sin(th)), wallZ * Math.sin(th)); i ? g.lineTo(x, y) : g.moveTo(x, y); }
    g.closePath(); g.stroke(); g.setLineDash([]);
  }
  // the plasma, as filled surfaces from the outside in
  let lo = Infinity, hi = -Infinity;
  for (const y of v) if (isFinite(y)) { lo = Math.min(lo, y); hi = Math.max(hi, y); }
  const surface = (i, extra = 0) => { g.beginPath(); for (let j = 0; j <= 64; j++) { const [x, y] = point(i, j / 64 * 2 * Math.PI, extra); j ? g.lineTo(x, y) : g.moveTo(x, y); } g.closePath(); };
  for (let i = last; i >= 1; i--) {
    const [a, b, c] = rampAt((v[i] - lo) / (hi - lo || 1));
    g.fillStyle = `rgb(${a | 0},${b | 0},${c | 0})`;
    surface(i); g.fill();
  }
  // the light you would see through a port comes from the edge, in D-alpha. An ELM is a flash of it; a sawtooth crash lights the core
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const age = e => still ? 1e9 : now - e, elm = Math.exp(-age(S.flash.elm) / 220), saw = Math.exp(-age(S.flash.saw) / 160);
  g.save();
  g.shadowColor = css("--dalpha"); g.shadowBlur = 10 + 34 * elm; g.strokeStyle = `rgba(255,70,120,${.42 + .58 * elm})`; g.lineWidth = 1 + 2 * elm;
  surface(last); g.stroke();
  g.restore();
  if (saw > .02) { const q1 = Math.max(2, Math.round(last * .38)); g.save(); g.globalAlpha = .75 * saw; g.fillStyle = "#fffdf5"; surface(q1); g.fill(); g.restore(); }
  const [ax, ay] = point(0, 0);
  g.strokeStyle = "rgba(0,0,0,.55)"; g.lineWidth = 1; g.beginPath(); g.moveTo(ax - 5, ay); g.lineTo(ax + 5, ay); g.moveTo(ax, ay - 5); g.lineTo(ax, ay + 5); g.stroke();
  // the free-boundary solver's own last closed surface, and what was measured
  if (shape && shape.lcfs) { g.save(); g.setLineDash([4, 3]); g.strokeStyle = "#f4efe3"; g.lineWidth = 1; outline(shape.lcfs); g.stroke(); g.restore(); }
  if (measured) { g.save(); g.strokeStyle = css("--efit"); g.lineWidth = 1.25; outline(shape.measured_lcfs); g.stroke(); g.restore(); }
  // islands: a chain of m lobes on the surface, to scale in width; rotation slowed so it can be seen
  (f.islands || []).forEach(isl => {
    if (!(isl.w > 0) || isl.rho == null) return;
    const q = Math.max(1, Math.min(last, Math.round(isl.rho * last))), half = isl.w / 2, spin = (isl.f && Math.abs(isl.f) > 1 && !still) ? now * 3e-4 * Math.sign(isl.f) : 0;
    const th0 = (isl.phi || 0) / isl.m + spin, locked = isl.f != null && Math.abs(isl.f) < 1;
    g.lineWidth = 1; g.strokeStyle = locked ? css("--critical") : "#f4efe3"; g.fillStyle = "#050505"; g.globalAlpha = .62;
    g.beginPath();
    const steps = 180;
    for (let j = 0; j <= steps; j++) { const th = j / steps * 2 * Math.PI, d = half * Math.abs(Math.cos(isl.m * (th - th0) / 2)); const [x, y] = point(q, th, d); j ? g.lineTo(x, y) : g.moveTo(x, y); }
    for (let j = steps; j >= 0; j--) { const th = j / steps * 2 * Math.PI, d = half * Math.abs(Math.cos(isl.m * (th - th0) / 2)); const [x, y] = point(q, th, -d); g.lineTo(x, y); }
    g.closePath(); g.fill("evenodd"); g.globalAlpha = 1; g.stroke();
  });
  // the gyrotron beam, from an upper launcher to where it lands
  const gy = f.gyro;
  if (gy && gy.P > 0 && gy.aim != null) {
    const q = Math.max(1, Math.min(last, Math.round(gy.aim * last))), [x1, y1] = point(q, 1.0), [x0, y0] = P(Rc(last) + (G.R_out[last] - Rc(last)) * 1.1, z0 + G.a * edgeK * 1.12);
    g.strokeStyle = css("--warn"); g.lineWidth = .75 + 2 * Math.min(1, gy.P / (gy.units * gy.unit_MW || 1)); g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
    g.fillStyle = css("--warn"); g.fillRect(x1 - 2, y1 - 2, 4, 4);
  }
  // shards, along the outboard midplane
  g.fillStyle = "#fffdf5";
  for (const s of f.shards || []) s.x.forEach((d, i) => { if (s.r[i] > 0) { const [x, y] = P(G.R_out[last] - d, z0 + (i - 3) * .04 * G.a); g.fillRect(x - 1.5, y - 1.5, 3, 3); } });
  // gas, at the edge
  const gas = S.data.gas_rate, seed = S.data.seed_rate;
  if ((gas && gas[gas.length - 1] > 0) || (seed && seed[seed.length - 1] > 0)) { const [x, y] = P(G.R_in[last] - G.a * .12, z0); g.fillStyle = css("--s3"); for (let j = 0; j < 3; j++) g.fillRect(x + j * 5, y + (j % 2) * 5 - 2, 2, 2); }
  $("port-read").textContent = `R ${fmt(G.R0, 3)} m  a ${fmt(G.a, 3)} m\nkappa ${fmt(edgeK, 3)}  delta ${fmt(edgeD, 2)}`;
  $("xs-ramp").style.background = `linear-gradient(90deg, ${[0, .125, .25, .375, .5, .625, .75, .875, 1].map(u => { const [a2, b2, c2] = rampAt(u); return `rgb(${a2 | 0},${b2 | 0},${c2 | 0})`; }).join(",")})`;
  $("xs-scale").textContent = `${HEAT[S.heat] ? HEAT[S.heat][0] : "T_e"} ${fmt(lo, 3)} to ${fmt(hi, 3)} ${HEAT[S.heat] ? HEAT[S.heat][1] : "keV"}`;
}

// ---------- the backplane: every model plugged into TORAX's state
const PLANE = {
  W: 132, x: i => 9.5 + i * 141.5, T0: 16, TH: 52, BUS: 106, BH: 30, B0: 174,
  // id, name, what it reads off the state, what it writes back
  top: [["programme", "Programme", null, "Ip, n, P"], ["beams", "NBI beams", "n, T", "heat, fuel"], ["fuel", "Gas, pellets", "n line", "S_n"], ["impurities", "Impurities", "n, T", "Z_eff, rad"], ["free", "FreeGSNKE", "p', FF'", "surfaces"], ["eped", "EPED-NN", "beta, n", "p_ped, w"], ["gyro", "Gyrotrons", null, "ECCD"]],
  bot: [["lh", "L-H", "P_SOL", "state"], ["elms", "ELMs", "p_top", "crash"], ["saw", "Sawteeth", "q, shear", "mix"], ["limit", "Beta limit", "beta, li", "collapse"], ["islands", "Islands", "q, beta", "flatten"], ["sensors", "Sensors", "w, rho", null], ["decision", "Decision", null, null]],
  links: [["beams", "free", "fast-ion pressure", "top"], ["saw", "islands", "seed", "bot1"], ["elms", "islands", null, "bot1"], ["limit", "lh", "back to L", "bot2"], ["free", "lh", null, "none"]],
};
function buildPlane() {
  const svg = $("plane"), NS = "http://www.w3.org/2000/svg", P = PLANE;
  svg.setAttribute("viewBox", "0 0 1000 250");
  svg.replaceChildren();
  const el = (tag, attrs, parent = svg) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.append(e); return e; };
  const label = (x, y, text, cls, parent) => { const e = el("text", { x, y, class: cls }, parent); e.textContent = text; return e; };
  const wires = {}, tips = {}, tiles = {};
  const stub = (id, x, y0, y1, text) => {
    (wires[id] ||= []).push(el("line", { x1: x, x2: x, y1: y0, y2: y1, class: "wire" }));
    const up = y1 < y0, hw = 3.2, tipY = y1;
    (tips[id] ||= []).push(el("polygon", { points: `${x - hw},${tipY + (up ? 5 : -5)} ${x + hw},${tipY + (up ? 5 : -5)} ${x},${tipY}`, class: "tip" }));
    if (text) label(x + 5, (y0 + y1) / 2 + 3, text, "what");
  };
  const wiring = (row, tileY, top) => row.forEach(([id, , reads, writes], i) => {
    const x = P.x(i), near = top ? tileY + P.TH : tileY, far = top ? P.BUS : P.BUS + P.BH;
    if (reads) stub(id, x + 22, far, near, reads);
    if (writes) stub(id, x + 74, near, far, writes);
  });
  wiring(P.top, P.T0, true);
  wiring(P.bot, P.B0, false);
  // the bus itself
  el("rect", { x: 9.5, y: P.BUS, width: 981, height: P.BH, class: "bus" });
  label(20, P.BUS + 20, "TORAX", "bus-name");
  const busText = label(84, P.BUS + 19, "", "bus-text");
  // the links that do not go through the state
  const route = (id, d, tipAt, tipDir, text, tx, ty) => {
    (wires[id] ||= []).push(el("path", { d, class: "wire" }));
    const [x, y] = tipAt, hw = 3.2, pts = { up: `${x - hw},${y + 5} ${x + hw},${y + 5} ${x},${y}`, down: `${x - hw},${y - 5} ${x + hw},${y - 5} ${x},${y}`, right: `${x - 5},${y - hw} ${x - 5},${y + hw} ${x},${y}` }[tipDir];
    (tips[id] ||= []).push(el("polygon", { points: pts, class: "tip" }));
    if (text) label(tx, ty, text, "what");
  };
  const cx = i => P.x(i) + P.W / 2, bb = P.B0 + P.TH;
  route("free", `M${cx(1)} ${P.T0} V7 H${cx(4)} V${P.T0}`, [cx(4), P.T0], "down", "fast-ion pressure", cx(2) + 22, 5);
  route("islands", `M${cx(1)} ${bb} V${bb + 9} H${cx(4) - 10} V${bb}`, [cx(4) - 10, bb], "up", "ELMs and crashes seed islands", cx(2) - 4, bb + 7);
  route("islands", `M${cx(2)} ${bb} V${bb + 9}`, [cx(2), bb + 9], "down", null);
  route("limit", `M${cx(3)} ${bb} V${bb + 20} H${cx(0)} V${bb}`, [cx(0), bb], "up", "a collapse ends H-mode", cx(0) + 10, bb + 18);
  route("lh", `M${P.x(0) + P.W} ${P.B0 + P.TH / 2} H${P.x(1)}`, [P.x(1), P.B0 + P.TH / 2], "right", null);
  route("sensors", `M${P.x(5) + P.W} ${P.B0 + P.TH / 2} H${P.x(6)}`, [P.x(6), P.B0 + P.TH / 2], "right", null);
  route("decision", `M${P.x(6) + 100} ${P.B0} V${P.T0 + P.TH}`, [P.x(6) + 100, P.T0 + P.TH], "up", "P, aim", P.x(6) + 104, P.B0 - 14);
  // the tiles
  const tile = ([id, name], i, y) => {
    const g = el("g", { class: "mod" }), x = P.x(i);
    const t = el("title", {}, g); t.textContent = name;
    el("rect", { x, y, width: P.W, height: P.TH, class: "tile" }, g);
    el("rect", { x: x + P.W - 11, y: y + 6, width: 6, height: 6, class: "lamp" }, g);
    label(x + 7, y + 15, name, "name", g);
    tiles[id] = { g, a: label(x + 7, y + 30, "", "read", g), b: label(x + 7, y + 43, "", "read", g) };
  };
  P.top.forEach((row, i) => tile(row, i, P.T0));
  P.bot.forEach((row, i) => tile(row, i, P.B0));
  S.plane = { wires, tips, tiles, busText };
}
function paintPlane(q) {
  if (!S.plane) buildPlane();
  const c = q.coupling || {}, now = performance.now(), { wires, tips, tiles, busText } = S.plane;
  const f = 22;
  for (const id in tiles) {
    const m = c[id] || {}, t = tiles[id];
    t.g.setAttribute("class", "mod" + (m.on ? " on" : "") + (m.flag ? ` ${m.flag}` : ""));
    t.a.textContent = (m.a || "").slice(0, f);
    t.b.textContent = (m.b || "").slice(0, f);
    if (m.n != null && m.n !== S.counts[id]) { if (S.counts[id] != null) S.hits[id] = now; S.counts[id] = m.n; }
    const hit = now - (S.hits[id] ?? -1e9) < 650;
    for (const w of wires[id] || []) w.setAttribute("class", "wire" + (m.on ? " on" : "") + (hit ? " hit" : ""));
    for (const w of tips[id] || []) w.setAttribute("class", "tip" + (m.on ? " on" : "") + (hit ? " hit" : ""));
  }
  busText.textContent = c.state || "";
}

// ---------- the console
function buildConsole() {
  const spec = S.meta.spec, groups = { Instabilities: "instability", Actuators: "actuator", Faults: "fault", Loops: "loop" };
  const tabs = [...Object.keys(groups), "TORAX", "Log"];
  $("tabs").replaceChildren(...tabs.map(t => { const b = document.createElement("button"); b.textContent = t; b.setAttribute("role", "tab"); b.setAttribute("aria-selected", t === S.tab); b.onclick = () => { S.tab = t; buildConsole(); }; return b; }));
  const box = $("console");
  box.replaceChildren();
  if (groups[S.tab]) for (const [name, e] of Object.entries(spec)) { if (e.kind === groups[S.tab]) box.append(entry(name, e)); }
  else if (S.tab === "TORAX") box.append(toraxPanel());
  else box.append(logPanel());
}
const LEVEL_SHORT = { coupled: "tool", torax: "torax", reduced: "reduced", prescribed: "what-if", margin: "0D" };
function entry(name, e) {
  const el = document.createElement("details"); el.className = "entry";
  const head = document.createElement("summary"), h = document.createElement("h3"), lv = document.createElement("span");
  h.textContent = e.title;
  lv.className = "level"; lv.dataset.l = e.level; lv.textContent = LEVEL_SHORT[e.level]; lv.title = `${LEVELS[e.level][0]}: ${LEVELS[e.level][1]}`;
  const fold = document.createElement("div"); fold.className = "fold";
  const doc = document.createElement("p"); doc.textContent = e.doc;
  const params = document.createElement("div"); params.className = "params";
  const inputs = {};
  for (const [k, p] of Object.entries(e.params || {})) {
    const lab = document.createElement("label");
    lab.title = p.doc || "";
    lab.append(document.createTextNode(k + (p.unit ? ` (${p.unit})` : "")));
    let inp;
    if (p.options) { inp = document.createElement("select"); inp.append(...p.options.map(o => { const op = document.createElement("option"); op.value = op.textContent = o; return op; })); inp.value = p.default; }
    else { inp = document.createElement("input"); inp.type = "number"; inp.step = "any"; inp.value = p.default; if (p.min !== undefined) inp.min = p.min; if (p.max !== undefined) inp.max = p.max; }
    inputs[k] = [inp, !p.options];
    lab.append(inp);
    params.append(lab);
  }
  const go = document.createElement("div"); go.className = "go";
  const fire = document.createElement("button"); fire.className = "fire"; fire.textContent = "Fire";
  const at = document.createElement("input"); at.type = "number"; at.step = "any"; at.placeholder = "now"; at.setAttribute("aria-label", "Time to fire");
  const note = document.createElement("small"); note.textContent = "at t =";
  fire.onclick = ev => {
    ev.preventDefault(); ev.stopPropagation();
    const c = { cmd: name };
    for (const [k, [inp, num]] of Object.entries(inputs)) c[k] = num ? parseFloat(inp.value) : inp.value;
    if (e.rho_pair) { c.rho = [c.rho0, c.rho1]; delete c.rho0; delete c.rho1; }
    if (at.value !== "") c.t = parseFloat(at.value);
    send(c);
  };
  go.append(note, at);
  head.append(h, lv, fire);
  fold.append(doc, params, go);
  el.append(head, fold);
  return el;
}
function toraxPanel() {
  const box = document.createElement("div");
  const p = document.createElement("p"); p.textContent = "Any of TORAX's runtime parameters, live: a dotted path and a number. The compiled step is not rebuilt."; box.append(p);
  const list = document.createElement("ul"); list.className = "log"; list.id = "params-list"; box.append(list);
  const row = document.createElement("div"); row.className = "go";
  const path = document.createElement("input"); path.placeholder = "sources.gas_puff.S_total"; path.style.flex = "1"; path.setAttribute("aria-label", "Parameter path");
  const val = document.createElement("input"); val.type = "number"; val.step = "any"; val.setAttribute("aria-label", "Value");
  const b = document.createElement("button"); b.textContent = "Set"; b.className = "fire";
  b.onclick = () => send({ cmd: "set", path: path.value, value: parseFloat(val.value) });
  row.append(path, val, b); box.append(row);
  const paint = () => { list.replaceChildren(...Object.entries(S.st.params || {}).map(([k, v]) => { const li = document.createElement("li"); li.textContent = `${k} = ${fmt(v, 4)}`; li.style.cursor = "pointer"; li.onclick = () => { path.value = k; val.value = v; }; return li; })); };
  paint();
  return box;
}
function logPanel() {
  const box = document.createElement("div");
  const row = document.createElement("div"); row.className = "go";
  const name = document.createElement("input"); name.placeholder = "shot name"; name.style.flex = "1"; name.setAttribute("aria-label", "Name to save the shot under");
  const save = document.createElement("button"); save.textContent = "Save shot"; save.className = "fire";
  save.onclick = async () => { const a = await api("/api/save", { name: name.value || `shot_${Date.now()}` }); toast(a.path ? `Saved ${a.path}` : a.error); savedList(); };
  row.append(name, save); box.append(row);
  const dl = document.createElement("div"); dl.className = "go";
  for (const [text, href] of [["Traces, CSV", "/api/csv"], ["Scenario and commands, JSON", "/api/export"]]) { const a = document.createElement("a"); a.textContent = text; a.href = href; a.download = ""; a.style.fontSize = "11.5px"; a.style.color = "var(--ink)"; dl.append(a); }
  box.append(dl);
  const hs = document.createElement("h3"); hs.textContent = "Saved shots: overlay in grey"; box.append(hs);
  const saved = document.createElement("ul"); saved.className = "log"; saved.id = "saved"; box.append(saved);
  const savedList = async () => { const names = await api("/api/saved"); saved.replaceChildren(...names.map(n => { const li = document.createElement("li"), b = document.createElement("button"); b.textContent = n; b.className = "ghost"; b.onclick = async () => { const shot = await api(`/api/shot?name=${encodeURIComponent(n)}`); S.ghost = { rows: shot.rows.filter((_, i) => i % Math.max(1, Math.floor(shot.rows.length / 1500)) === 0) }; S.dirty = true; toast(`${n} overlaid`); }; li.append(b); return li; })); };
  savedList();
  const hg = document.createElement("h3"); hg.textContent = "Earlier branches: overlay in grey"; box.append(hg);
  const gl = document.createElement("ul"); gl.className = "log"; gl.id = "ghosts"; box.append(gl);
  const h = document.createElement("h3"); h.textContent = "Commands and events"; box.append(h);
  const log = document.createElement("ul"); log.className = "log"; log.id = "log-list"; box.append(log);
  return box;
}
function paintLog() {
  const log = $("log-list");
  if (log) {
    const st = S.st, items = [...(st.commands || []).map(c => [c.t, JSON.stringify(c.command) + (Object.keys(c.answer || {}).length ? "  -> " + JSON.stringify(c.answer) : "")]), ...(st.events || []).filter(e => !["pellet", "spi", "gas", "laser_blowoff", "nbi", "icrh", "ecrh", "rmp", "ip", "density_control", "radiation_control", "mitigation", "fishbone", "alfven", "wall_flake", "marfe", "back_transition", "transport_barrier", "rwm", "vde", "limits"].includes(e.what)).map(e => [e.t, e.what]), ...(st.injected || []).map(e => [e.t, `island ${e.mode} ${fmt(e.w_cm, 3)} cm (${e.kind})`])].sort((a, b) => b[0] - a[0]).slice(0, 120);
    log.replaceChildren(...items.map(([t, text]) => { const li = document.createElement("li"), tm = document.createElement("time"); tm.textContent = fmt(t, 5); li.append(tm, document.createTextNode(text)); return li; }));
  }
  const gl = $("ghosts");
  if (gl) gl.replaceChildren(...(S.st.ghosts || []).map((g, i) => { const li = document.createElement("li"), b = document.createElement("button"); b.textContent = g.name; b.className = "ghost"; b.onclick = async () => { S.ghost = await api(`/api/ghost?i=${i}`); S.dirty = true; }; li.append(b); return li; }));
  const pl = $("params-list");
  if (pl && document.activeElement && !document.activeElement.closest("#console")) pl.replaceChildren(...Object.entries(S.st.params || {}).map(([k, v]) => { const li = document.createElement("li"); li.textContent = `${k} = ${fmt(v, 4)}`; return li; }));
}
function tableView() {
  const det = document.querySelector(".table-view");
  if (!det.open || !S.t.length) return;
  const keys = ["P_fusion", "P_SOL", "P_rad", "W_thermal", "T_e0", "T_i0", "n_line", "P_cmd", ...Object.keys(S.data).filter(k => /^w_\d+_\d+$/.test(k))].filter(k => k in S.data);
  const tb = document.createElement("table"), head = tb.createTHead().insertRow();
  const th = t => { const c = document.createElement("th"); c.textContent = t; head.append(c); };
  th("t (s)"); keys.forEach(k => th(labelOf(k)));
  const body = tb.createTBody();
  for (let i = S.t.length - 1; i >= Math.max(0, S.t.length - 25); i--) { const r = body.insertRow(); r.insertCell().textContent = fmt(S.t[i], 5); keys.forEach(k => { r.insertCell().textContent = fmt(S.data[k][i] * scaleOf(k)); }); }
  $("table").replaceChildren(tb);
}

// ---------- the frame loop
function render(now) {
  const animating = (frameAt(S.pinned ? S.cursor : null)?.islands || []).some(i => i.w > 0 && i.f && Math.abs(i.f) > 1) || now - S.flash.elm < 900 || now - S.flash.saw < 700;
  if (S.meta && (S.dirty || animating)) {
    if (S.dirty) {
      const R = range(), cur = S.cursor;
      drawRug(R, cur); drawStrips(R, cur); drawHeat(R); drawAxis(R); drawProfiles(cur);
      paintLog();
      if (Math.random() < .1) tableView();
      S.dirty = false;
    }
    drawXsec(S.pinned || S.cursor != null ? S.cursor : null, now);
  }
  requestAnimationFrame(render);
}

// ---------- controls
$("run").onclick = () => api(S.st.running ? "/api/pause" : "/api/run", S.st.running ? {} : { speed: parseFloat($("speed").value) });
$("step").onclick = () => api("/api/step", { n: 1 });
$("speed").onchange = () => { if (S.st.running) api("/api/run", { speed: parseFloat($("speed").value) }); };
$("scenario").onchange = async e => { await api("/api/load", { scenario: e.target.value }); S.rev = -1; };
$("decision-apply").onclick = () => send({ cmd: "decision", spec: S.meta.decisions[$("decision").value] });
$("window").onchange = e => { S.window = parseFloat(e.target.value); S.zoom = null; S.dirty = true; };
$("heat-field").onchange = e => { S.heat = e.target.value; S.dirty = true; };
$("add-strip").onclick = () => { $("strip-menu").hidden = !$("strip-menu").hidden; };
$("pin-clear").onclick = () => { S.pinned = false; S.cursor = null; S.zoom = null; S.measured = null; $("pin-clear").hidden = true; S.dirty = true; };
let marks = 0;
$("branch").onclick = async () => { marks += 1; const a = await api("/api/branch", { name: `mark ${marks}` }); toast(a.error || `mark ${marks} at ${fmt(a.t, 5)} s`); };
$("rewind").onclick = async () => { const n = $("branches").value; if (!n) return toast("no mark to go back to"); const a = await api("/api/rewind", { name: n }); if (a.error) toast(a.error); else { const q = S.st.ghosts ? S.st.ghosts.length : 0; S.ghost = await api(`/api/ghost?i=${q}`); } };
addEventListener("keydown", e => {
  if (e.target.closest("input, select, textarea")) return;
  if (e.code === "Space") { e.preventDefault(); $("run").click(); }
  if (e.key === "s") $("step").click();
});
addEventListener("resize", () => (S.dirty = true));
document.querySelector(".table-view").addEventListener("toggle", tableView);

buildProfiles();
buildPlane();
(async () => { await loadMeta().catch(() => {}); poll(); requestAnimationFrame(render); const again = setInterval(async () => { if (S.meta && S.meta.geometry) return clearInterval(again); await loadMeta().catch(() => {}); }, 1500); })();
