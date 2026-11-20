"use strict";
/* Athena test harness front end. All server/LLM text is inserted with
   textContent (never innerHTML): responses under test are untrusted. */

const $ = (id) => document.getElementById(id);
const STATES = ["verified", "plausible", "unsubstantiated", "contradicted", "inadmissible"];
const STATE_HELP = {
  verified: "consistent with supplied known facts and passes all checks",
  plausible: "structurally sound, but not independently verified",
  unsubstantiated: "asserted without available support (absence of evidence)",
  contradicted: "conflicts with a supplied fact or another claim",
  inadmissible: "breaks a structural rule or a declared boundary; unusable at any risk level",
};
const VERDICT = {
  accepted: ["✓", "Accepted", "All claims passed at this risk level."],
  accepted_with_warnings: ["⚠", "Accepted with warnings", "Usable, but some claims carry warnings or are unverified."],
  rejected: ["✕", "Rejected", "At least one claim is inadmissible or contradicted, or the risk level forbids acceptance."],
  escalated: ["⇧", "Escalated: human review required", "The risk level means the pipeline will not decide autonomously."],
};

function el(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined && v !== false) n.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined) n.append(kid);
  return n;
}
const add = (parent, ...kids) => parent.append(...kids.flat().filter((k) => k !== null && k !== undefined));
const lines = (id) => $(id).value.split("\n").map((s) => s.trim()).filter(Boolean);
let ID_MAP = {};
const human = (t) => Object.entries(ID_MAP).reduce((acc, [id, n]) => acc.split(id).join("#" + n).split(id.slice(0, 8)).join("#" + n), String(t));
const label = (s) => s.replaceAll("_", " ");

/* ---------------------------------------------------------------- presets */
const PRESETS = {
  grounded: {
    name: "Grounded answer (traffic)",
    query: "You are a traffic operations advisor. Report the current reading for Corridor A and advise the duty officer. Cite the sensor ID for every number. Do not infer causation from correlation.",
    response: "Sensor CAM-042 reports 87 veh/km on Corridor A. The speed limit on Corridor A is 40 km/h. The operator should notify the duty officer.",
    goal: "Brief the duty officer", action_class: "recommend", consequence_class: "low", domain: "traffic operations",
    known_facts: "Sensor CAM-042 reported 87 veh/km on Corridor A\nCorridor A speed limit is 40 km/h",
    constraints: "Must cite the sensor ID for every numerical claim", boundaries: "Must not infer causation from correlation",
  },
  overreach: {
    name: "Unsupported cause and forecast",
    query: "You are a traffic operations advisor. Analyse Corridor A and recommend whether to trigger a manual signal override. Do not infer causation from correlation.",
    response: "Vehicle density on Corridor A is 94 veh/km, which is an anomaly. This was caused by the earlier signal fault, and it will definitely clear soon. I recommend triggering a manual override.",
    goal: "Decide whether to trigger a manual signal override", action_class: "decide", consequence_class: "significant",
    domain: "traffic operations", decision_window: "90 seconds",
    term_definitions: "anomaly = sensor reading exceeding 3 sigma from the 24h rolling mean",
    boundaries: "Must not infer causation from correlation", known_facts: "", constraints: "",
  },
  contradicted: {
    name: "Contradicts a supplied fact",
    query: "Summarise the road rules and live status for Corridor A.",
    response: "Corridor A has a 60 km/h speed limit and vehicle density is currently 12 veh/km.",
    goal: "Inform a route planner", action_class: "inform", consequence_class: "low", domain: "traffic operations",
    known_facts: "Corridor A speed limit is 40 km/h", constraints: "", boundaries: "",
  },
};
// "automate" reuses the grounded text so only the risk context differs.
PRESETS.automate = { ...PRESETS.grounded, name: "Same grounded answer, automated + critical",
  action_class: "automate", consequence_class: "critical", goal: "Apply the timing change with no human review" };

function fillForm(p) {
  const set = (id, v) => { $(id).value = v ?? ""; };
  for (const id of ["query", "response", "goal", "domain", "decision_window", "actor", "description",
                    "known_facts", "constraints", "boundaries", "term_definitions", "model"]) set(id, p[id]);
  if (!p.domain) $("domain").value = "general";
  $("action_class").value = p.action_class || "recommend";
  $("consequence_class").value = p.consequence_class || "low";
  if (p.known_facts || p.constraints || p.boundaries || p.term_definitions) $("contract").open = true;
  refreshRisk();
}
for (const [k, p] of Object.entries(PRESETS)) $("preset").append(el("option", { value: k, text: p.name }));
$("preset").addEventListener("change", (e) => { if (PRESETS[e.target.value]) fillForm(PRESETS[e.target.value]); });

/* ---------------------------------------------------------------- server */
async function api(path, opts) {
  const r = await fetch(path, opts);
  let body = null;
  try { body = await r.json(); } catch { /* non-JSON error page */ }
  if (!r.ok) throw new Error((body && (body.error || body.detail)) || `HTTP ${r.status}`);
  return body;
}

async function loadHealth() {
  const h = $("health");
  try {
    const d = await api("/api/health");
    h.replaceChildren("model ", el("b", { text: d.model }), " · API key ",
      el("b", { class: d.key_configured ? "ok" : "bad", text: d.key_configured ? "configured" : "MISSING" }));
    if (!d.key_configured) showFormError("No Gemini API key on the server. Set GEMINI_API_KEY (or GOOGLE_API_KEY), restart, and reload.");
    window.__maxChars = d.max_chars;
  } catch (e) { h.textContent = "server unreachable"; }
}

let riskSeq = 0;
async function refreshRisk() {
  const seq = ++riskSeq;
  try {
    const q = new URLSearchParams({ action_class: $("action_class").value, consequence_class: $("consequence_class").value });
    const d = await api("/api/risk?" + q);
    if (seq !== riskSeq) return;
    $("risk").textContent = d.risk_level;
    $("risk").className = "chip " + d.risk_level;
  } catch { $("risk").textContent = "?"; }
}
$("action_class").addEventListener("change", refreshRisk);
$("consequence_class").addEventListener("change", refreshRisk);

/* ---------------------------------------------------------------- submit */
function showFormError(msg) { const e = $("formerr"); e.textContent = msg; e.hidden = !msg; }

function termsFromText() {
  const out = {};
  for (const l of lines("term_definitions")) {
    const i = l.indexOf("=");
    if (i > 0) out[l.slice(0, i).trim()] = l.slice(i + 1).trim();
  }
  return out;
}

function payload() {
  return {
    query: $("query").value, response: $("response").value, goal: $("goal").value,
    action_class: $("action_class").value, consequence_class: $("consequence_class").value,
    domain: $("domain").value || "general", actor: $("actor").value, decision_window: $("decision_window").value,
    description: $("description").value, known_facts: lines("known_facts"), constraints: lines("constraints"),
    boundaries: lines("boundaries"), term_definitions: termsFromText(), model: $("model").value,
  };
}

let lastResult = null, timer = null;
$("form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  showFormError("");
  const p = payload();
  const missing = ["query", "response", "goal"].filter((k) => !p[k].trim());
  if (missing.length) return showFormError("Please fill in: " + missing.join(", ") + ".");
  const max = window.__maxChars || 20000;
  if (p.query.length > max || p.response.length > max) return showFormError(`Query and response are limited to ${max} characters each.`);

  setBusy(true);
  try {
    const d = await api("/api/validate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(p) });
    lastResult = { request: p, result: d };
    render(d, p);
  } catch (e) {
    $("result").hidden = true;
    showFormError(e.message);
  } finally { setBusy(false); }
});
$("form").addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") $("form").requestSubmit();
});
$("clear").addEventListener("click", () => {
  $("form").reset(); $("preset").value = ""; $("domain").value = "general"; showFormError("");
  $("result").hidden = true; $("empty").hidden = false; refreshRisk();
});

function setBusy(on) {
  $("run").disabled = on;
  $("busy").hidden = !on;
  if (on) {
    $("empty").hidden = true; $("result").hidden = true;
    const t0 = Date.now();
    clearInterval(timer);
    timer = setInterval(() => { $("busyhint").textContent = `Running… ${Math.round((Date.now() - t0) / 1000)} s elapsed (several model calls).`; }, 1000);
  } else clearInterval(timer);
}

/* ---------------------------------------------------------------- render */
function stateBadge(s) { return el("span", { class: `badge s-${s}`, title: STATE_HELP[s], text: s }); }

function locateSpan(text, span, taken) {
  // exact first, then whitespace-tolerant; returns [start,end] or null
  let i = text.indexOf(span);
  if (i >= 0) return overlap(taken, i, i + span.length) ? null : [i, i + span.length];
  const esc = span.trim().replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s+");
  const m = esc ? new RegExp(esc).exec(text) : null;
  if (!m) return null;
  return overlap(taken, m.index, m.index + m[0].length) ? null : [m.index, m.index + m[0].length];
}
const overlap = (taken, a, b) => taken.some(([s, e]) => a < e && b > s);

function highlighted(text, claims) {
  const taken = [], marks = [];
  claims.forEach((c, idx) => {
    const r = locateSpan(text, c.source_span, taken);
    if (r) { taken.push(r); marks.push({ r, c, n: idx + 1 }); }
  });
  marks.sort((a, b) => a.r[0] - b.r[0]);
  const box = el("div", { class: "hl", role: "group", "aria-label": "Response with extracted claims marked" });
  let pos = 0;
  for (const m of marks) {
    if (m.r[0] > pos) box.append(text.slice(pos, m.r[0]));
    box.append(el("mark", {
      class: `s-${m.c.state}`, tabindex: "0", title: `Claim ${m.n}: ${m.c.state}`,
      onclick: () => jump(m.n), onkeydown: (e) => { if (e.key === "Enter") jump(m.n); },
    }, text.slice(m.r[0], m.r[1]), el("sup", { text: String(m.n) })));
    pos = m.r[1];
  }
  if (pos < text.length) box.append(text.slice(pos));
  return { box, located: marks.length };
}
function jump(n) {
  const c = $("claim-" + n); if (!c) return;
  c.scrollIntoView({ behavior: "smooth", block: "center" });
  c.classList.add("flash"); setTimeout(() => c.classList.remove("flash"), 1400);
}

function claimCard(c, n, numById) {
  const refs = (ids) => ids.map((id) => "#" + (numById[id] ?? "?")).join(", ");
  const card = el("article", { class: `claim s-${c.state}`, id: "claim-" + n },
    el("header", {}, el("span", { class: "n", text: "#" + n }), stateBadge(c.state),
      el("span", { class: "tag", text: c.type }), el("span", { class: "tag", text: c.modality }),
      c.propagated_failure ? el("span", { class: "tag", title: "state inherited from an upstream claim", text: "inherited failure" }) : null,
      c.has_warnings ? el("span", { class: "tag", text: "warnings" }) : null),
    el("p", { class: "prop", text: c.proposition }),
    el("p", { class: "why" }, human(c.rationale).split(" | ").map((t) => el("span", { text: t }))),
    el("p", { class: "span", text: "“" + c.source_span + "”" }));

  const kv = [];
  if (c.scope) kv.push(["scope", c.scope]);
  if (c.confidence_language) kv.push(["confidence language", c.confidence_language]);
  if (c.time_horizon) kv.push(["time horizon", c.time_horizon.raw + (c.time_horizon.normalized ? ` (${c.time_horizon.normalized})` : " (not quantified)")]);
  if (c.assumptions.length) kv.push(["assumptions", c.assumptions.join("; ")]);
  if (c.conditions.length) kv.push(["conditions", c.conditions.join("; ")]);
  if (c.depends_on.length) kv.push(["depends on", refs(c.depends_on)]);
  if (c.negates.length) kv.push(["negates", refs(c.negates)]);
  if (kv.length) add(card, el("p", { class: "kv" }, kv.flatMap(([k, v], i) => [i ? " · " : "", el("b", { text: k + ": " }), v])));
  for (const w of c.warnings) card.append(el("div", { class: "note warn", text: human(w) }));

  const rows = c.guards.map((g) => el("tr", {},
    el("td", { text: g.guard_id }), el("td", { class: "o-" + g.outcome, text: g.outcome }),
    el("td", { text: human(g.message) + (Object.keys(g.detail).length ? "  " + Object.entries(g.detail).map(([k, v]) => `${k}: ${v}`).join(" · ") : "") })));
  if (rows.length) card.append(el("details", {}, el("summary", { text: `Guards (${c.guards.filter((g) => g.outcome === "failed").length} failed, ${c.guards.filter((g) => g.outcome === "errored").length} errored)` }),
    el("table", { class: "g" }, el("tr", {}, el("th", { text: "Guard" }), el("th", { text: "Outcome" }), el("th", { text: "Why" })), rows)));
  return card;
}

function render(d, req) {
  const root = $("result"); root.replaceChildren();
  ID_MAP = Object.fromEntries(d.claims.map((c, i) => [c.claim_id, i + 1]));
  const [icon, title, sub] = VERDICT[d.verdict.code] || ["?", d.verdict.code, ""];
  add(root, 
    el("div", { class: `verdict v-${d.verdict.code}` }, el("div", { class: "icon", "aria-hidden": "true", text: icon }),
      el("div", {}, el("h3", { text: title }), el("p", { text: d.verdict.rationale }), el("p", { class: "hint", text: sub }))),
    el("div", { class: "meta" },
      el("span", {}, "Risk level ", el("b", { class: `chip ${d.verdict.risk_level}`, text: d.verdict.risk_level })),
      el("span", {}, "Claims ", el("b", { text: String(d.total_claims) })),
      el("span", {}, "Inherited failures ", el("b", { text: String(d.propagated_failures) })),
      el("span", {}, "Run ", el("b", { text: (d.run_id || "").slice(0, 8) }))));

  // distribution
  const total = Math.max(d.total_claims, 1);
  add(root, el("h3", { class: "sec", text: "Claim states" }),
    el("div", { class: "dist", role: "img", "aria-label": STATES.map((s) => `${d.counts[s]} ${s}`).join(", ") },
      STATES.filter((s) => d.counts[s]).map((s) => el("span", { class: `s-${s}`, style: `width:${(100 * d.counts[s]) / total}%`, title: `${s}: ${d.counts[s]}` }))),
    el("div", { class: "legend" }, STATES.map((s) => el("span", { class: `s-${s}`, title: STATE_HELP[s] }, el("i", { class: "dot" }), `${s} ${d.counts[s]}`))));

  // honest notes
  const notes = [...d.notes, ...d.pipeline_warnings.map((t) => ({ level: "warn", text: human(t) }))];
  if (notes.length) add(root, el("h3", { class: "sec", text: "What this run could and could not establish" }),
    notes.map((n) => el("div", { class: "note " + (n.level === "warn" ? "warn" : ""), text: n.text })));

  // highlighted response
  if (d.claims.length) {
    const { box, located } = highlighted(req.response, d.claims);
    add(root, el("h3", { class: "sec", text: "Response, claims marked" }), box);
    if (located < d.claims.length) add(root, el("p", { class: "hint", text: `${d.claims.length - located} claim(s) could not be located in the text (overlapping or paraphrased span); see the list below.` }));
  }

  // claims
  const numById = Object.fromEntries(d.claims.map((c, i) => [c.claim_id, i + 1]));
  add(root, el("h3", { class: "sec", text: `Claims (${d.claims.length})` }), d.claims.map((c, i) => claimCard(c, i + 1, numById)));

  // decomposition
  const dp = d.decomposition;
  if (dp) {
    const list = (title, xs) => xs.length ? [el("b", { text: title }), el("ul", { class: "plain" }, xs.map((x) => el("li", { text: x })))] : [];
    add(root, el("details", {}, el("summary", { text: "How the prompt was understood (decomposition)" }),
      list("Explicit constraints", dp.explicit_constraints), list("Implicit constraints (inferred)", dp.implicit_constraints),
      list("Explicit boundaries", dp.explicit_boundaries), list("Implicit boundaries (inferred)", dp.implicit_boundaries),
      list("Undeclared terms", dp.undeclared_terms), list("Facts found in the prompt", dp.injected_facts),
      list("Extraction notes", dp.extraction_notes),
      Object.keys(dp.term_definitions).length ? [el("b", { text: "Term definitions" }), el("ul", { class: "plain" }, Object.entries(dp.term_definitions).map(([k, v]) => el("li", { text: `${k}: ${v}` })))] : []));
  }
  add(root, el("details", {}, el("summary", { text: "Raw JSON" }), el("pre", { class: "raw", text: JSON.stringify(d, null, 2) })),
    el("div", { class: "tools" },
      el("button", { type: "button", text: "Download JSON", onclick: downloadJson }),
      el("button", { type: "button", text: "Copy JSON", onclick: (e) => { navigator.clipboard?.writeText(JSON.stringify(lastResult, null, 2)); e.target.textContent = "Copied"; setTimeout(() => (e.target.textContent = "Copy JSON"), 1200); } })));
  root.hidden = false; $("empty").hidden = true;
}

function downloadJson() {
  const blob = new Blob([JSON.stringify(lastResult, null, 2)], { type: "application/json" });
  const a = el("a", { href: URL.createObjectURL(blob), download: `athena-${(lastResult.result.run_id || "run").slice(0, 8)}.json` });
  document.body.append(a); a.click(); a.remove();
}

loadHealth(); refreshRisk();
