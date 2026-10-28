"use strict";
const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const LANG = { en: "English", es: "Spanish" };
const state = { visits: [], visit: null, shown: 0, run: 0, health: {} };

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try {
      const d = (await r.json()).detail;
      msg = Array.isArray(d) ? d.map((x) => x.msg || String(x)).join("; ") : d || msg;
    } catch (_) { /* keep status text */ }
    throw new Error(msg);
  }
  return r.json();
}

/* ---------- highlighting: findings carry folded words (lowercase, no accents); map them back onto the original ---------- */
function fold(text) {
  return Array.from(text.normalize("NFC")).map((ch) => {
    const f = ch.normalize("NFD").replace(/\p{M}/gu, "").toLowerCase().replace("’", "'");
    return f.length ? f[0] : ch;
  }).join("");
}
function marked(text, spans) {
  // spans: [{words, cls}] -> HTML with <mark> around the first whole-word match of each
  text = text.normalize("NFC");
  const chars = Array.from(text);
  const f = fold(text);
  const tags = new Array(chars.length + 1).fill("");
  const used = new Array(chars.length).fill(false);
  for (const s of spans) {
    if (!s.words) continue;
    const w = fold(s.words);
    let i = -1, from = 0;
    const wordy = /[\p{L}\p{N}]/u;
    while ((i = f.indexOf(w, from)) !== -1) {
      const okEdges = !(i > 0 && wordy.test(f[i - 1]) && wordy.test(w[0])) && !(wordy.test(f[i + w.length] || "") && wordy.test(w[w.length - 1]));
      if (okEdges && !used.slice(i, i + w.length).some(Boolean)) break;
      from = i + 1;
    }
    if (i === -1) continue;
    for (let k = i; k < i + w.length; k++) used[k] = true;
    tags[i] += `<mark class="${s.cls}">`;
    tags[i + w.length] = "</mark>" + tags[i + w.length];
  }
  let out = "";
  chars.forEach((c, k) => { out += tags[k] + esc(c); });
  return out + tags[chars.length];
}

/* ---------- visits ---------- */
async function loadVisits() {
  state.visits = await getJSON("/api/visits");
  const box = $("#visits");
  box.innerHTML = state.visits.map((v, i) =>
    `<button type="button" aria-pressed="${i === 0}" data-id="${v.id}"><span class="t">${esc(v.title)}</span><span class="s">${esc(v.setting)}</span></button>`).join("");
  box.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-id]");
    if (!b) return;
    box.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    openVisit(b.dataset.id);
  });
  if (state.visits.length) openVisit(state.visits[0].id);
}

async function openVisit(id) {
  stop();
  state.visit = await getJSON(`/api/visit/${id}`);
  state.shown = 0;
  $("#lh-clin").textContent = `${state.visit.clinician}`;
  $("#lh-pat").textContent = `${state.visit.patient}`;
  $("#turns").innerHTML = "";
  $("#before").hidden = true;
  $("#status").textContent = `${state.visit.turns.length} turns. Press Play, or step through.`;
}

function spansFor(findings) {
  const src = [], ren = [];
  for (const f of findings) {
    if (f.severity === "low") continue;
    if (f.source && (f.type === "omission" || f.type === "substitution") && f.kind !== "order") src.push({ words: f.source, cls: "lost" });
    if (f.rendition && (f.type === "substitution" || f.type === "addition") && f.kind !== "order") ren.push({ words: f.rendition, cls: "changed" });
  }
  return { src, ren, hole: findings.some((f) => f.type === "omission" && f.severity !== "low") };
}

function cardHTML(t) {
  const c = t.check, d = c.decision;
  if (d.level === "ok") return "";
  if (d.level === "note") {
    const say = ((c.explanation || {}).say || "a possible change in critical content").replace(/^Re-confirm\.\s*/, "");
    return `<div class="notecard">Check later: ${esc(say)}</div>`;
  }
  // "seconds after the turn" only means something when speech recognition actually ran on this turn's audio
  const after = t.heard && t.heard.asr ? (t.heard.asr_ms || 0) + (c.ms ? c.ms.to_decision : 0) : 0;
  const chips = c.findings.filter((f) => f.severity !== "low").slice(0, 5).map((f) => `<li>${esc(f.message)}</li>`).join("");
  const why = [];
  if (c.p_faithful !== null && c.p_faithful !== undefined) why.push(`<span>Our checker: ${Math.round((1 - c.p_faithful) * 100)}% likely changed</span>`);
  if (c.findings.length) why.push(`<span>Code: ${c.findings.filter((f) => f.severity !== "low").length} word-level change${c.findings.length === 1 ? "" : "s"}</span>`);
  why.push(`<span>Decided by ${esc(d.why)}</span>`);
  return `<div class="card ${d.title === "Possible change" ? "possible" : ""}" role="note">
    <div class="band"><span>${esc(d.title)}</span>${after >= 50 ? `<span class="when">${(after / 1000).toFixed(1)} s after the turn</span>` : ""}</div>
    <p class="say">${esc((c.explanation || {}).say || "")}</p>
    ${chips ? `<ul>${chips}</ul>` : ""}
    <p class="why">${why.join("")}</p></div>`;
}

function turnHTML(t, stage) {
  const clin = t.speaker === "clinician";
  const h = t.heard || t.script;
  const sp = stage >= 3 ? spansFor(t.check.findings) : { src: [], ren: [], hole: false };
  const srcName = clin ? state.visit.clinician : state.visit.patient;
  const src = `<div class="cell src ${clin ? "clin" : "pat"}"><span class="who"><span class="lang ${t.src_lang}">${t.src_lang.toUpperCase()}</span>${esc(srcName)}</span>
    <p class="said" lang="${t.src_lang}">${marked(h.source, sp.src)}</p></div>`;
  const mid = `<div class="cell mid"><span class="who"><span class="lang ${t.ren_lang}">${t.ren_lang.toUpperCase()}</span>Interpreter</span>
    <p class="said" lang="${t.ren_lang}">${stage >= 2 ? marked(h.rendition, sp.ren) + (sp.hole ? '<span class="hole" role="img" aria-label="missing words"></span>' : "") : "…"}</p>
    ${stage >= 3 && h.asr ? `<div class="heard">Heard by speech recognition. Script: “${esc(t.script.rendition)}”</div>` : ""}
    ${stage >= 3 ? cardHTML(t) : ""}
    ${stage >= 3 && t.planted && $("#planted").checked ? `<div class="planted"><b>Planted ${esc(t.planted.type)}:</b> ${esc(t.planted.what)}</div>` : ""}
    ${stage >= 3 && !t.planted && $("#planted").checked && t.check.decision.level === "card" ? `<div class="planted"><b>No planted error here:</b> this card is a false alarm.</div>` : ""}</div>`;
  const empty = `<div class="cell empty ${clin ? "pat" : "clin"}"></div>`;
  return src + mid + empty;
}

function showTurn(i, stage) {
  const t = state.visit.turns[i];
  let li = document.querySelector(`#turns li[data-i="${i}"]`);
  if (!li) {
    li = document.createElement("li");
    li.className = "turn";
    li.dataset.i = i;
    $("#turns").appendChild(li);
  }
  li.classList.toggle("pending", stage < 3);
  li.classList.toggle("enter", stage === 3);
  li.innerHTML = turnHTML(t, stage);
  return li;
}

/* ---------- replay ---------- */
// Every await checks the run token, so Pause, Next turn and switching visits never leave a turn half-shown or skipped.
function stop() { state.run++; state.busy = false; $("#play").textContent = "Play visit"; }
function sleep(ms, run) {
  return new Promise((resolve) => {
    const t0 = Date.now();
    const iv = setInterval(() => {
      if ((run !== undefined && run !== state.run) || Date.now() - t0 >= ms) { clearInterval(iv); resolve(); }
    }, 50);
  });
}
function playClip(url, fallbackMs, run) {
  return new Promise((resolve) => {
    if (!url || !$("#sound").checked) { sleep(fallbackMs, run).then(resolve); return; }
    const a = new Audio("/static/" + url);
    let settled = false;
    const done = () => { if (!settled) { settled = true; clearInterval(iv); resolve(); } };
    const iv = setInterval(() => { if (run !== state.run) { a.pause(); done(); } }, 100);
    a.addEventListener("ended", done, { once: true });
    a.addEventListener("error", () => sleep(fallbackMs, run).then(done), { once: true });
    a.play().catch(() => sleep(fallbackMs, run).then(done));
  });
}
async function playFrom(i) {
  const run = ++state.run;
  $("#play").textContent = "Pause";
  for (let k = i; k < state.visit.turns.length; k++) {
    if (!(await revealTurn(k, run))) return;
    state.shown = k + 1;
    await sleep(500, run);
    if (run !== state.run) return;
  }
  if (run === state.run) { finish(); stop(); }
}
async function revealTurn(k, run) {
  // -> true when the turn was fully shown; false (and the turn removed, to replay later) when interrupted
  const t = state.visit.turns[k];
  const li = showTurn(k, 1);
  const visit = state.visit;
  const gone = () => { if (state.visit === visit) li.remove(); return false; };
  li.scrollIntoView({ block: "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  $("#status").textContent = `Turn ${k + 1} of ${state.visit.turns.length}`;
  const len = (s) => 600 + 55 * (s || "").length;
  await playClip(t.audio && t.audio.src, len(t.script.source), run);
  if (run !== state.run) return gone();
  showTurn(k, 2);
  await playClip(t.audio && t.audio.ren, len(t.script.rendition), run);
  if (run !== state.run) return gone();
  const wait = t.heard && t.heard.asr ? Math.min((t.heard.asr_ms || 0) + ((t.check.ms || {}).to_decision || 0), 3000) : 300;
  await sleep(wait, run);
  if (run !== state.run) return gone();
  showTurn(k, 3);
  announce(t);
  return true;
}
function announce(t) {
  const d = t.check.decision;
  if (d.level === "card") $("#announce").textContent = `${d.title}: ${(t.check.explanation || {}).say || ""}`;
}
function finish() {
  const v = state.visit;
  const items = [];
  v.turns.forEach((t) => {
    const d = t.check.decision;
    if (d.level !== "ok") items.push({ t: (t.check.explanation || {}).say || d.title, q: t.heard ? t.heard.source : t.script.source, by: `Turn ${t.n}, ${d.level === "card" ? "re-confirm card" : "note"}` });
  });
  (v.review || []).forEach((r) => items.push({ t: r.confirm, q: r.source_quote, by: `Turn ${r.turn}, found by Nemotron 3 Ultra after the visit` }));
  const caught = v.turns.filter((t) => t.planted && t.check.decision.level === "card").length;
  const planted = v.turns.filter((t) => t.planted).length;
  const fa = v.turns.filter((t) => !t.planted && t.check.decision.level === "card").length;
  $("#before-list").innerHTML = items.length ? items.map((x, i) => `<li><label><input type="checkbox" data-i="${i}"><span><span class="t">${esc(x.t)}</span><br><span class="q">“${esc(x.q)}”</span><br><span class="by">${esc(x.by)}</span></span></label></li>`).join("")
    : "<li>Nothing to re-confirm.</li>";
  $("#before").hidden = false;
  $("#status").textContent = `Visit done. Cards on ${caught} of ${planted} planted errors; ${fa} false alarm${fa === 1 ? "" : "s"}.`;
}
$("#before-list").addEventListener("change", (e) => { const li = e.target.closest("li"); if (li) li.classList.toggle("done", e.target.checked); });
$("#play").addEventListener("click", () => {
  if ($("#play").textContent === "Pause") { stop(); return; }
  if (state.shown >= state.visit.turns.length) { $("#turns").innerHTML = ""; $("#before").hidden = true; state.shown = 0; }
  playFrom(state.shown);
});
$("#step").addEventListener("click", async () => {
  if (state.busy) return;  // one turn at a time
  stop();
  if (state.shown >= state.visit.turns.length) return;
  const run = ++state.run;
  state.busy = true;
  const ok = await revealTurn(state.shown, run);
  if (run !== state.run) return;
  state.busy = false;
  if (!ok) return;
  state.shown++;
  if (state.shown === state.visit.turns.length) finish();
});
$("#all").addEventListener("click", () => {
  stop();
  state.visit.turns.forEach((_, i) => showTurn(i, 3));
  document.querySelectorAll("#turns li").forEach((li) => li.classList.remove("enter"));
  state.shown = state.visit.turns.length;
  finish();
});
$("#planted").addEventListener("change", () => {
  for (let i = 0; i < state.shown; i++) showTurn(i, 3);  // only fully shown turns are re-drawn
  document.querySelectorAll("#turns li").forEach((li) => li.classList.remove("enter"));
});

/* ---------- check a sentence ---------- */
const langIn = () => document.querySelector('input[name="lang"]:checked').value;
document.querySelectorAll('input[name="lang"]').forEach((r) => r.addEventListener("change", () => { $("#lang-out").textContent = LANG[langIn() === "en" ? "es" : "en"]; }));
$("#tryform").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("#t-status").textContent = "Checking…";
  $("#t-out").innerHTML = "";
  try {
    const src_lang = langIn();
    const res = await getJSON("/api/check", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: $("#t-src").value, rendition: $("#t-ren").value, src_lang }) });
    const t = { check: res, heard: null, speaker: "clinician" };
    const parts = [];
    if (res.decision.level === "ok") parts.push(`<p class="okline">No critical change found.</p>`);
    else parts.push(cardHTML(t));
    const live = [];
    live.push(res.checker === "on" ? "our fine-tuned checker on Nebius" : "code only (our checker’s GPU is switched off right now)");
    if (res.live.explainer && res.decision.level === "card") live.push("Nemotron 3.5 Lightning for the wording");
    parts.push(`<p class="small">Checked with ${live.join(" and ")} in ${Math.round(res.ms.total)} ms.</p>`);
    $("#t-out").innerHTML = parts.join("");
    $("#t-status").textContent = "";
  } catch (err) {
    $("#t-status").textContent = `Couldn’t check: ${err.message}`;
  }
});

/* recording (only when the speech endpoint is up): browser audio -> 16 kHz WAV -> /api/asr */
function wav16k(buf) {
  const ch = buf.getChannelData(0), n = ch.length, out = new DataView(new ArrayBuffer(44 + n * 2));
  const w = (o, s) => { for (let i = 0; i < s.length; i++) out.setUint8(o + i, s.charCodeAt(i)); };
  w(0, "RIFF"); out.setUint32(4, 36 + n * 2, true); w(8, "WAVE"); w(12, "fmt "); out.setUint32(16, 16, true); out.setUint16(20, 1, true);
  out.setUint16(22, 1, true); out.setUint32(24, 16000, true); out.setUint32(28, 32000, true); out.setUint16(32, 2, true); out.setUint16(34, 16, true);
  w(36, "data"); out.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) out.setInt16(44 + i * 2, Math.max(-1, Math.min(1, ch[i])) * 0x7fff, true);
  return new Blob([out], { type: "audio/wav" });
}
async function toWav(blob) {
  const ctx = new AudioContext();
  const decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
  const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * 16000), 16000);
  const src = off.createBufferSource(); src.buffer = decoded; src.connect(off.destination); src.start();
  return wav16k(await off.startRendering());
}
function setupMics() {
  if (!state.health.checker_endpoint || !navigator.mediaDevices || !window.MediaRecorder) return;
  document.querySelectorAll(".mic").forEach((b) => {
    b.hidden = false;
    let rec = null, chunks = [];
    b.addEventListener("click", async () => {
      if (rec) { rec.stop(); return; }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        rec = new MediaRecorder(stream); chunks = [];
        rec.ondataavailable = (e) => chunks.push(e.data);
        rec.onstop = async () => {
          stream.getTracks().forEach((t) => t.stop());
          b.classList.remove("rec"); b.textContent = "Transcribing…";
          try {
            const fd = new FormData(); fd.append("file", await toWav(new Blob(chunks)), "turn.wav");
            const d = await getJSON("/api/asr", { method: "POST", body: fd });
            $("#" + b.dataset.for).value = d.text;
            b.textContent = "Record";
          } catch (err) { b.textContent = "Record"; $("#t-status").textContent = err.message; }
          rec = null;
        };
        rec.start(); b.classList.add("rec"); b.textContent = "Stop";
      } catch (err) { $("#t-status").textContent = "Microphone not available: " + err.message; }
    });
  });
}

/* ---------- results ---------- */
const pct = (x) => `${Math.round(x * 100)}%`;
async function loadResults() {
  let d;
  try { d = await getJSON("/api/eval"); } catch (_) { return; }
  const e = d.e2e;
  if (e && (e.from_audio || e.from_script_text)) {
    const a = e.from_audio || e.from_script_text;
    const audio = !!e.from_audio;
    const figs = [
      [`${a.caught} of ${a.errors}`, audio ? "planted interpreter errors caught, straight from the audio" : "planted interpreter errors caught (from the script text)"],
      [`${a.patient_speech_caught}`, "caught in the patient’s own words, the errors AI scribes copy most"],
      [`${a.false_alarms} of ${a.faithful_turns}`, "correct turns flagged anyway"],
    ];
    if (audio && e.ms_median_audio_to_card >= 50) figs.push([`${(e.ms_median_audio_to_card / 1000).toFixed(1)} s`, "from the interpreter finishing to the card"]);
    const notes = (a.notes_on_faithful_turns || []).length;
    const how = audio
      ? `Our six scripted visits, voiced, heard by Parakeet on our Nebius endpoint${e.checker_used ? ", checked with our fine-tuned checker" : ""}. Speech recognition word error rate: English ${e.wer && e.wer.en !== null ? (e.wer.en * 100).toFixed(1) + "%" : "n/a"}, Spanish ${e.wer && e.wer.es !== null ? (e.wer.es * 100).toFixed(1) + "%" : "n/a"}.`
      : `Our six scripted visits, checked from the script text${e.checker_used ? " with our fine-tuned checker" : " by code alone (our checker wasn’t running)"}. The rules were written while looking at these visits, so this is an optimistic number; the benchmark below is the fair test.`;
    $("#e2e").innerHTML = figs.map(([b2, s2]) => `<div class="fig"><b>${esc(b2)}</b><span>${esc(s2)}</span></div>`).join("") +
      `<p class="small" style="grid-column:1/-1">For comparison, two ambient AI scribes carried 55–60% of the same kind of planted errors into the visit note (Rabotin et al., JMIR Medical Informatics 2026). ${esc(how)}${notes ? ` ${notes} correct turn${notes === 1 ? "" : "s"} also got a quiet “check later” note.` : ""}</p>`;
  } else {
    $("#e2e").innerHTML = `<p class="small">End-to-end results haven’t been recorded yet.</p>`;
  }
  const b = d.bench;
  if (b && b.results) {
    let html = "";
    for (const [set, rows] of Object.entries(b.results)) {
      const mt = (b.meta || {})[set] || {};
      const size = mt.rows ? `<p class="small">${mt.rows.toLocaleString()} turns${mt.of > mt.rows ? `, a random sample of ${mt.of.toLocaleString()}; every method scored on the same turns` : ""}.</p>` : "";
      html += `<div class="bench"><h3>${esc((b.set_names || {})[set] || set)}</h3>${size}<div class="tablewrap"><table><thead><tr><th>Method</th><th>Errors caught</th><th>False alarms</th><th>F1</th><th>Time per turn</th><th>Cost per 1,000 turns</th></tr></thead><tbody>`;
      for (const [m, r] of Object.entries(rows)) {
        html += `<tr class="${m === "faithful" ? "ours" : ""}"><td>${esc((b.names || {})[m] || m)}</td>
          <td><span class="bar" style="width:${Math.round(r.recall * 70)}px"></span>${r.caught}/${r.errors} (${pct(r.recall)})</td>
          <td><span class="bar fa" style="width:${Math.round(r.false_alarm_rate * 70)}px"></span>${r.false_alarms} (${(r.false_alarm_rate * 100).toFixed(1)}%)</td>
          <td>${r.f1.toFixed(2)}</td><td>${r.ms_median !== null ? `${r.ms_median < 10 ? r.ms_median.toFixed(1) : Math.round(r.ms_median)} ms` : "–"}</td><td>$${r.cost_per_1000.toFixed(2)}${r.failed ? ` <span class="small">(${r.failed} not answered, left out)</span>` : ""}</td></tr>`;
      }
      html += `</tbody></table></div></div>`;
    }
    const tuned = b.thresholds && b.thresholds.gate;
    $("#bench").innerHTML = html + `<p class="small">${tuned ? `Every threshold (checker alone, rerank before fine-tuning, and the decision rule) was chosen on ${b.thresholds.dev_rows ? b.thresholds.dev_rows.toLocaleString() + " " : ""}held-out practice turns, never on these test sets, with the same rule for each: the best F1 with false alarms at or under 5%. ` : ""}ACES (Amrhein et al., 2023) is licensed CC BY-NC-SA 4.0 and is downloaded at test time, not redistributed; its unit-conversion items and duplicate rows are left out.</p>`;
  } else {
    $("#bench").innerHTML = `<p class="small">The benchmark hasn’t been run yet.</p>`;
  }
  const tr = d.training;
  if (tr && tr.fine_tuned) {
    $("#training").innerHTML = `<div class="bench"><h3>Fine-tuning on Nebius</h3><p class="small">NVIDIA llama-nemotron-rerank-1b-v2, ${tr.train_pairs.toLocaleString()} synthetic training pairs, ${tr.epochs} epochs on one ${esc(tr.gpu)} in a Nebius Serverless Job (${Math.round(tr.seconds / 60)} min).
      Held-out test, critical errors: AUC ${tr.base_model.test_auc} before fine-tuning, ${tr.fine_tuned.test_auc} after; F1 ${tr.base_model.test_at_base_dev_threshold.f1} before, ${tr.fine_tuned.test_at_dev_threshold.f1} after.</p></div>`;
  }
}

// The opening button scrolls to the replay and starts the first visit (a click, so sound is allowed).
document.querySelector(".cta .primary-link")?.addEventListener("click", (ev) => {
  ev.preventDefault();
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  $("#replay").scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
  setTimeout(() => { const b = $("#play"); if (b && !b.disabled && !state.busy) b.click(); }, reduce ? 0 : 450);
});

(async function init() {
  try { state.health = await getJSON("/api/health"); } catch (_) { state.health = {}; }
  setupMics();
  try { await loadVisits(); } catch (err) { $("#status").textContent = `Couldn’t load the visits: ${err.message}`; }
  loadResults();
})();
