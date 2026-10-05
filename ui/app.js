/* ophbench clinician annotation UI — vanilla ES2020, no build step, no external dependencies.
 *
 * Layout of this file
 *   1. Pure helpers (no DOM access) — exported via module.exports for node-based tests.
 *   2. The browser application, started only when `window` and `document` exist.
 *
 * Data files (all relative to index.html, see DESIGN.md §9 / README_UI.md):
 *   data/meta.json, data/index.json, data/assignments.json, data/samples/<sample_id>.json
 * Annotations are autosaved to localStorage under "ophbench.ann.<annotator>" as the
 * ANNOTATION RECORD {ui_version, annotator, exported_at, samples: {sample_id: {...}}}.
 *
 * Offline batches (DESIGN.md §13): when the export carries batch numbers the list offers a batch
 * filter and a "Load videos folder" button. The chosen folder (File System Access API, handle kept
 * in IndexedDB; fallback <input webkitdirectory>) is indexed by file stem and a sample's video is
 * played from the matching local file via URL.createObjectURL - nothing is uploaded.
 */
"use strict";

// ====================================================================== 1. pure helpers
const UI_VERSION = 1;
const ANNOTATOR_KEY = "ophbench.annotator";
const THEME_KEY = "ophbench.theme";
const PREFS_KEY = "ophbench.prefs";

const CORRECTNESS_OPTIONS = [
  ["correct", "Correct"],
  ["partial", "Partially correct"],
  ["incorrect", "Incorrect"],
  ["cannot_verify", "Cannot verify"],
];
const CORRECTNESS_VALUES = CORRECTNESS_OPTIONS.map((o) => o[0]);
const RATING_OPTIONS = [
  ["relevance", "Clinical relevance"],
  ["difficulty", "Difficulty"],
  ["agentic", "Agentic / multi-step"],
  ["clarity", "Clarity"],
];
const RATING_KEYS = RATING_OPTIONS.map((o) => o[0]);
const FLAG_OPTIONS = [
  ["video_quality", "Video quality issue"],
  ["metadata_wrong", "Metadata wrong"],
  ["not_suitable", "Not suitable"],
  ["duplicate", "Duplicate"],
];
const STATUS_LABELS = { todo: "To do", in_progress: "In progress", done: "Done" };
const STATUS_GLYPH = { todo: "○", in_progress: "◔", done: "●" };

function pad2(n) {
  return (n < 10 ? "0" : "") + n;
}

/** Seconds -> "MM:SS" or "H:MM:SS" (tenths optional). */
function fmtTime(sec, tenths) {
  if (sec === null || sec === undefined || !isFinite(Number(sec))) return "–:––";
  const t = Math.max(0, Number(sec));
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const s = t - h * 3600 - m * 60;
  const ss = tenths ? (s < 10 ? "0" : "") + (Math.floor(s * 10) / 10).toFixed(1) : pad2(Math.floor(s));
  return (h ? h + ":" + pad2(m) : pad2(m)) + ":" + ss;
}

function fmtRange(a, b) {
  return fmtTime(a) + "–" + fmtTime(b);
}

/** Seconds -> "12 min 03 s" / "5.5 s" / "?". */
function fmtDuration(sec) {
  if (sec === null || sec === undefined || !isFinite(Number(sec))) return "?";
  const t = Number(sec);
  if (t < 60) return t.toFixed(1) + " s";
  const m = Math.floor(t / 60);
  const s = Math.round(t - m * 60);
  return m + " min " + pad2(s) + " s";
}

function prettyLabel(s) {
  return String(s === null || s === undefined ? "" : s).replace(/_/g, " ");
}

/** "#/s/<id>" -> {view:"sample", id}; "#/help" -> {view:"help"}; anything else -> {view:"list"}. */
function parseHash(hash) {
  const h = String(hash || "").replace(/^#/, "");
  const m = /^\/s\/(.+)$/.exec(h);
  if (m) {
    let id = m[1];
    try {
      id = decodeURIComponent(id);
    } catch (e) {
      /* keep raw */
    }
    return { view: "sample", id };
  }
  if (h === "/help") return { view: "help" };
  return { view: "list" };
}

function sampleHash(id) {
  return "#/s/" + encodeURIComponent(id);
}

/** Join MEDIA_BASE_URL and a relative media path; absolute URLs pass through. */
function joinUrl(base, rel) {
  if (!rel) return null;
  if (/^(?:[a-z]+:)?\/\//i.test(rel) || /^(?:data|blob):/i.test(rel)) return rel;
  if (!base) return rel;
  return String(base).replace(/\/+$/, "") + "/" + String(rel).replace(/^\/+/, "");
}

function storageKey(annotator) {
  return "ophbench.ann." + annotator;
}

function nowIso() {
  return new Date().toISOString();
}

/** Normalise an annotator id: keep [A-Za-z0-9_.-], max 64 chars; "" when nothing is left. */
function cleanAnnotator(s) {
  return String(s || "")
    .trim()
    .replace(/[^A-Za-z0-9_.-]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 64);
}

/** Canonical spelling of an annotator id: the known id that matches case-insensitively, else the cleaned input. */
function canonicalAnnotator(id, known) {
  const clean = cleanAnnotator(id);
  if (!clean) return "";
  const lower = clean.toLowerCase();
  for (const k of known || []) if (String(k).toLowerCase() === lower) return String(k);
  return clean;
}

function toRating(v) {
  if (v === null || v === undefined || v === "") return null;
  const n = Number(v);
  return Number.isInteger(n) && n >= 1 && n <= 5 ? n : null;
}

function emptyQuestionAnn() {
  return {
    selected: false,
    correctness: "",
    relevance: null,
    difficulty: null,
    agentic: null,
    clarity: null,
    question_edit: "",
    answer_edit: "",
    edited: false,
    comment: "",
  };
}

function normalizeQuestionAnn(q) {
  const src = q && typeof q === "object" ? q : {};
  const out = emptyQuestionAnn();
  out.selected = src.selected === true || src.selected === "true" || src.selected === "TRUE";
  out.correctness = CORRECTNESS_VALUES.includes(src.correctness) ? src.correctness : "";
  for (const k of RATING_KEYS) out[k] = toRating(src[k]);
  out.question_edit = typeof src.question_edit === "string" ? src.question_edit : "";
  out.answer_edit = typeof src.answer_edit === "string" ? src.answer_edit : "";
  out.comment = typeof src.comment === "string" ? src.comment : "";
  out.edited = src.edited === true || out.question_edit !== "" || out.answer_edit !== "";
  return out;
}

function emptySampleAnn() {
  return { status: "in_progress", updated_at: "", batch: null, flags: [], comment: "", questions: {} };
}

function normalizeSampleAnn(s) {
  const src = s && typeof s === "object" ? s : {};
  const out = emptySampleAnn();
  out.status = src.status === "done" ? "done" : "in_progress";
  out.updated_at = typeof src.updated_at === "string" ? src.updated_at : "";
  out.batch = toBatch(src.batch);
  out.flags = Array.isArray(src.flags) ? src.flags.filter((f) => typeof f === "string" && f) : [];
  out.comment = typeof src.comment === "string" ? src.comment : "";
  const qs = src.questions && typeof src.questions === "object" && !Array.isArray(src.questions) ? src.questions : {};
  for (const qid of Object.keys(qs)) out.questions[qid] = normalizeQuestionAnn(qs[qid]);
  return out;
}

function emptyRecord(annotator) {
  return { ui_version: UI_VERSION, annotator: annotator || "", exported_at: "", samples: {} };
}

/** Coerce anything into a valid ANNOTATION RECORD (unknown keys dropped, types fixed). */
function normalizeRecord(obj, annotator) {
  const src = obj && typeof obj === "object" && !Array.isArray(obj) ? obj : {};
  const rec = emptyRecord(typeof src.annotator === "string" && src.annotator ? src.annotator : annotator);
  rec.exported_at = typeof src.exported_at === "string" ? src.exported_at : "";
  const samples = src.samples && typeof src.samples === "object" && !Array.isArray(src.samples) ? src.samples : {};
  for (const sid of Object.keys(samples)) rec.samples[sid] = normalizeSampleAnn(samples[sid]);
  return rec;
}

/** Shape errors for an imported file; [] when acceptable. */
function validateRecord(obj) {
  const errors = [];
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) {
    errors.push("the file is not a JSON object");
    return errors;
  }
  if (obj.ui_version !== undefined && Number(obj.ui_version) !== UI_VERSION) {
    errors.push("unsupported ui_version " + String(obj.ui_version));
  }
  if (!obj.samples || typeof obj.samples !== "object" || Array.isArray(obj.samples)) {
    errors.push("missing 'samples' object");
  }
  if (obj.annotator !== undefined && typeof obj.annotator !== "string") errors.push("'annotator' must be a string");
  return errors;
}

/** Compare two ISO timestamps (missing/invalid -> epoch 0). */
function cmpTime(a, b) {
  const ta = Date.parse(a || "") || 0;
  const tb = Date.parse(b || "") || 0;
  return ta - tb;
}

/** Merge `incoming` into `local`: per sample, the newer updated_at wins (ties keep local). */
function mergeRecords(local, incoming) {
  const base = normalizeRecord(local, local && local.annotator);
  const inc = normalizeRecord(incoming, base.annotator);
  const stats = { added: 0, updated: 0, skipped: 0 };
  for (const sid of Object.keys(inc.samples)) {
    const cur = base.samples[sid];
    const s = inc.samples[sid];
    if (!cur) {
      base.samples[sid] = s;
      stats.added++;
    } else if (cmpTime(s.updated_at, cur.updated_at) > 0) {
      base.samples[sid] = s;
      stats.updated++;
    } else {
      stats.skipped++;
    }
  }
  return { record: base, stats };
}

function countSelected(sampleAnn) {
  const qs = sampleAnn && sampleAnn.questions ? sampleAnn.questions : {};
  return Object.keys(qs).filter((qid) => qs[qid] && qs[qid].selected).length;
}

/** Enforce SELECT_MIN..SELECT_MAX; returns {ok, n, message}. */
function selectionCheck(sampleAnn, min, max) {
  const n = countSelected(sampleAnn);
  const plural = (k) => (k === 1 ? "question" : "questions");
  if (n < min) return { ok: false, n, message: "Select at least " + min + " " + plural(min) + " for the benchmark (" + n + " selected)" };
  if (n > max) return { ok: false, n, message: "Select at most " + max + " " + plural(max) + " for the benchmark (" + n + " selected)" };
  return { ok: true, n, message: "" };
}

/** "todo" when the sample was never touched, else the stored status. */
function sampleStatus(sampleAnn) {
  if (!sampleAnn) return "todo";
  return sampleAnn.status === "done" ? "done" : "in_progress";
}

function progressFor(ids, record) {
  const out = { total: ids.length, done: 0, in_progress: 0, todo: 0, pct: 0 };
  const samples = record && record.samples ? record.samples : {};
  for (const id of ids) out[sampleStatus(samples[id])]++;
  out.pct = out.total ? Math.round((100 * out.done) / out.total) : 0;
  return out;
}

/** Which samples an annotator sees: everything when there are no assignments (or the annotator is unknown). */
function visibleSampleIds(index, assignments, annotator) {
  const all = (index || []).map((r) => r.sample_id);
  const annos = assignments && Array.isArray(assignments.annotators) ? assignments.annotators : [];
  if (!annos.length) return { ids: all, scope: "all" };
  const table = assignments.by_annotator && typeof assignments.by_annotator === "object" ? assignments.by_annotator : {};
  let by = table[annotator];
  if (!Array.isArray(by)) {
    const lower = String(annotator || "").toLowerCase();
    const key = Object.keys(table).find((k) => k.toLowerCase() === lower);
    if (key) by = table[key];
  }
  if (!Array.isArray(by)) return { ids: all, scope: "unassigned" };
  const set = new Set(by);
  return { ids: all.filter((id) => set.has(id)), scope: "assigned" };
}

/** Apply the list filters {dataset, status, query}. */
function filterRows(rows, filters, record) {
  const f = filters || {};
  const q = String(f.query || "").trim().toLowerCase();
  const samples = record && record.samples ? record.samples : {};
  return (rows || []).filter((r) => {
    if (f.dataset && r.dataset !== f.dataset) return false;
    if (f.batch !== undefined && f.batch !== null && f.batch !== "" && String(r.batch) !== String(f.batch)) return false;
    const ann = samples[r.sample_id];
    const st = sampleStatus(ann);
    if (f.status === "todo" || f.status === "in_progress" || f.status === "done") {
      if (st !== f.status) return false;
    } else if (f.status === "flagged") {
      if (!ann || !ann.flags || !ann.flags.length) return false;
    } else if (f.status === "issues") {
      if (!r.has_issues) return false;
    }
    if (q) {
      const hay = [r.sample_id, r.dataset, r.dataset_title, r.procedure, r.procedure_category].join(" ").toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
}

/** Next id after currentId (wrapping) that is not done; null when everything is done. */
function nextUnfinished(ids, currentId, record) {
  const n = ids.length;
  if (!n) return null;
  const samples = record && record.samples ? record.samples : {};
  const start = ids.indexOf(currentId);
  for (let k = 1; k <= n; k++) {
    const id = ids[(((start + k) % n) + n) % n];
    if (id !== currentId && sampleStatus(samples[id]) !== "done") return id;
  }
  return null;
}

/** Deterministic colour slots (1-8) by first appearance; labels beyond 8 reuse hues with a hatch texture. */
function assignSegmentColors(segments) {
  const map = new Map();
  const ordered = (segments || []).slice().sort((a, b) => Number(a.start_s) - Number(b.start_s));
  let i = 0;
  for (const s of ordered) {
    const key = String(s.label);
    if (map.has(key)) continue;
    map.set(key, { slot: (i % 8) + 1, hatch: i >= 8 });
    i++;
  }
  return map;
}

/** Store an edited text: "" when it equals the original; recompute `edited`. */
function applyTextEdit(qAnn, field, value, original) {
  qAnn[field] = value === original ? "" : value;
  qAnn.edited = qAnn.question_edit !== "" || qAnn.answer_edit !== "";
  return qAnn;
}

/** Repair records that stored the unchanged original text as an edit. */
function reconcileEdits(qAnn, question) {
  if (question) {
    if (qAnn.question_edit === question.question) qAnn.question_edit = "";
    if (qAnn.answer_edit === question.answer) qAnn.answer_edit = "";
  }
  qAnn.edited = qAnn.question_edit !== "" || qAnn.answer_edit !== "";
  return qAnn;
}

function buildExportPayload(record) {
  const out = JSON.parse(JSON.stringify(normalizeRecord(record, record && record.annotator)));
  out.ui_version = UI_VERSION;
  out.exported_at = nowIso();
  return out;
}

function exportFilename(annotator, date) {
  const d = date || new Date();
  const stamp =
    d.getFullYear() + pad2(d.getMonth() + 1) + pad2(d.getDate()) + "_" + pad2(d.getHours()) + pad2(d.getMinutes());
  return "ophbench_annotations_" + (annotator || "anonymous") + "_" + stamp + ".json";
}

/** meta.json select_min/select_max win; config.js SELECT_MIN/MAX are fallbacks; defaults 1..2. */
function resolveSelectLimits(cfg, meta) {
  const num = (v) => (v === null || v === undefined || v === "" ? null : Number.isFinite(Number(v)) ? Number(v) : null);
  let min = num(meta && meta.select_min);
  if (min === null) min = num(cfg && cfg.SELECT_MIN);
  if (min === null) min = 1;
  let max = num(meta && meta.select_max);
  if (max === null) max = num(cfg && cfg.SELECT_MAX);
  if (max === null) max = 2;
  min = Math.max(0, Math.floor(min));
  max = Math.max(min, Math.floor(max));
  return { min, max };
}

function selectRuleText(limits) {
  if (limits.min === limits.max) return "exactly " + limits.min;
  return limits.min + "–" + limits.max;
}

// ---- offline batches (DESIGN.md §13): local video folder + batch numbers
const VIDEO_EXT_RE = /\.(mp4|m4v|mov|mkv|webm)$/i;

/** "dir/cataract101__case_269.mp4" -> "cataract101__case_269" (one extension stripped). */
function fileStem(name) {
  const base = String(name || "").split(/[\\/]/).pop();
  const i = base.lastIndexOf(".");
  return i > 0 ? base.slice(0, i) : base;
}

/** stem -> {name, file|null, handle|null} for entries {name, file?, handle?}; non-video names are ignored. */
function indexLocalFiles(entries) {
  const map = new Map();
  for (const e of entries || []) {
    const name = e && (e.name || (e.file && e.file.name));
    if (!name || !VIDEO_EXT_RE.test(name)) continue;
    const stem = fileStem(name);
    if (!map.has(stem)) map.set(stem, { name, file: e.file || null, handle: e.handle || null });
  }
  return map;
}

/** File stems under which a sample's video may be stored in the shared folder (video.local_file, sample_id). */
function localCandidates(sample) {
  const out = [];
  const v = (sample && sample.video) || {};
  if (v.local_file) out.push(fileStem(v.local_file));
  if (sample && sample.sample_id && !out.includes(sample.sample_id)) out.push(sample.sample_id);
  return out;
}

/** Entry of the local file index matching a sample (or an index row), or null. */
function findLocalEntry(sample, map) {
  if (!map || !map.size) return null;
  for (const stem of localCandidates(sample)) if (map.has(stem)) return map.get(stem);
  return null;
}

function toBatch(v) {
  if (v === null || v === undefined || v === "") return null;
  const n = Number(v);
  return Number.isInteger(n) && n >= 1 ? n : null;
}

/** Distinct batch numbers of the index rows (ascending); [] when the export carries no batches. */
function batchList(index) {
  const set = new Set();
  for (const r of index || []) {
    const b = toBatch(r && r.batch);
    if (b !== null) set.add(b);
  }
  return [...set].sort((a, b) => a - b);
}

const OphBenchPure = {
  UI_VERSION,
  CORRECTNESS_OPTIONS,
  RATING_OPTIONS,
  FLAG_OPTIONS,
  fmtTime,
  fmtRange,
  fmtDuration,
  prettyLabel,
  parseHash,
  sampleHash,
  joinUrl,
  storageKey,
  cleanAnnotator,
  canonicalAnnotator,
  toRating,
  emptyQuestionAnn,
  emptySampleAnn,
  emptyRecord,
  normalizeQuestionAnn,
  normalizeSampleAnn,
  normalizeRecord,
  validateRecord,
  cmpTime,
  mergeRecords,
  countSelected,
  selectionCheck,
  sampleStatus,
  progressFor,
  visibleSampleIds,
  filterRows,
  nextUnfinished,
  assignSegmentColors,
  applyTextEdit,
  reconcileEdits,
  buildExportPayload,
  exportFilename,
  resolveSelectLimits,
  selectRuleText,
  fileStem,
  indexLocalFiles,
  localCandidates,
  findLocalEntry,
  toBatch,
  batchList,
};

if (typeof module !== "undefined" && module.exports) module.exports = OphBenchPure;
if (typeof window !== "undefined") window.OphBenchPure = OphBenchPure;

// ====================================================================== 2. browser application
if (typeof window !== "undefined" && typeof document !== "undefined") {
  (function app() {
    const CONFIG = Object.assign(
      { MEDIA_BASE_URL: "", SUBMIT_URL: "", SELECT_MIN: 1, SELECT_MAX: 2 },
      window.OPHBENCH_CONFIG || {}
    );

    const state = {
      meta: null,
      index: [],
      assignments: null,
      annotator: "",
      record: null,
      visible: { ids: [], scope: "all" },
      filters: { dataset: "", status: "", query: "", batch: "" },
      filteredIds: [],
      localFiles: new Map(), // stem -> {name, file, handle} of the shared videos folder (never uploaded)
      localSource: null, // {kind: picker|input|reconnect, label, persistent}
      dirHandle: null,
      objectUrl: null,
      sampleCache: new Map(),
      current: null,
      currentSample: null,
      limits: { min: 1, max: 2 },
      loadError: null,
      prefs: { thumbs: false, auto_advance: true },
      cardRefs: {},
      saveTimer: null,
      filtersBound: false,
    };

    const $ = (id) => document.getElementById(id);

    // ---------------------------------------------------------------- tiny DOM helper
    function h(tag, attrs) {
      const el = document.createElement(tag);
      if (attrs) {
        for (const k of Object.keys(attrs)) {
          const v = attrs[k];
          if (v === null || v === undefined || v === false) continue;
          if (k === "class") el.className = v;
          else if (k === "text") el.textContent = v;
          else if (k === "value") el.value = v;
          else if (k === "checked") el.checked = !!v;
          else if (k === "dataset") Object.assign(el.dataset, v);
          else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
          else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
          else if (v === true) el.setAttribute(k, "");
          else el.setAttribute(k, String(v));
        }
      }
      for (let i = 2; i < arguments.length; i++) appendChildren(el, arguments[i]);
      return el;
    }

    function appendChildren(el, c) {
      if (c === null || c === undefined || c === false) return;
      if (Array.isArray(c)) {
        for (const x of c) appendChildren(el, x);
        return;
      }
      el.appendChild(c.nodeType ? c : document.createTextNode(String(c)));
    }

    function clear(el) {
      while (el.firstChild) el.removeChild(el.firstChild);
      return el;
    }

    function setText(id, text) {
      const el = $(id);
      if (el) el.textContent = text;
    }

    // ---------------------------------------------------------------- toast & notices
    let toastTimer = null;
    function toast(msg, isError) {
      const el = $("toast");
      el.textContent = msg;
      el.classList.toggle("error", !!isError);
      el.classList.add("show");
      clearTimeout(toastTimer);
      toastTimer = setTimeout(() => el.classList.remove("show"), isError ? 5000 : 2800);
    }

    function notice(kind, text, detail) {
      const el = h("div", { class: "notice " + kind }, text);
      if (detail) el.appendChild(h("pre", {}, detail));
      return el;
    }

    // ---------------------------------------------------------------- preferences & theme
    function loadPrefs() {
      try {
        Object.assign(state.prefs, JSON.parse(localStorage.getItem(PREFS_KEY) || "{}"));
      } catch (e) {
        /* ignore */
      }
    }

    function savePrefs(patch) {
      Object.assign(state.prefs, patch || {});
      try {
        localStorage.setItem(PREFS_KEY, JSON.stringify(state.prefs));
      } catch (e) {
        /* ignore */
      }
    }

    function applyTheme(theme) {
      const t = theme === "light" || theme === "dark" ? theme : "auto";
      if (t === "auto") document.documentElement.removeAttribute("data-theme");
      else document.documentElement.setAttribute("data-theme", t);
      try {
        localStorage.setItem(THEME_KEY, t);
      } catch (e) {
        /* ignore */
      }
      const btn = $("btn-theme");
      if (btn) btn.textContent = "Theme: " + t;
    }

    function cycleTheme() {
      const cur = localStorage.getItem(THEME_KEY) || "auto";
      const order = ["auto", "light", "dark"];
      applyTheme(order[(order.indexOf(cur) + 1) % order.length]);
    }

    // ---------------------------------------------------------------- data loading
    async function fetchJson(url, optional) {
      let res;
      try {
        res = await fetch(url, { cache: "no-cache" });
      } catch (e) {
        if (optional) return null;
        throw new Error(url + ": " + (e && e.message ? e.message : "network error"));
      }
      if (!res.ok) {
        if (optional) return null;
        throw new Error(url + ": HTTP " + res.status);
      }
      return res.json();
    }

    async function loadData() {
      const [meta, index, assignments] = await Promise.all([
        fetchJson("data/meta.json", true),
        fetchJson("data/index.json", false),
        fetchJson("data/assignments.json", true),
      ]);
      state.meta = meta || {};
      state.index = Array.isArray(index) ? index : [];
      if (assignments && typeof assignments === "object") {
        state.assignments = assignments;
      } else {
        // Fallback: derive assignments from index rows' assigned_to.
        const by = {};
        for (const r of state.index) for (const a of r.assigned_to || []) (by[a] = by[a] || []).push(r.sample_id);
        state.assignments = { annotators: Object.keys(by).sort(), by_annotator: by, overlap: [] };
      }
      const titles = (state.meta && state.meta.dataset_meta) || {};
      for (const r of state.index) {
        if (!r.dataset_title) r.dataset_title = (titles[r.dataset] && titles[r.dataset].title) || r.dataset;
      }
    }

    function showLoadError(err) {
      state.loadError = err;
      const el = $("load-error");
      clear(el);
      el.hidden = false;
      el.appendChild(h("strong", {}, "Could not load the sample index. "));
      el.appendChild(
        document.createTextNode(
          "Run `./ophbench export-ui` to generate ui/data/, and open this page over HTTP (python3 scripts/serve_ui.py) — browsers block fetch() for file:// pages. Error: " +
            (err && err.message ? err.message : String(err))
        )
      );
      $("progress-card").hidden = true;
      $("list-card").hidden = true;
    }

    // ---------------------------------------------------------------- local videos folder (offline batches)
    const IDB_NAME = "ophbench";
    const IDB_STORE = "handles";
    const IDB_KEY = "videos_dir";

    function idbOpen() {
      return new Promise((resolve) => {
        if (!window.indexedDB) return resolve(null);
        let req;
        try {
          req = window.indexedDB.open(IDB_NAME, 1);
        } catch (e) {
          return resolve(null);
        }
        req.onupgradeneeded = () => {
          try {
            req.result.createObjectStore(IDB_STORE);
          } catch (e) {
            /* exists */
          }
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => resolve(null);
        req.onblocked = () => resolve(null);
      });
    }

    async function idbGet(key) {
      const db = await idbOpen();
      if (!db) return null;
      return new Promise((resolve) => {
        try {
          const rq = db.transaction(IDB_STORE, "readonly").objectStore(IDB_STORE).get(key);
          rq.onsuccess = () => resolve(rq.result || null);
          rq.onerror = () => resolve(null);
        } catch (e) {
          resolve(null);
        }
      });
    }

    async function idbSet(key, value) {
      const db = await idbOpen();
      if (!db) return false;
      return new Promise((resolve) => {
        try {
          const rq = db.transaction(IDB_STORE, "readwrite").objectStore(IDB_STORE).put(value, key);
          rq.onsuccess = () => resolve(true);
          rq.onerror = () => resolve(false);
        } catch (e) {
          resolve(false);
        }
      });
    }

    /** List video files of a directory handle (one level of sub-folders, e.g. a parent holding batch_001/). */
    async function indexDirectoryHandle(dir, depth) {
      const entries = [];
      try {
        for await (const [name, handle] of dir.entries()) {
          if (handle.kind === "file") entries.push({ name, handle });
          else if (handle.kind === "directory" && depth > 0) entries.push(...(await indexDirectoryHandle(handle, depth - 1)));
        }
      } catch (e) {
        /* permission revoked or unreadable entry: keep what we have */
      }
      return entries;
    }

    function setLocalFiles(map, source) {
      state.localFiles = map;
      state.localSource = source;
      renderVideosStatus();
      if (state.current && !$("view-sample").hidden) openSample(state.current); // re-render with the local video
    }

    async function pickVideosFolder() {
      if (typeof window.showDirectoryPicker === "function") {
        let dir;
        try {
          dir = await window.showDirectoryPicker({ mode: "read" });
        } catch (e) {
          if (e && e.name === "AbortError") return;
          toast("Could not open the folder picker: " + (e && e.message ? e.message : String(e)), true);
          return;
        }
        const entries = await indexDirectoryHandle(dir, 1);
        state.dirHandle = dir;
        try {
          await idbSet(IDB_KEY, dir);
        } catch (e) {
          /* handle not storable: the folder must be re-selected next session */
        }
        setLocalFiles(indexLocalFiles(entries), { kind: "picker", label: dir.name || "folder", persistent: true });
        toast("Videos folder loaded: " + state.localFiles.size + " video file(s) found");
        return;
      }
      const input = $("file-videos");
      if (input) input.click(); // fallback: <input webkitdirectory>
    }

    function onVideosInput(files) {
      const entries = [];
      let folder = "";
      for (const f of files || []) {
        entries.push({ name: f.name, file: f });
        if (!folder && f.webkitRelativePath) folder = String(f.webkitRelativePath).split("/")[0];
      }
      setLocalFiles(indexLocalFiles(entries), { kind: "input", label: folder || "selected folder", persistent: false });
      toast("Videos folder loaded: " + state.localFiles.size + " video file(s). This browser forgets the folder on reload; select it again next session.");
    }

    /** On start-up: reuse the folder handle of a previous session when the browser still grants access. */
    async function restoreVideosFolder() {
      if (typeof window.showDirectoryPicker !== "function") return;
      let dir = null;
      try {
        dir = await idbGet(IDB_KEY);
      } catch (e) {
        dir = null;
      }
      if (!dir || typeof dir.queryPermission !== "function") return;
      let perm = "prompt";
      try {
        perm = await dir.queryPermission({ mode: "read" });
      } catch (e) {
        return;
      }
      state.dirHandle = dir;
      if (perm === "granted") {
        const entries = await indexDirectoryHandle(dir, 1);
        setLocalFiles(indexLocalFiles(entries), { kind: "picker", label: dir.name || "folder", persistent: true });
      } else {
        state.localSource = { kind: "reconnect", label: dir.name || "folder", persistent: true };
        renderVideosStatus();
      }
    }

    async function reconnectVideosFolder() {
      const dir = state.dirHandle;
      if (!dir || typeof dir.requestPermission !== "function") return pickVideosFolder();
      let perm = "denied";
      try {
        perm = await dir.requestPermission({ mode: "read" });
      } catch (e) {
        perm = "denied";
      }
      if (perm !== "granted") {
        toast("Permission to read the videos folder was not granted; choose the folder again", true);
        return pickVideosFolder();
      }
      const entries = await indexDirectoryHandle(dir, 1);
      setLocalFiles(indexLocalFiles(entries), { kind: "picker", label: dir.name || "folder", persistent: true });
    }

    /** "N of M videos of this batch found" / hints; also relabels the button for the reconnect case. */
    function renderVideosStatus() {
      const el = $("videos-status");
      const btn = $("btn-videos");
      const bar = $("videos-bar");
      if (!el) return;
      const offline = batchList(state.index).length > 0 || !!(state.meta && state.meta.offline_media);
      if (bar) bar.hidden = !offline && !state.localFiles.size;
      const src = state.localSource;
      if (btn) btn.textContent = src && src.kind === "reconnect" ? "Reconnect videos folder" : src && src.kind !== "reconnect" ? "Change videos folder" : "Load videos folder";
      if (src && src.kind === "reconnect") {
        el.textContent = "The videos folder “" + src.label + "” from your previous session needs permission again — press Reconnect.";
        return;
      }
      if (!state.localFiles.size) {
        el.textContent =
          "No videos folder loaded. Choose the shared batch folder to play the videos from your own computer (files are read locally, nothing is uploaded); until then the contact sheet is shown.";
        return;
      }
      const visible = new Set(state.visible.ids);
      let rows = state.index.filter((r) => visible.has(r.sample_id));
      let scope = "your list";
      if (state.filters.batch) {
        rows = rows.filter((r) => String(r.batch) === String(state.filters.batch));
        scope = "batch " + state.filters.batch;
      }
      const found = rows.filter((r) => findLocalEntry({ sample_id: r.sample_id, video: { local_file: r.sample_id + ".mp4" } }, state.localFiles)).length;
      el.textContent =
        "Videos folder “" + src.label + "”: " + found + " of " + rows.length + " videos of " + scope + " found" +
        (src.persistent ? "." : " (re-select the folder after reloading the page).");
    }

    async function resolveLocalVideo(sample) {
      const entry = findLocalEntry(sample, state.localFiles);
      if (!entry) return null;
      try {
        const file = entry.file || (entry.handle && typeof entry.handle.getFile === "function" ? await entry.handle.getFile() : null);
        if (!file) return null;
        return { url: URL.createObjectURL(file), name: entry.name, size: file.size };
      } catch (e) {
        return null;
      }
    }

    function revokeObjectUrl() {
      if (state.objectUrl) {
        try {
          URL.revokeObjectURL(state.objectUrl);
        } catch (e) {
          /* ignore */
        }
        state.objectUrl = null;
      }
    }

    // ---------------------------------------------------------------- annotator & record
    function loadRecord() {
      let raw = null;
      try {
        raw = JSON.parse(localStorage.getItem(storageKey(state.annotator)) || "null");
      } catch (e) {
        raw = null;
      }
      state.record = normalizeRecord(raw, state.annotator);
      state.record.annotator = state.annotator;
      state.visible = visibleSampleIds(state.index, state.assignments, state.annotator);
    }

    function saveRecord() {
      clearTimeout(state.saveTimer);
      state.saveTimer = null;
      if (!state.annotator || !state.record) return;
      try {
        localStorage.setItem(storageKey(state.annotator), JSON.stringify(state.record));
      } catch (e) {
        toast("Could not save: browser storage is full or blocked. Export JSON now to keep your work!", true);
      }
    }

    function scheduleSave() {
      clearTimeout(state.saveTimer);
      state.saveTimer = setTimeout(saveRecord, 300);
    }

    function setAnnotator(id) {
      state.annotator = id;
      try {
        localStorage.setItem(ANNOTATOR_KEY, id);
      } catch (e) {
        /* ignore */
      }
      loadRecord();
      renderAnnotator();
      renderList();
    }

    function knownAnnotators() {
      const out = new Set();
      for (const a of (state.assignments && state.assignments.annotators) || []) out.add(String(a));
      for (const a of Object.keys((state.assignments && state.assignments.by_annotator) || {})) out.add(a);
      for (const a of (state.meta && state.meta.annotators) || []) out.add(String(a));
      return [...out].sort();
    }

    function renderAnnotator() {
      const have = !!state.annotator;
      $("annotator-form").hidden = have;
      $("annotator-current").hidden = !have;
      $("annotator-chip").hidden = !have;
      if (have) {
        setText("annotator-name", state.annotator);
        setText("annotator-chip-name", state.annotator);
        const scope = state.visible.scope;
        setText(
          "annotator-scope",
          scope === "assigned"
            ? "(" + state.visible.ids.length + " assigned samples)"
            : scope === "unassigned"
            ? "(no assignment found for this ID — showing all " + state.visible.ids.length + " samples)"
            : "(" + state.visible.ids.length + " samples)"
        );
      }
      const dl = $("annotator-list");
      clear(dl);
      for (const a of knownAnnotators()) dl.appendChild(h("option", { value: a }));
    }

    function touchSample(sid) {
      let s = state.record.samples[sid];
      if (!s) {
        s = emptySampleAnn();
        state.record.samples[sid] = s;
      }
      if (s.batch === null || s.batch === undefined) {
        const row = state.index.find((r) => r.sample_id === sid);
        if (row && toBatch(row.batch) !== null) s.batch = toBatch(row.batch);
      }
      s.updated_at = nowIso();
      return s;
    }

    function ensureQuestionAnn(sid, qid) {
      const s = touchSample(sid);
      if (!s.questions[qid]) s.questions[qid] = emptyQuestionAnn();
      return s.questions[qid];
    }

    /** After any mutation: keep "done" only while the selection rule still holds, then save + refresh. */
    function afterChange(sid, immediate) {
      const s = state.record.samples[sid];
      if (s && s.status === "done") {
        const chk = selectionCheck(s, state.limits.min, state.limits.max);
        if (!chk.ok) {
          s.status = "in_progress";
          toast("Sample reopened: " + chk.message, true);
        }
      }
      if (immediate) saveRecord();
      else scheduleSave();
      refreshStatusUI();
    }

    // ---------------------------------------------------------------- routing
    function showView(name) {
      $("view-list").hidden = name !== "list";
      $("view-sample").hidden = name !== "sample";
    }

    function route() {
      const r = parseHash(location.hash);
      if (r.view === "help") {
        openHelp();
        location.hash = "#/";
        return;
      }
      if (r.view === "sample") {
        if (state.loadError) {
          location.hash = "#/";
          return;
        }
        if (!state.annotator) {
          toast("Enter your annotator ID first");
          location.hash = "#/";
          return;
        }
        showView("sample");
        openSample(r.id);
        return;
      }
      state.current = null;
      state.currentSample = null;
      revokeObjectUrl();
      showView("list");
      renderList();
      window.scrollTo(0, 0);
    }

    // ---------------------------------------------------------------- list view
    function statusChip(st) {
      return h("span", { class: "chip status-" + st }, h("span", { class: "dot", "aria-hidden": "true" }), STATUS_LABELS[st] || st);
    }

    function populateDatasetFilter(rows) {
      const sel = $("filter-dataset");
      const current = sel.value;
      clear(sel);
      sel.appendChild(h("option", { value: "" }, "All datasets"));
      const counts = {};
      for (const r of rows) counts[r.dataset] = (counts[r.dataset] || 0) + 1;
      const titles = (state.meta && state.meta.dataset_meta) || {};
      for (const ds of Object.keys(counts).sort()) {
        const title = (titles[ds] && titles[ds].title) || ds;
        sel.appendChild(h("option", { value: ds }, title + " (" + counts[ds] + ")"));
      }
      sel.value = current;
      if (sel.value !== current) state.filters.dataset = "";
    }

    function populateBatchFilter(rows) {
      const sel = $("filter-batch");
      const wrap = $("filter-batch-label");
      if (!sel) return;
      const batches = batchList(rows);
      if (wrap) wrap.hidden = !batches.length;
      const current = sel.value;
      clear(sel);
      sel.appendChild(h("option", { value: "" }, "All batches"));
      const counts = {};
      for (const r of rows) {
        const b = toBatch(r.batch);
        if (b !== null) counts[b] = (counts[b] || 0) + 1;
      }
      for (const b of batches) sel.appendChild(h("option", { value: String(b) }, "Batch " + b + " (" + counts[b] + ")"));
      sel.value = current;
      if (sel.value !== current) state.filters.batch = "";
    }

    function renderProgress(ids, scopeLabel) {
      const p = progressFor(ids, state.record);
      const pct = (n) => (p.total ? (100 * n) / p.total : 0).toFixed(2) + "%";
      $("p-done").style.width = pct(p.done);
      $("p-progress").style.width = pct(p.in_progress);
      $("p-todo").style.width = pct(p.todo);
      $("progress-bar").setAttribute("aria-valuenow", String(p.pct));
      setText("progress-text", p.done + " of " + p.total + " complete (" + p.pct + "%)" + (scopeLabel ? " — " + scopeLabel : ""));
      const legend = clear($("progress-legend"));
      const item = (cls, label, n) =>
        h("span", {}, h("span", { class: "dot", style: { background: cls }, "aria-hidden": "true" }), label + " " + n);
      legend.appendChild(item("var(--status-done)", STATUS_GLYPH.done + " Done", p.done));
      legend.appendChild(item("var(--status-progress)", STATUS_GLYPH.in_progress + " In progress", p.in_progress));
      legend.appendChild(item("var(--surface-3)", STATUS_GLYPH.todo + " To do", p.todo));
    }

    function renderList() {
      renderAnnotator();
      if (state.loadError) return;
      const have = !!state.annotator;
      $("progress-card").hidden = !have;
      $("list-card").hidden = !have;
      if (!have) return;
      const visible = new Set(state.visible.ids);
      const rows = state.index.filter((r) => visible.has(r.sample_id));
      populateDatasetFilter(rows);
      populateBatchFilter(rows);
      $("filter-status").value = state.filters.status;
      $("filter-query").value = state.filters.query;
      $("toggle-thumbs").checked = !!state.prefs.thumbs;
      const filtered = filterRows(rows, state.filters, state.record);
      state.filteredIds = filtered.map((r) => r.sample_id);
      if (state.filters.batch) {
        renderProgress(rows.filter((r) => String(r.batch) === String(state.filters.batch)).map((r) => r.sample_id), "batch " + state.filters.batch);
      } else {
        renderProgress(state.visible.ids);
      }
      setText(
        "list-count",
        filtered.length === rows.length
          ? rows.length + " samples"
          : filtered.length + " of " + rows.length + " samples shown"
      );
      renderRows(filtered);
      updateSubmitButton();
      renderListNotices();
      renderVideosStatus();
    }

    function renderListNotices() {
      const box = clear($("list-notices"));
      if (state.visible.scope === "unassigned") {
        box.appendChild(
          notice(
            "warn",
            "No assignment list was found for annotator “" +
              state.annotator +
              "”, so all samples are shown. Check the spelling of your ID if you expected a personal list."
          )
        );
      }
      if (state.prefs.last_submit_at) {
        box.appendChild(h("p", { class: "small muted" }, "Last submitted: " + new Date(state.prefs.last_submit_at).toLocaleString()));
      }
    }

    function renderRows(rows) {
      const tbody = clear($("sample-rows"));
      $("list-empty").hidden = rows.length > 0;
      const showThumbs = !!state.prefs.thumbs;
      $("th-thumb").hidden = !showThumbs;
      const showBatch = batchList(state.index).length > 0;
      const thBatch = $("th-batch");
      if (thBatch) thBatch.hidden = !showBatch;
      const frag = document.createDocumentFragment();
      rows.forEach((r, i) => {
        const ann = state.record.samples[r.sample_id];
        const st = sampleStatus(ann);
        const tr = h("tr", {
          class: st === "done" ? "is-done" : "",
          dataset: { id: r.sample_id },
          onclick: (e) => {
            if (e.target.closest("a")) return;
            location.hash = sampleHash(r.sample_id);
          },
        });
        tr.appendChild(h("td", { class: "num" }, String(i + 1)));
        if (showThumbs) {
          tr.appendChild(
            h("td", {}, h("img", { class: "thumb", loading: "lazy", alt: "", src: joinUrl(CONFIG.MEDIA_BASE_URL, r.thumb) || "" }))
          );
        }
        tr.appendChild(h("td", { class: "id" }, h("a", { href: sampleHash(r.sample_id) }, r.sample_id)));
        tr.appendChild(h("td", {}, r.dataset_title || r.dataset));
        if (showBatch) tr.appendChild(h("td", { class: "num col-opt" }, toBatch(r.batch) === null ? "" : String(r.batch)));
        tr.appendChild(h("td", { class: "col-opt" }, r.procedure || ""));
        tr.appendChild(h("td", { class: "col-opt" }, prettyLabel(r.procedure_category)));
        tr.appendChild(h("td", { class: "num" }, fmtDuration(r.duration_s)));
        tr.appendChild(h("td", { class: "num col-opt" }, r.n_questions === undefined || r.n_questions === null ? "" : String(r.n_questions)));
        const stCell = h("td", { class: "nowrap" }, statusChip(st));
        if (r.has_issues) stCell.appendChild(h("span", { class: "badge warn", title: "Automatic validation flagged at least one question" }, " ⚠ issues"));
        if (ann && ann.flags && ann.flags.length) stCell.appendChild(h("span", { class: "badge", title: ann.flags.join(", ") }, " ⚑ " + ann.flags.length));
        tr.appendChild(stCell);
        tr.appendChild(h("td", { class: "num" }, ann ? String(countSelected(ann)) : "–"));
        frag.appendChild(tr);
      });
      tbody.appendChild(frag);
    }

    function updateSubmitButton() {
      const btn = $("btn-submit");
      if (!CONFIG.SUBMIT_URL) {
        btn.disabled = true;
        btn.title = "Submit is not configured (SUBMIT_URL in config.js is empty). Use Export JSON and send the file.";
      } else {
        btn.disabled = false;
        btn.title = "Send your annotations to the study server";
      }
    }

    // ---------------------------------------------------------------- export / import / submit
    function exportJson() {
      const payload = buildExportPayload(state.record);
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = h("a", { href: url, download: exportFilename(state.annotator) });
      document.body.appendChild(a);
      a.click();
      setTimeout(() => {
        URL.revokeObjectURL(url);
        a.remove();
      }, 1000);
      savePrefs({ last_export_at: nowIso() });
      toast("Exported " + Object.keys(payload.samples).length + " annotated samples");
    }

    async function importFile(file) {
      let obj;
      try {
        obj = JSON.parse(await file.text());
      } catch (e) {
        toast("Import failed: the file is not valid JSON", true);
        return;
      }
      const errs = validateRecord(obj);
      if (errs.length) {
        toast("Import failed: " + errs.join("; "), true);
        return;
      }
      if (obj.annotator && obj.annotator !== state.annotator) {
        const ok = window.confirm(
          "This file belongs to annotator “" + obj.annotator + "” but you are “" + state.annotator + "”. Merge it into your annotations anyway?"
        );
        if (!ok) return;
      }
      const merged = mergeRecords(state.record, obj);
      state.record = merged.record;
      state.record.annotator = state.annotator;
      saveRecord();
      renderList();
      toast("Imported: " + merged.stats.added + " added, " + merged.stats.updated + " updated, " + merged.stats.skipped + " unchanged (older or identical)");
    }

    async function submitAnnotations() {
      const url = CONFIG.SUBMIT_URL;
      const box = clear($("submit-result"));
      if (!url) {
        toast("Submit is not configured; use Export JSON", true);
        return;
      }
      const payload = buildExportPayload(state.record);
      const n = Object.keys(payload.samples).length;
      if (!n) {
        toast("Nothing to submit yet");
        return;
      }
      const btn = $("btn-submit");
      btn.disabled = true;
      const old = btn.textContent;
      btn.textContent = "Submitting…";
      try {
        // Google Apps Script web apps: CORS works with a text/plain body (no preflight) and a followed redirect.
        const res = await fetch(url, {
          method: "POST",
          body: JSON.stringify(payload),
          headers: { "Content-Type": "text/plain;charset=utf-8" },
          redirect: "follow",
        });
        let text = "";
        try {
          text = await res.text();
        } catch (e) {
          text = "";
        }
        const ok = res.ok || res.type === "opaque" || res.status === 0;
        if (ok) {
          savePrefs({ last_submit_at: nowIso() });
          box.appendChild(notice("success", "Submitted " + n + " annotated samples. Server response:", text || "(empty response, status " + res.status + ")"));
          toast("Submitted " + n + " samples");
        } else {
          box.appendChild(notice("error", "Submit failed (HTTP " + res.status + "). Your work is still saved in this browser — use Export JSON as a backup.", text));
          toast("Submit failed: HTTP " + res.status, true);
        }
      } catch (e) {
        box.appendChild(
          notice(
            "error",
            "Submit failed: " + (e && e.message ? e.message : String(e)) + ". Your work is still saved in this browser — use Export JSON as a backup."
          )
        );
        toast("Submit failed (network)", true);
      } finally {
        btn.textContent = old;
        updateSubmitButton();
      }
    }

    // ---------------------------------------------------------------- sample view
    function navIds() {
      if (state.current && state.filteredIds.includes(state.current)) return state.filteredIds;
      return state.visible.ids;
    }

    function goRelative(delta) {
      const ids = navIds();
      if (!ids.length) return;
      const i = ids.indexOf(state.current);
      const j = i < 0 ? 0 : (((i + delta) % ids.length) + ids.length) % ids.length;
      location.hash = sampleHash(ids[j]);
    }

    function goNextUnfinished() {
      const id = nextUnfinished(navIds(), state.current, state.record);
      if (!id) {
        toast("All samples in this list are complete");
        return;
      }
      location.hash = sampleHash(id);
    }

    async function openSample(id) {
      state.current = id;
      state.cardRefs = {};
      revokeObjectUrl();
      const row = state.index.find((r) => r.sample_id === id) || null;
      setText("sample-title", id);
      const ids = navIds();
      const pos = ids.indexOf(id);
      setText("sample-pos", pos >= 0 ? pos + 1 + " / " + ids.length : "");
      const notices = clear($("sample-notices"));
      clear($("col-left"));
      clear($("col-right"));
      refreshStatusUI();
      if (!row) notices.appendChild(notice("warn", "This sample is not in the current index (the link may be stale); you can still annotate it."));
      else if (!state.visible.ids.includes(id) && state.visible.scope === "assigned") {
        notices.appendChild(notice("warn", "This sample is not in your assignment list; your annotation will still be recorded."));
      }
      let sample = state.sampleCache.get(id);
      if (!sample) {
        $("col-left").appendChild(h("p", { class: "muted" }, "Loading sample…"));
        try {
          sample = await fetchJson("data/samples/" + encodeURIComponent(id) + ".json", false);
        } catch (e) {
          if (state.current !== id) return;
          clear($("col-left"));
          notices.appendChild(notice("error", "Could not load this sample: " + (e && e.message ? e.message : String(e))));
          return;
        }
        state.sampleCache.set(id, sample);
      }
      if (state.current !== id) return; // navigated away meanwhile
      const local = await resolveLocalVideo(sample);
      if (state.current !== id) {
        if (local) URL.revokeObjectURL(local.url);
        return;
      }
      state.objectUrl = local ? local.url : null;
      state.currentSample = sample;
      renderSample(sample, row, local);
      window.scrollTo(0, 0);
    }

    function refreshStatusUI() {
      const sid = state.current;
      const chipBox = $("sample-status-chip");
      if (!sid) return;
      const ann = state.record.samples[sid];
      const st = sampleStatus(ann);
      clear(chipBox).appendChild(statusChip(st));
      const n = countSelected(ann);
      const chk = selectionCheck(ann || emptySampleAnn(), state.limits.min, state.limits.max);
      const refs = state.cardRefs;
      if (refs._selectedCount) refs._selectedCount.textContent = n + " of " + state.limits.max + " selected";
      if (refs._complete) {
        refs._complete.hidden = st === "done";
        refs._complete.disabled = !chk.ok;
        refs._complete.title = chk.ok ? "Mark this sample as reviewed" : chk.message;
      }
      if (refs._reopen) refs._reopen.hidden = st !== "done";
      if (refs._hint) {
        refs._hint.textContent =
          st === "done"
            ? "Marked complete" + (ann && ann.updated_at ? " — last change " + new Date(ann.updated_at).toLocaleString() : "") + "."
            : chk.ok
            ? "Selection rule satisfied (" + chk.n + " selected). Press Mark complete when you are done with this sample."
            : chk.message + ".";
      }
      if (refs._bottomStatus) clear(refs._bottomStatus).appendChild(statusChip(st));
    }

    // ---- left column: video, timeline, metadata -----------------------------------------
    function renderSample(sample, row, local) {
      const left = clear($("col-left"));
      const right = clear($("col-right"));
      const player = buildPlayer(sample, local || null);
      left.appendChild(player.card);
      left.appendChild(buildMetadataPanel(sample, row));
      left.appendChild(buildDatasetPanel(sample));
      left.appendChild(buildSummaryPanel(sample));
      const sid = sample.sample_id;
      const questions = Array.isArray(sample.questions) ? sample.questions : [];
      questions.forEach((q, i) => right.appendChild(buildQuestionCard(sid, q, i, player.seekTo)));
      if (!questions.length) right.appendChild(notice("warn", "This sample has no questions."));
      right.appendChild(buildSamplePanel(sid));
      refreshStatusUI();
    }

    function buildPlayer(sample, local) {
      const v = sample.video || {};
      const segments = Array.isArray(sample.segments) ? sample.segments : [];
      const frames = Array.isArray(sample.frames) ? sample.frames : [];
      let duration = Number(v.duration_s) || 0;
      if (!duration && segments.length) duration = Math.max.apply(null, segments.map((s) => Number(s.end_s) || 0));
      const localUrl = local && local.url ? local.url : null;
      const previewUrl = localUrl || joinUrl(CONFIG.MEDIA_BASE_URL, v.preview);
      const sheetUrl = joinUrl(CONFIG.MEDIA_BASE_URL, v.contact_sheet);

      const box = h("div", { class: "video-box" });
      const noticeEl = h("div", { class: "video-notice", hidden: true });
      const readout = h("span", { class: "time-readout" }, "–:–– / " + fmtTime(duration));
      let video = null;
      let fallback = false;

      function showFallback(msg) {
        fallback = true;
        clear(box);
        if (sheetUrl) box.appendChild(h("img", { class: "fallback", src: sheetUrl, alt: "Contact sheet of sampled frames with timestamps" }));
        else box.appendChild(h("div", { class: "empty", style: { color: "#fff" } }, "No media available for this sample."));
        noticeEl.hidden = false;
        noticeEl.textContent = msg + " Showing the contact sheet instead; timestamps are printed on each frame.";
      }

      if (previewUrl) {
        video = h("video", { controls: true, preload: "metadata", playsinline: true, "aria-label": "Surgery preview video" });
        video.src = previewUrl;
        if (localUrl) {
          noticeEl.hidden = false;
          noticeEl.textContent = "Playing the local file “" + local.name + "” from your videos folder (read locally, not uploaded).";
        }
        video.addEventListener("error", () =>
          showFallback(localUrl ? "The local video file “" + local.name + "” could not be played by this browser." : "The preview video could not be loaded.")
        );
        video.addEventListener("loadedmetadata", () => {
          if (isFinite(video.duration) && duration && video.duration < duration - 2) {
            noticeEl.hidden = false;
            noticeEl.textContent =
              "Note: the preview (" + fmtTime(video.duration) + ") is shorter than the annotated timeline (" + fmtTime(duration) + "); later segments cannot be seeked.";
          }
          if (!duration && isFinite(video.duration)) {
            duration = video.duration;
            drawTimeline();
          }
        });
        video.addEventListener("timeupdate", () => {
          readout.textContent = fmtTime(video.currentTime) + " / " + fmtTime(duration || video.duration);
          if (duration) playhead.style.left = Math.min(100, (100 * video.currentTime) / duration) + "%";
        });
        box.appendChild(video);
      } else if (v.local_file || (state.meta && state.meta.offline_media)) {
        showFallback("The video is not hosted online. Load the shared videos folder (button on the list page) to play it.");
      } else {
        showFallback("No preview video was exported for this sample.");
      }

      const playhead = h("div", { class: "playhead", style: { left: "0%" }, "aria-hidden": "true" });

      function seekTo(t) {
        if (!video || fallback) {
          toast("Video not available — see the contact sheet frame nearest " + fmtTime(t));
          return;
        }
        const target = Math.max(0, Math.min(Number(t) || 0, isFinite(video.duration) ? video.duration : Number(t) || 0));
        try {
          video.currentTime = target;
          const p = video.play();
          if (p && p.catch) p.catch(() => {});
        } catch (e) {
          /* ignore */
        }
        if (window.matchMedia && window.matchMedia("(max-width: 900px)").matches) box.scrollIntoView({ behavior: "smooth", block: "start" });
      }

      // ---- timeline
      const colors = assignSegmentColors(segments.filter((s) => s.kind !== "flag"));
      const flagColors = assignSegmentColors(segments.filter((s) => s.kind === "flag"));
      const bar = h("div", { class: "timeline", role: "group", "aria-label": "Phase timeline; click to seek" });
      const flagBar = h("div", { class: "timeline flags", role: "group", "aria-label": "Flag segments", hidden: true });
      const ticks = h("div", { class: "ticks", "aria-label": "Frames shown to the model", hidden: true });
      const legend = h("div", { class: "timeline-legend" });
      const tip = $("tl-tooltip");

      function showTip(e, seg) {
        tip.hidden = false;
        clear(tip);
        tip.appendChild(h("strong", {}, seg.label));
        tip.appendChild(document.createTextNode(fmtRange(seg.start_s, seg.end_s) + " (" + fmtDuration(Number(seg.end_s) - Number(seg.start_s)) + ")"));
        let x = 12;
        let y = 14;
        if (e && typeof e.clientX === "number") {
          x = e.clientX + 12;
          y = e.clientY + 14;
        } else if (e && e.target && e.target.getBoundingClientRect) {
          const r = e.target.getBoundingClientRect();
          x = r.left;
          y = r.bottom + 6;
        }
        const w = tip.offsetWidth || 160;
        if (x + w > window.innerWidth - 8) x = Math.max(8, window.innerWidth - w - 8);
        tip.style.left = x + "px";
        tip.style.top = y + "px";
      }
      function hideTip() {
        tip.hidden = true;
      }

      function segButton(seg, cmap) {
        const dur = duration || 1;
        const start = Math.max(0, Number(seg.start_s) || 0);
        const end = Math.max(start, Number(seg.end_s) || 0);
        const leftPct = (100 * start) / dur;
        const widthPct = Math.max(0, (100 * (end - start)) / dur);
        const c = cmap.get(String(seg.label)) || { slot: 1, hatch: false };
        const btn = h("button", {
          type: "button",
          class: "seg" + (c.hatch ? " hatch" : ""),
          style: { left: "calc(" + leftPct + "% + 1px)", width: "calc(" + widthPct + "% - 2px)", background: "var(--s" + c.slot + ")" },
          title: seg.label + " — " + fmtRange(start, end),
          "aria-label": seg.label + ", " + fmtRange(start, end) + ", seek",
          onclick: (e) => {
            e.stopPropagation();
            seekTo(start);
          },
          onpointermove: (e) => showTip(e, { label: seg.label, start_s: start, end_s: end }),
          onpointerleave: hideTip,
          onfocus: (e) => showTip(e, { label: seg.label, start_s: start, end_s: end }),
          onblur: hideTip,
        });
        if (widthPct > 7) btn.textContent = seg.label;
        return btn;
      }

      function drawTimeline() {
        clear(bar);
        clear(flagBar);
        clear(ticks);
        clear(legend);
        const phases = segments.filter((s) => s.kind !== "flag");
        const flags = segments.filter((s) => s.kind === "flag");
        for (const seg of phases) bar.appendChild(segButton(seg, colors));
        bar.appendChild(playhead);
        flagBar.hidden = !flags.length;
        for (const seg of flags) flagBar.appendChild(segButton(seg, flagColors));
        ticks.hidden = !frames.length || !duration;
        for (const f of frames) {
          const t = Number(f.t_s) || 0;
          ticks.appendChild(
            h("button", {
              type: "button",
              class: "tick",
              style: { left: (100 * t) / (duration || 1) + "%" },
              title: "Frame shown to the model at " + fmtTime(t, true) + (f.label ? " — " + f.label : ""),
              "aria-label": "Seek to frame at " + fmtTime(t),
              onclick: () => seekTo(t),
            })
          );
        }
        for (const [label, c] of colors.entries()) {
          legend.appendChild(
            h("span", {}, h("span", { class: "sw" + (c.hatch ? " hatch" : ""), style: { background: "var(--s" + c.slot + ")" }, "aria-hidden": "true" }), label)
          );
        }
        for (const [label, c] of flagColors.entries()) {
          legend.appendChild(
            h("span", {}, h("span", { class: "sw" + (c.hatch ? " hatch" : ""), style: { background: "var(--s" + c.slot + ")" }, "aria-hidden": "true" }), "flag: " + label)
          );
        }
        if (!phases.length) bar.appendChild(h("span", { class: "small muted", style: { position: "absolute", left: "8px", top: "7px" } }, "No phase labels for this video"));
      }
      bar.addEventListener("click", (e) => {
        if (!duration) return;
        const r = bar.getBoundingClientRect();
        const frac = Math.max(0, Math.min(1, (e.clientX - r.left) / (r.width || 1)));
        seekTo(frac * duration);
      });
      drawTimeline();

      const card = h(
        "div",
        { class: "card" },
        box,
        noticeEl,
        h("div", { class: "row between", style: { marginTop: "0.5rem" } }, h("span", { class: "small text-2" }, "Phase timeline — click a segment or the bar to seek"), readout),
        h("div", { class: "timeline-wrap" }, bar, flagBar, ticks, legend),
        sample.timeline_note ? h("p", { class: "timeline-note" }, sample.timeline_note) : null
      );
      return { card, seekTo };
    }

    function kvRow(label, value) {
      return h("tr", {}, h("th", { scope: "row" }, label), h("td", {}, value === null || value === undefined ? "" : String(value)));
    }

    function buildMetadataPanel(sample, row) {
      const v = sample.video || {};
      const table = h("table", { class: "kv" });
      table.appendChild(kvRow("Procedure", sample.procedure || (row && row.procedure) || ""));
      table.appendChild(kvRow("Category", prettyLabel(sample.procedure_category || (row && row.procedure_category))));
      table.appendChild(kvRow("Duration", fmtDuration(v.duration_s)));
      if (v.width && v.height) table.appendChild(kvRow("Resolution", v.width + "×" + v.height + (v.fps ? " @ " + Number(v.fps).toFixed(1) + " fps" : "")));
      for (const m of Array.isArray(sample.metadata) ? sample.metadata : []) {
        if (!m || m.label === undefined) continue;
        table.appendChild(kvRow(String(m.label), m.value));
      }
      return h("details", { class: "panel", open: true }, h("summary", {}, "Metadata"), h("div", { class: "panel-body" }, table));
    }

    function buildDatasetPanel(sample) {
      return h(
        "details",
        { class: "panel" },
        h("summary", {}, "About the dataset: " + (sample.dataset_title || sample.dataset || "")),
        h(
          "div",
          { class: "panel-body stack small" },
          sample.dataset_blurb ? h("p", {}, sample.dataset_blurb) : null,
          sample.license ? h("p", {}, h("strong", {}, "Licence: "), sample.license) : null,
          sample.citation ? h("p", { class: "text-2" }, h("strong", {}, "Citation: "), sample.citation) : null
        )
      );
    }

    function buildSummaryPanel(sample) {
      if (!sample.video_summary && !sample.generator_notes) return document.createDocumentFragment();
      return h(
        "details",
        { class: "panel" },
        h("summary", {}, "Model’s video summary"),
        h(
          "div",
          { class: "panel-body stack small" },
          h("p", { class: "muted" }, "Written by the question generator from sampled frames + labels; it may be wrong — verify against the video."),
          sample.video_summary ? h("p", { class: "summary-text" }, sample.video_summary) : null,
          sample.generator_notes ? h("p", { class: "summary-text text-2" }, h("strong", {}, "Generator notes: "), sample.generator_notes) : null
        )
      );
    }

    // ---- right column: question cards ----------------------------------------------------
    function setSelected(sid, qid, on) {
      const s = state.record.samples[sid];
      const n = countSelected(s);
      const cur = s && s.questions[qid] ? !!s.questions[qid].selected : false;
      if (on && !cur && n >= state.limits.max) {
        toast("At most " + state.limits.max + " question" + (state.limits.max === 1 ? "" : "s") + " can be selected per sample; unselect one first", true);
        return false;
      }
      const q = ensureQuestionAnn(sid, qid);
      q.selected = !!on;
      const ref = state.cardRefs[qid];
      if (ref) {
        ref.checkbox.checked = q.selected;
        ref.card.classList.toggle("is-selected", q.selected);
      }
      afterChange(sid, true);
      return true;
    }

    function radioGroup(name, options, current, onChange, extraClass) {
      const row = h("div", { class: "radio-row" });
      const labels = [];
      for (const [value, text] of options) {
        const input = h("input", { type: "radio", name, value: String(value), checked: String(current) === String(value) });
        const label = h("label", { class: (extraClass ? extraClass(value) : "") + (String(current) === String(value) ? " checked" : "") }, input, text);
        input.addEventListener("change", () => {
          for (const l of labels) l.classList.remove("checked");
          label.classList.add("checked");
          onChange(value);
        });
        labels.push(label);
        row.appendChild(label);
      }
      row._clear = () => {
        for (const l of labels) {
          l.classList.remove("checked");
          l.querySelector("input").checked = false;
        }
      };
      return row;
    }

    function textField(labelText, value, original, onInput, cls) {
      const ta = h("textarea", { class: cls || "", "aria-label": labelText, value });
      const editedChip = h("span", { class: "chip edited", hidden: value === original }, "edited");
      const reset = h("button", { type: "button", class: "btn link small", hidden: value === original }, "Reset to original");
      const origDetails = h("details", { class: "inline", hidden: value === original }, h("summary", {}, "Show original"), h("p", { class: "small text-2 summary-text" }, original));
      function refresh() {
        const edited = ta.value !== original;
        ta.classList.toggle("is-edited", edited);
        editedChip.hidden = !edited;
        reset.hidden = !edited;
        origDetails.hidden = !edited;
      }
      ta.addEventListener("input", () => {
        onInput(ta.value);
        refresh();
      });
      reset.addEventListener("click", () => {
        ta.value = original;
        onInput(original);
        refresh();
      });
      refresh();
      return h("div", { class: "field" }, h("div", { class: "field-head" }, labelText, editedChip, reset), ta, origDetails);
    }

    function buildQuestionCard(sid, q, i, seekTo) {
      const qid = q.qid || "q" + (i + 1);
      const existing = state.record.samples[sid] && state.record.samples[sid].questions[qid];
      const ann = existing ? reconcileEdits(existing, q) : emptyQuestionAnn();
      const card = h("div", { class: "card qcard" + (ann.selected ? " is-selected" : ""), id: "card-" + qid });

      // header
      const checkbox = h("input", { type: "checkbox", checked: ann.selected, id: "sel-" + qid });
      checkbox.addEventListener("change", () => {
        if (!setSelected(sid, qid, checkbox.checked)) checkbox.checked = false;
      });
      const head = h(
        "div",
        { class: "qhead" },
        h("span", { class: "qid" }, "Q" + (i + 1)),
        h("span", { class: "badge cat", title: "Question category" }, prettyLabel(q.category)),
        q.difficulty ? h("span", { class: "badge diff-" + q.difficulty, title: "Difficulty claimed by the generator" }, prettyLabel(q.difficulty)) : null,
        q.answer_type ? h("span", { class: "badge", title: "Answer type" }, prettyLabel(q.answer_type)) : null,
        typeof q.confidence === "number" ? h("span", { class: "badge", title: "Generator confidence that the gold answer is correct" }, "conf " + Math.round(q.confidence * 100) + "%") : null,
        h("label", { class: "select-label", for: "sel-" + qid, title: "Keep this question in the benchmark (shortcut: " + (i + 1) + ")" }, checkbox, "Select for benchmark")
      );
      card.appendChild(head);

      if (Array.isArray(q.validation_issues) && q.validation_issues.length) {
        card.appendChild(
          h("div", { class: "validation" }, h("strong", {}, "⚠ Automatic validation issues"), h("ul", {}, q.validation_issues.map((x) => h("li", {}, String(x)))))
        );
      }

      // question text (editable)
      card.appendChild(
        textField(
          "Question",
          ann.question_edit || q.question || "",
          q.question || "",
          (val) => {
            const qa = ensureQuestionAnn(sid, qid);
            applyTextEdit(qa, "question_edit", val, q.question || "");
            afterChange(sid, false);
          },
          "question"
        )
      );
      if (Array.isArray(q.options) && q.options.length) {
        card.appendChild(h("div", { class: "field-head" }, "Options"));
        card.appendChild(h("ul", { class: "options" }, q.options.map((o) => h("li", {}, String(o)))));
      }
      card.appendChild(
        textField(
          "Gold answer",
          ann.answer_edit || q.answer || "",
          q.answer || "",
          (val) => {
            const qa = ensureQuestionAnn(sid, qid);
            applyTextEdit(qa, "answer_edit", val, q.answer || "");
            afterChange(sid, false);
          },
          "answer"
        )
      );

      // rationale + evidence
      const evidence = Array.isArray(q.evidence_timestamps) ? q.evidence_timestamps : [];
      card.appendChild(
        h(
          "details",
          { class: "panel", open: true },
          h("summary", {}, "Rationale & evidence (" + evidence.length + ")"),
          h(
            "div",
            { class: "panel-body" },
            q.answer_rationale ? h("p", { class: "rationale" }, q.answer_rationale) : h("p", { class: "muted small" }, "No rationale provided."),
            evidence.length
              ? h(
                  "ul",
                  { class: "evidence-list" },
                  evidence.map((ev) =>
                    h(
                      "li",
                      {},
                      h(
                        "button",
                        {
                          type: "button",
                          class: "chip evidence",
                          title: "Seek the video to " + fmtTime(ev.start_s),
                          onclick: () => seekTo(ev.start_s),
                        },
                        "▶ " + fmtRange(ev.start_s, ev.end_s)
                      ),
                      h("span", {}, ev.observation || "")
                    )
                  )
                )
              : null
          )
        )
      );

      // agentic details
      const skills = Array.isArray(q.agentic_skills) ? q.agentic_skills : [];
      const plan = Array.isArray(q.tool_plan) ? q.tool_plan : [];
      const used = Array.isArray(q.metadata_used) ? q.metadata_used : [];
      card.appendChild(
        h(
          "details",
          { class: "panel" },
          h("summary", {}, "Why this is hard / agentic plan"),
          h(
            "div",
            { class: "panel-body stack small" },
            q.why_hard ? h("p", {}, h("strong", {}, "Why hard: "), q.why_hard) : null,
            skills.length ? h("div", { class: "chips" }, skills.map((s) => h("span", { class: "chip skill" }, prettyLabel(s)))) : null,
            plan.length ? h("div", {}, h("strong", {}, "Tool plan"), h("ol", { class: "plan" }, plan.map((p) => h("li", {}, String(p))))) : null,
            used.length ? h("p", { class: "text-2" }, h("strong", {}, "Metadata used: "), used.join(", ")) : null
          )
        )
      );

      // correctness
      const corr = radioGroup(
        "corr-" + sid + "-" + qid,
        CORRECTNESS_OPTIONS,
        ann.correctness,
        (val) => {
          ensureQuestionAnn(sid, qid).correctness = val;
          afterChange(sid, true);
        },
        (val) => "c-" + val
      );
      card.appendChild(h("fieldset", { class: "ctl" }, h("legend", {}, "Is the gold answer correct?"), corr));

      // ratings
      const ratings = h("div", { class: "ratings" });
      for (const [key, label] of RATING_OPTIONS) {
        const group = radioGroup(
          key + "-" + sid + "-" + qid,
          [1, 2, 3, 4, 5].map((n) => [n, String(n)]),
          ann[key],
          (val) => {
            ensureQuestionAnn(sid, qid)[key] = Number(val);
            afterChange(sid, true);
          }
        );
        const clearBtn = h("button", { type: "button", class: "btn link small rating-clear", title: "Clear rating" }, "clear");
        clearBtn.addEventListener("click", () => {
          group._clear();
          ensureQuestionAnn(sid, qid)[key] = null;
          afterChange(sid, true);
        });
        ratings.appendChild(h("fieldset", { class: "ctl" }, h("legend", {}, label + " (1–5)"), h("div", { class: "row", style: { gap: "0.3rem" } }, group, clearBtn)));
      }
      card.appendChild(ratings);

      // comment
      const comment = h("textarea", { class: "small", placeholder: "Comment on this question (optional)", "aria-label": "Comment on question " + (i + 1), value: ann.comment });
      comment.addEventListener("input", () => {
        ensureQuestionAnn(sid, qid).comment = comment.value;
        afterChange(sid, false);
      });
      card.appendChild(h("div", { class: "field" }, h("div", { class: "field-head" }, "Comment"), comment));

      state.cardRefs[qid] = { card, checkbox, index: i };
      return card;
    }

    // ---- sample-level panel -----------------------------------------------------------------
    function buildSamplePanel(sid) {
      const ann = state.record.samples[sid] || emptySampleAnn();
      const flags = h("div", { class: "flags" });
      for (const [value, label] of FLAG_OPTIONS) {
        const cb = h("input", { type: "checkbox", value, checked: ann.flags.includes(value) });
        cb.addEventListener("change", () => {
          const s = touchSample(sid);
          s.flags = s.flags.filter((f) => f !== value);
          if (cb.checked) s.flags.push(value);
          afterChange(sid, true);
        });
        flags.appendChild(h("label", {}, cb, label));
      }
      const comment = h("textarea", { placeholder: "Sample-level comment (optional)", "aria-label": "Sample comment", value: ann.comment });
      comment.addEventListener("input", () => {
        touchSample(sid).comment = comment.value;
        afterChange(sid, false);
      });

      const selectedCount = h("span", { class: "chip" }, "");
      const complete = h("button", { type: "button", class: "btn success" }, "✓ Mark complete");
      complete.addEventListener("click", () => {
        const s = touchSample(sid);
        const chk = selectionCheck(s, state.limits.min, state.limits.max);
        if (!chk.ok) {
          toast(chk.message, true);
          return;
        }
        s.status = "done";
        saveRecord();
        refreshStatusUI();
        toast("Marked complete");
        if (state.prefs.auto_advance) {
          const next = nextUnfinished(navIds(), sid, state.record);
          if (next) location.hash = sampleHash(next);
          else toast("All samples in this list are complete — remember to Submit or Export");
        }
      });
      const reopen = h("button", { type: "button", class: "btn", hidden: true }, "Reopen");
      reopen.addEventListener("click", () => {
        touchSample(sid).status = "in_progress";
        saveRecord();
        refreshStatusUI();
      });
      const hint = h("span", { class: "hint" }, "");
      const autoAdv = h("input", { type: "checkbox", checked: !!state.prefs.auto_advance });
      autoAdv.addEventListener("change", () => savePrefs({ auto_advance: autoAdv.checked }));
      const bottomStatus = h("span", {});

      const nav = h(
        "div",
        { class: "btn-group" },
        h("button", { type: "button", class: "btn", onclick: () => goRelative(-1) }, "← Prev"),
        h("button", { type: "button", class: "btn", onclick: () => goRelative(1) }, "Next →"),
        h("button", { type: "button", class: "btn", onclick: goNextUnfinished }, "Next unfinished »")
      );

      state.cardRefs._selectedCount = selectedCount;
      state.cardRefs._complete = complete;
      state.cardRefs._reopen = reopen;
      state.cardRefs._hint = hint;
      state.cardRefs._bottomStatus = bottomStatus;

      return h(
        "div",
        { class: "card sample-panel" },
        h("div", { class: "row between" }, h("h2", {}, "Sample review"), h("span", { class: "row" }, bottomStatus, selectedCount)),
        h("fieldset", { class: "ctl" }, h("legend", {}, "Flags"), flags),
        h("div", { class: "field" }, h("div", { class: "field-head" }, "Comment"), comment),
        h(
          "div",
          { class: "action-bar" },
          complete,
          reopen,
          h("label", { class: "small text-2", style: { display: "inline-flex", gap: "0.35em", alignItems: "center" } }, autoAdv, "auto-advance after completing"),
          h("span", { class: "grow" }),
          nav,
          hint
        )
      );
    }

    // ---------------------------------------------------------------- help dialog
    function openHelp() {
      const dlg = $("dlg-help");
      if (dlg.open) return;
      if (typeof dlg.showModal === "function") dlg.showModal();
      else dlg.setAttribute("open", "");
    }

    function closeHelp() {
      const dlg = $("dlg-help");
      if (typeof dlg.close === "function" && dlg.open) dlg.close();
      else dlg.removeAttribute("open");
    }

    function updateSelectRuleText() {
      const rule = selectRuleText(state.limits);
      document.querySelectorAll(".select-rule").forEach((el) => (el.textContent = rule));
      document.querySelectorAll(".select-min").forEach((el) => (el.textContent = String(state.limits.min)));
      document.querySelectorAll(".select-max").forEach((el) => (el.textContent = String(state.limits.max)));
    }

    // ---------------------------------------------------------------- keyboard
    function isTypingTarget(t) {
      if (!t || !t.tagName) return false;
      const tag = t.tagName.toLowerCase();
      return tag === "input" || tag === "textarea" || tag === "select" || t.isContentEditable;
    }

    function onKey(e) {
      if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) return;
      const dlg = $("dlg-help");
      if (e.key === "Escape") {
        if (dlg.open) {
          closeHelp();
          e.preventDefault();
        }
        return;
      }
      if (isTypingTarget(e.target) || dlg.open) return;
      if (e.key === "?") {
        openHelp();
        e.preventDefault();
        return;
      }
      if (!state.current || $("view-sample").hidden) return;
      if (e.key === "j") goRelative(1);
      else if (e.key === "k") goRelative(-1);
      else if (e.key === "n") goNextUnfinished();
      else if (e.key === "1" || e.key === "2" || e.key === "3") {
        const idx = Number(e.key) - 1;
        const qid = Object.keys(state.cardRefs).find((k) => !k.startsWith("_") && state.cardRefs[k].index === idx);
        if (!qid) return;
        const s = state.record.samples[state.current];
        const cur = s && s.questions[qid] ? !!s.questions[qid].selected : false;
        setSelected(state.current, qid, !cur);
      } else return;
      e.preventDefault();
    }

    // ---------------------------------------------------------------- static bindings & boot
    function bindStatic() {
      $("annotator-form").addEventListener("submit", (e) => {
        e.preventDefault();
        const id = canonicalAnnotator($("annotator-input").value, knownAnnotators());
        if (!id) {
          toast("Please enter an annotator ID (letters, digits, . _ -)", true);
          return;
        }
        setAnnotator(id);
      });
      $("btn-switch").addEventListener("click", () => {
        saveRecord();
        state.annotator = "";
        try {
          localStorage.removeItem(ANNOTATOR_KEY);
        } catch (e) {
          /* ignore */
        }
        state.record = null;
        renderAnnotator();
        $("progress-card").hidden = true;
        $("list-card").hidden = true;
        $("annotator-input").value = "";
        $("annotator-input").focus();
      });
      $("filter-dataset").addEventListener("change", (e) => {
        state.filters.dataset = e.target.value;
        renderList();
      });
      $("filter-status").addEventListener("change", (e) => {
        state.filters.status = e.target.value;
        renderList();
      });
      let qTimer = null;
      $("filter-query").addEventListener("input", (e) => {
        clearTimeout(qTimer);
        const val = e.target.value;
        qTimer = setTimeout(() => {
          state.filters.query = val;
          renderList();
          $("filter-query").focus();
        }, 150);
      });
      $("toggle-thumbs").addEventListener("change", (e) => {
        savePrefs({ thumbs: e.target.checked });
        renderList();
      });
      const fb = $("filter-batch");
      if (fb) {
        fb.addEventListener("change", (e) => {
          state.filters.batch = e.target.value;
          renderList();
        });
      }
      const bv = $("btn-videos");
      if (bv) {
        bv.addEventListener("click", () => {
          if (state.localSource && state.localSource.kind === "reconnect") reconnectVideosFolder();
          else pickVideosFolder();
        });
      }
      const fv = $("file-videos");
      if (fv) {
        fv.addEventListener("change", (e) => {
          onVideosInput(e.target.files);
          e.target.value = "";
        });
      }
      $("btn-export").addEventListener("click", exportJson);
      $("btn-import").addEventListener("click", () => $("file-import").click());
      $("file-import").addEventListener("change", (e) => {
        const f = e.target.files && e.target.files[0];
        if (f) importFile(f);
        e.target.value = "";
      });
      $("btn-submit").addEventListener("click", submitAnnotations);
      $("btn-theme").addEventListener("click", cycleTheme);
      $("btn-help").addEventListener("click", openHelp);
      $("btn-help-close").addEventListener("click", closeHelp);
      $("dlg-help").addEventListener("click", (e) => {
        if (e.target === e.currentTarget) closeHelp(); // click on backdrop
      });
      $("btn-prev").addEventListener("click", () => goRelative(-1));
      $("btn-next").addEventListener("click", () => goRelative(1));
      $("btn-next-unfinished").addEventListener("click", goNextUnfinished);
      document.addEventListener("keydown", onKey);
      window.addEventListener("beforeunload", () => {
        if (state.saveTimer) saveRecord();
      });
      document.addEventListener("visibilitychange", () => {
        if (document.visibilityState === "hidden" && state.saveTimer) saveRecord();
      });
    }

    async function boot() {
      loadPrefs();
      applyTheme(localStorage.getItem(THEME_KEY) || "auto");
      bindStatic();
      try {
        await loadData();
      } catch (e) {
        showLoadError(e);
      }
      state.annotator = canonicalAnnotator(localStorage.getItem(ANNOTATOR_KEY) || "", knownAnnotators());
      state.limits = resolveSelectLimits(CONFIG, state.meta);
      updateSelectRuleText();
      if (state.annotator) loadRecord();
      renderAnnotator();
      window.addEventListener("hashchange", route);
      route();
      restoreVideosFolder().catch(() => {});
    }

    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
    else boot();
  })();
}
