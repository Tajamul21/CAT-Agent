/* CAT-Agent clinician review app (vanilla JS, no build step, no network calls).
 *
 * The clinician enters a name/ID, opens the batch folder they received (videos + <sample_id>.qa.json
 * files with the GPT-generated questions) and reviews each video. Everything stays on their computer.
 *
 * Saving (only once a name/ID has been entered):
 *   - every change is saved in this browser (localStorage);
 *   - in Chrome/Edge, when the folder was opened with write access, progress is also written into the
 *     batch folder itself (catagent_progress_<id>.json) a moment after each change, and reloaded
 *     automatically the next time that folder is opened;
 *   - "Save progress" saves immediately (to the folder, or as a downloaded progress file that can be
 *     put back into the batch folder or loaded with "Load saved progress").
 * Exports (JSON / CSV) keep the original GPT question/answer next to the edited version.
 */
(function () {
  "use strict";

  const CFG = Object.assign({ SELECT_MIN: 1, SELECT_MAX: 2 },
    (typeof window !== "undefined" && window.OPHBENCH_CONFIG) || {});
  const UI_VERSION = 2;
  const TOOL = "CAT-Agent clinician review";
  const VIDEO_EXT = ["mp4", "m4v", "mov", "webm", "mkv", "ogv"];
  const CORRECTNESS = [
    ["correct", "Correct"], ["partial", "Partly"], ["incorrect", "Incorrect"], ["cannot_verify", "Can't tell"],
  ];
  const PALETTE = ["#7cc4b8", "#9bb7e3", "#e7b07a", "#c3a6dd", "#e59aa8", "#a8cf8e", "#e6cf72", "#8fc9e0",
    "#d9a3c8", "#b9c08a", "#f0a989", "#a3b0c2"];
  const SA = {
    L1_perception: ["L1 Perception", "sa1"], L2_comprehension: ["L2 Comprehension", "sa2"],
    L3_projection: ["L3 Projection", "sa3"],
  };
  const LENGTH = { one_word: "one word", short_phrase: "short", multi_line: "multi-line" };
  const FOLDER_SAVE_DELAY_MS = 1500;
  const TICK_S = 5;              // activity clock resolution
  const IDLE_MS = 120000;        // no input for 2 min (and no video playing) = idle, not counted

  // ------------------------------------------------------------------ state
  const state = {
    reviewer: "",
    cases: [],          // [{id, doc, video: File|null}]
    current: -1,
    filter: "all",
    search: "",
    ann: {},            // sample_id -> annotation record
    editing: {},        // qid -> "question" | "answer" while an edit form is open (current case only)
    videoUrl: null,
    dirHandle: null,    // FileSystemDirectoryHandle of the batch folder (Chrome/Edge), when writable
    canWrite: false,
    unsaved: false,     // changes made while no name/ID was entered
    lastSaved: null,    // {where, at}
    activity: { by_sample: {}, sessions: [] },  // active seconds per video + work sessions
    session: null,      // the current session object (inside activity.sessions)
    lastInput: Date.now(),
  };

  // ------------------------------------------------------------------ helpers
  const $ = (sel, root = document) => root.querySelector(sel);

  function el(tag, props, ...kids) {
    const node = document.createElement(tag);
    if (props) {
      for (const [k, v] of Object.entries(props)) {
        if (v === undefined || v === null || v === false) continue;
        if (k === "class") node.className = v;
        else if (k === "text") node.textContent = v;
        else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
        else if (k === "value") node.value = v;
        else if (k === "style" && typeof v === "object") Object.assign(node.style, v);
        else if (k in node && typeof v !== "string") node[k] = v;
        else node.setAttribute(k, v === true ? "" : String(v));
      }
    }
    for (const kid of kids.flat(Infinity)) {
      if (kid === null || kid === undefined || kid === false) continue;
      node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return node;
  }

  function fmtTime(t) {
    if (t === null || t === undefined || isNaN(t)) return "?";
    t = Math.max(0, Number(t));
    const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = Math.floor(t % 60);
    const mm = String(m).padStart(h ? 2 : 1, "0"), ss = String(s).padStart(2, "0");
    return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
  }
  function fmtDuration(t) {
    if (!t) return "";
    t = Number(t);
    if (t < 60) return `${Math.round(t)} s`;
    return `${Math.floor(t / 60)} min ${String(Math.round(t % 60)).padStart(2, "0")} s`;
  }
  function fmtActive(sec) {
    sec = Math.round(Number(sec) || 0);
    if (sec < 60) return `${sec} s`;
    const h = Math.floor(sec / 3600), m = Math.round((sec % 3600) / 60);
    return h ? `${h} h ${String(m).padStart(2, "0")} min` : `${m} min`;
  }
  const humanize = (s) => String(s || "").replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
  const prettyFamily = (s) => String(s || "").replace(/\s*->\s*/g, " → ");
  const nowIso = () => new Date().toISOString();
  const clock = (d) => d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const stem = (name) => String(name || "").split("/").pop().replace(/\.qa\.json$/i, "").replace(/\.[^.]+$/, "");
  const ext = (name) => (String(name).split(".").pop() || "").toLowerCase();
  const safeName = (s) => String(s || "reviewer").trim().replace(/[^A-Za-z0-9_.-]+/g, "_");
  const stamp = () => new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-");
  function colorFor(label) {
    let h = 0;
    for (const ch of String(label)) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
    return PALETTE[h % PALETTE.length];
  }

  let toastTimer = null;
  function toast(msg, ms = 2600) {
    const t = $("#toast");
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove("show"), ms);
  }

  // ------------------------------------------------------------------ storage (only with a name/ID)
  const storeKey = () => `catagent.review.v2.${state.reviewer.toLowerCase()}`;
  const progressFileName = () => `catagent_progress_${safeName(state.reviewer)}.json`;
  let folderTimer = null;

  function normRecord(s) {
    const qs = {};
    for (const [qid, q] of Object.entries((s && s.questions) || {})) {
      const qe = q.question_edit || "", ae = q.answer_edit || "", oe = Array.isArray(q.options_edit) ? q.options_edit : null;
      qs[qid] = { selected: !!q.selected, correctness: q.correctness || "", comment: q.comment || "",
        question_edit: qe, answer_edit: ae, options_edit: oe, edited: !!(qe || ae || oe) };
    }
    return { status: (s && s.status) || "in_progress", updated_at: (s && s.updated_at) || nowIso(),
      comment: (s && s.comment) || "", flags: (s && s.flags) || [], questions: qs };
  }
  /** Merge sample records into state.ann; the newer `updated_at` wins. Returns the number taken. */
  function mergeAnn(samples) {
    let n = 0;
    for (const [sid, s] of Object.entries(samples || {})) {
      const cur = state.ann[sid];
      if (cur && cur.updated_at && s && s.updated_at && cur.updated_at >= s.updated_at) continue;
      state.ann[sid] = normRecord(s);
      n += 1;
    }
    return n;
  }
  function loadStore() {
    if (!state.reviewer) return;
    try {
      const raw = localStorage.getItem(storeKey());
      if (raw) mergeAnn(JSON.parse(raw));
    } catch (e) { /* unreadable store: start empty */ }
  }
  // ---- activity (active time per video and per session; only with a name/ID) ----
  const activityKey = () => `catagent.activity.v2.${state.reviewer.toLowerCase()}`;
  function loadActivity() {
    state.activity = { by_sample: {}, sessions: [] };
    state.session = null;
    if (!state.reviewer) return;
    try {
      const raw = localStorage.getItem(activityKey());
      if (raw) mergeActivity(JSON.parse(raw));
    } catch (e) { /* start empty */ }
  }
  function saveActivity() {
    if (!state.reviewer) return;
    try { localStorage.setItem(activityKey(), JSON.stringify(state.activity)); } catch (e) { /* ignore */ }
  }
  function mergeActivity(a) {
    if (!a || typeof a !== "object") return;
    for (const [sid, sec] of Object.entries(a.by_sample || {})) {
      state.activity.by_sample[sid] = Math.max(Number(state.activity.by_sample[sid]) || 0, Number(sec) || 0);
    }
    const seen = new Set(state.activity.sessions.map((s) => s.started_at));
    for (const s of a.sessions || []) {
      if (s && s.started_at && !seen.has(s.started_at)) { state.activity.sessions.push({ ...s }); seen.add(s.started_at); }
    }
    state.activity.sessions.sort((x, y) => String(x.started_at).localeCompare(String(y.started_at)));
    if (state.activity.sessions.length > 200) state.activity.sessions = state.activity.sessions.slice(-200);
  }
  const totalActive = () => Object.values(state.activity.by_sample).reduce((a, b) => a + (Number(b) || 0), 0);
  function startSession() {
    if (!state.reviewer || state.session) return;
    state.session = { started_at: nowIso(), last_at: nowIso(), active_s: 0, videos: [] };
    state.activity.sessions.push(state.session);
    saveActivity();
  }
  function tick() {
    if (!state.reviewer || state.current < 0 || !state.cases.length || document.visibilityState !== "visible") return;
    const v = $("#player");
    const playing = !!(v && !v.paused && !v.ended);
    if (!playing && Date.now() - state.lastInput > IDLE_MS) return;
    startSession();
    const sid = state.cases[state.current].id;
    state.activity.by_sample[sid] = (Number(state.activity.by_sample[sid]) || 0) + TICK_S;
    state.session.active_s += TICK_S;
    state.session.last_at = nowIso();
    if (!state.session.videos.includes(sid)) state.session.videos.push(sid);
    saveActivity();
    updateTimeLabels();
  }
  function updateTimeLabels() {
    const tot = $("#active-time");
    if (tot) tot.textContent = state.reviewer ? `Active time ${fmtActive(totalActive())}` : "";
    const here = $("#time-here");
    const c = state.cases[state.current];
    if (here && c) here.textContent = `⏱ ${fmtActive(state.activity.by_sample[c.id] || 0)} on this video`;
  }

  function setSaveStatus() {
    const s = $("#save-status");
    if (!s) return;
    s.classList.remove("hidden", "warn");
    if (!state.reviewer) {
      s.textContent = "Not saving: enter your name or ID";
      s.classList.add("warn");
    } else if (state.lastSaved) {
      s.textContent = `Saved ${state.lastSaved.where} · ${clock(state.lastSaved.at)}`;
    } else {
      s.textContent = state.canWrite ? "Saves to this browser and your batch folder" : "Saves to this browser";
    }
  }
  async function writeProgressToFolder() {
    if (!state.dirHandle || !state.canWrite || !state.reviewer) return false;
    try {
      const fh = await state.dirHandle.getFileHandle(progressFileName(), { create: true });
      const w = await fh.createWritable();
      await w.write(JSON.stringify(buildExport(), null, 2));
      await w.close();
      state.lastSaved = { where: "to folder", at: new Date() };
      setSaveStatus();
      return true;
    } catch (e) {
      state.canWrite = false;   // permission refused or folder gone: fall back to browser + download
      setSaveStatus();
      return false;
    }
  }
  /** Save the current state. Returns false (and saves nothing) when no name/ID was entered. */
  function persist() {
    if (!state.reviewer) {
      state.unsaved = true;
      setSaveStatus();
      return false;
    }
    try {
      localStorage.setItem(storeKey(), JSON.stringify(state.ann));
      state.unsaved = false;
      state.lastSaved = { where: "in browser", at: new Date() };
    } catch (e) {
      toast("This browser could not save. Use Save progress to download a progress file.");
    }
    if (state.dirHandle && state.canWrite) {
      clearTimeout(folderTimer);
      folderTimer = setTimeout(writeProgressToFolder, FOLDER_SAVE_DELAY_MS);
    }
    setSaveStatus();
    return true;
  }
  function setReviewer(name) {
    const next = String(name || "").trim();
    if (next === state.reviewer) return;
    const carry = state.reviewer ? {} : state.ann;   // work done before any name/ID was entered
    state.reviewer = next;
    try { localStorage.setItem("catagent.reviewer", next); } catch (e) { /* ignore */ }
    state.ann = {};
    loadActivity();
    if (next) {
      loadStore();
      mergeAnn(carry);
      if (Object.keys(carry).length) persist();
    } else {
      state.ann = carry;
    }
    state.lastSaved = null;
    for (const input of document.querySelectorAll(".reviewer-input")) if (input.value.trim() !== next) input.value = next;
    updateStartButtons();
    setSaveStatus();
  }
  function requireReviewer() {
    if (state.reviewer) return true;
    for (const input of document.querySelectorAll(".reviewer-input")) input.classList.add("needs");
    const visible = !$("#start").classList.contains("hidden") ? $("#reviewer-start") : $("#reviewer");
    if (visible) visible.focus();
    toast("Please enter your name or ID first. Nothing is saved without it.");
    return false;
  }

  // ------------------------------------------------------------------ annotations
  function annFor(sid, create = false) {
    let a = state.ann[sid];
    if (!a && create) {
      a = state.ann[sid] = { status: "in_progress", updated_at: nowIso(), comment: "", flags: [], questions: {} };
    }
    return a;
  }
  function qAnn(sid, qid, create = false) {
    const a = annFor(sid, create);
    if (!a) return null;
    if (!a.questions[qid] && create) {
      a.questions[qid] = { selected: false, correctness: "", edited: false, comment: "",
        question_edit: "", answer_edit: "", options_edit: null };
    }
    return a.questions[qid] || null;
  }
  function touch(sid) {
    const a = annFor(sid, true);
    a.updated_at = nowIso();
    if (a.status !== "done") a.status = "in_progress";
    persist();
    renderList();
    updateProgress();
  }
  function finalOf(q, qa) {
    return {
      question: qa && qa.question_edit ? qa.question_edit : q.question || "",
      answer: qa && qa.answer_edit ? qa.answer_edit : q.answer || "",
      options: qa && Array.isArray(qa.options_edit) ? qa.options_edit : (q.options || []),
    };
  }
  const statusOf = (sid) => (state.ann[sid] && state.ann[sid].status) || "todo";
  const selectedCount = (sid) => {
    const a = state.ann[sid];
    return a ? Object.values(a.questions).filter((q) => q.selected).length : 0;
  };

  // ------------------------------------------------------------------ opening the batch folder
  async function filesFromDirHandle(dir) {
    const files = [];
    for await (const [, h] of dir.entries()) {
      if (h.kind === "file") files.push(await h.getFile());
    }
    return files;
  }

  async function openFolder() {
    if (!requireReviewer()) return;
    if (typeof window.showDirectoryPicker === "function") {
      try {
        const dir = await window.showDirectoryPicker({ id: "catagent-batch", mode: "readwrite" });
        state.dirHandle = dir;
        state.canWrite = true;
        await loadFiles(await filesFromDirHandle(dir));
        return;
      } catch (e) {
        if (e && e.name === "AbortError") return;     // the user cancelled the picker
        state.dirHandle = null;
        state.canWrite = false;                         // e.g. write permission refused: read-only fallback
      }
    }
    $("#input-folder").click();
  }

  async function onDrop(dt) {
    if (!requireReviewer()) return;
    const items = dt.items ? Array.from(dt.items) : [];
    // both lookups must start synchronously, before the drop event ends
    const handlePromises = items.map((it) => (typeof it.getAsFileSystemHandle === "function" ? it.getAsFileSystemHandle() : null));
    const entries = items.map((it) => (it.webkitGetAsEntry ? it.webkitGetAsEntry() : null)).filter(Boolean);
    const plainFiles = Array.from(dt.files || []);
    try {
      const handles = (await Promise.all(handlePromises.filter(Boolean))).filter(Boolean);
      const dir = handles.find((h) => h.kind === "directory");
      if (dir) {
        let granted = false;
        try { granted = (await dir.requestPermission({ mode: "readwrite" })) === "granted"; } catch (e) { granted = false; }
        state.dirHandle = granted ? dir : null;
        state.canWrite = granted;
        await loadFiles(await filesFromDirHandle(dir));
        return;
      }
    } catch (e) { /* fall back to the entry API below */ }
    state.dirHandle = null;
    state.canWrite = false;
    if (!entries.length) { await loadFiles(plainFiles); return; }
    const out = [];
    async function walk(entry) {
      if (entry.isFile) {
        await new Promise((res) => entry.file((f) => { out.push(f); res(); }, () => res()));
      } else if (entry.isDirectory) {
        const reader = entry.createReader();
        let batch;
        do {
          batch = await new Promise((res) => reader.readEntries(res, () => res([])));
          for (const e of batch) await walk(e);
        } while (batch.length);
      }
    }
    for (const e of entries) await walk(e);
    await loadFiles(out);
  }

  function isProgressFile(data) {
    return data && typeof data === "object" && !Array.isArray(data) && data.samples &&
      typeof data.samples === "object" && !Array.isArray(data.samples) && "annotator" in data;
  }

  async function loadFiles(files) {
    files = Array.from(files || []);
    if (!files.length) return;
    const videos = new Map();   // stem -> File
    const docs = new Map();     // sample_id -> doc
    const progress = [];        // saved progress files of this reviewer
    let badJson = 0, otherReviewers = 0;
    for (const f of files) {
      const name = f.name || "";
      if (name.startsWith(".")) continue;
      const e = ext(name);
      if (VIDEO_EXT.includes(e)) { videos.set(stem(name).toLowerCase(), f); continue; }
      if (e !== "json") continue;
      try {
        const data = JSON.parse(await f.text());
        if (isProgressFile(data)) {
          if (String(data.annotator || "").trim().toLowerCase() === state.reviewer.toLowerCase()) progress.push(data);
          else otherReviewers += 1;
          continue;
        }
        const list = Array.isArray(data) ? data : Array.isArray(data.samples) ? data.samples : [data];
        for (const d of list) {
          if (d && Array.isArray(d.questions) && d.questions.length) {
            const sid = String(d.sample_id || stem(name));
            docs.set(sid, Object.assign({ sample_id: sid }, d));
          }
        }
      } catch (err) { badJson += 1; }
    }
    let restored = 0;
    for (const p of progress) { restored += mergeAnn(p.samples); mergeActivity(p.activity); }
    if (progress.length) saveActivity();
    if (restored) persist();
    if (!docs.size) {
      if (restored && state.cases.length) { renderCase(); renderList(); updateProgress(); toast(`Restored ${restored} saved review(s)`); return; }
      toast(videos.size ? "No question files found. Open the whole batch folder (it contains .qa.json files)."
        : "No videos or question files found in that selection.");
      return;
    }
    const existing = new Map(state.cases.map((c) => [c.id, c]));
    for (const [sid, doc] of docs) {
      const keys = [doc.video && doc.video.local_file && stem(doc.video.local_file), sid].filter(Boolean).map((k) => k.toLowerCase());
      const vid = keys.map((k) => videos.get(k)).find(Boolean) || (existing.get(sid) && existing.get(sid).video) || null;
      existing.set(sid, { id: sid, doc, video: vid });
    }
    for (const c of existing.values()) {
      if (!c.video) {
        const v = videos.get(c.id.toLowerCase()) || (c.doc.video && c.doc.video.local_file && videos.get(stem(c.doc.video.local_file).toLowerCase()));
        if (v) c.video = v;
      }
    }
    state.cases = Array.from(existing.values()).sort((a, b) => {
      const oa = a.doc.order ?? 1e9, ob = b.doc.order ?? 1e9;
      return oa - ob || a.id.localeCompare(b.id);
    });
    const withVideo = state.cases.filter((c) => c.video).length;
    startSession();
    showWorkspace();
    const firstTodo = state.cases.findIndex((c) => statusOf(c.id) !== "done");
    openCase(firstTodo >= 0 ? firstTodo : 0);
    const done = state.cases.filter((c) => statusOf(c.id) === "done").length;
    toast(`Loaded ${state.cases.length} video${state.cases.length === 1 ? "" : "s"}` +
      (done || restored ? ` · your saved progress is back (${done} done)` : "") +
      (withVideo < state.cases.length ? ` · ${state.cases.length - withVideo} without a video file` : "") +
      (otherReviewers ? ` · ${otherReviewers} other reviewer file(s) ignored` : "") +
      (badJson ? ` · ${badJson} unreadable file(s) skipped` : ""), 4200);
  }

  function showWorkspace() {
    $("#start").classList.add("hidden");
    $("#workspace").classList.remove("hidden");
    $("#btn-open").classList.remove("hidden");
    $("#btn-save").classList.remove("hidden");
    $("#btn-export").disabled = false;
    setSaveStatus();
    renderList();
    updateProgress();
    updateTimeLabels();
  }

  // ------------------------------------------------------------------ sidebar
  function visibleCases() {
    const q = state.search.trim().toLowerCase();
    return state.cases.map((c, i) => ({ c, i })).filter(({ c }) => {
      const st = statusOf(c.id);
      if (state.filter === "done" && st !== "done") return false;
      if (state.filter === "todo" && st === "done") return false;
      if (!q) return true;
      const hay = `${c.id} ${c.doc.procedure || ""} ${c.doc.dataset_title || c.doc.dataset || ""}`.toLowerCase();
      return hay.includes(q);
    });
  }
  function renderList() {
    const list = $("#case-list");
    list.replaceChildren(...visibleCases().map(({ c, i }) => {
      const st = statusOf(c.id);
      const n = selectedCount(c.id);
      return el("li", {
        class: `case-item${i === state.current ? " active" : ""}`, tabindex: 0, title: c.id,
        onclick: () => openCase(i),
        onkeydown: (e) => { if (e.key === "Enter") openCase(i); },
      },
      el("span", { class: `dot${st === "done" ? " done" : st === "in_progress" ? " progress-dot" : ""}`, "aria-label": st }),
      el("div", null,
        el("div", { class: "case-title", text: c.doc.procedure || c.id }),
        el("div", { class: "case-sub", text: [c.doc.dataset_title || c.doc.dataset, fmtDuration(c.doc.video && c.doc.video.duration_s)].filter(Boolean).join(" · ") })),
      el("span", { class: n ? "case-stars" : "case-num", text: n ? `★${n}` : String(i + 1) }));
    }));
  }
  function updateProgress() {
    const total = state.cases.length;
    const done = state.cases.filter((c) => statusOf(c.id) === "done").length;
    const pct = total ? Math.round((100 * done) / total) : 0;
    $("#progress-text").textContent = `${done} of ${total} done`;
    $("#progress-pct").textContent = `${pct}%`;
    $("#progress-bar").style.width = `${pct}%`;
  }

  // ------------------------------------------------------------------ case view
  function openCase(i) {
    if (i < 0 || i >= state.cases.length) return;
    state.current = i;
    state.editing = {};
    renderCase();
    renderList();
    const active = $(".case-item.active");
    if (active) active.scrollIntoView({ block: "nearest" });
    window.scrollTo(0, 0);
  }

  function seekTo(t) {
    const v = $("#player");
    if (!v) return;
    v.currentTime = Math.max(0, Number(t) || 0);
    v.play().catch(() => {});
  }

  function phaseSegments(doc) {
    return (doc.segments || []).filter((s) => !s.kind || ["phase", "step", "operation"].includes(s.kind));
  }

  function renderVideoPanel(c) {
    const doc = c.doc;
    if (state.videoUrl) { URL.revokeObjectURL(state.videoUrl); state.videoUrl = null; }
    let media;
    if (c.video) {
      state.videoUrl = URL.createObjectURL(c.video);
      media = el("video", { id: "player", src: state.videoUrl, controls: true, preload: "metadata", playsinline: true });
    } else {
      media = el("div", { class: "video-missing" },
        el("b", { text: "Video file not loaded" }),
        `Add ${doc.video && doc.video.local_file ? doc.video.local_file : c.id + ".mp4"} from your batch folder (Open folder, top right).`);
    }
    const segs = phaseSegments(doc);
    const duration = Number(doc.video && doc.video.duration_s) || Math.max(0, ...segs.map((s) => s.end_s || 0));
    let timeline = null, phaseNow = null;
    if (segs.length && duration > 0) {
      const head = el("div", { class: "playhead" });
      const bar = el("div", {
        class: "timeline", title: "Click to jump", style: { display: "block" },
        onclick: (e) => { const r = bar.getBoundingClientRect(); seekTo(((e.clientX - r.left) / r.width) * duration); },
      }, segs.map((s) => el("div", {
        class: "seg", title: `${s.label}  ${fmtTime(s.start_s)}–${fmtTime(s.end_s)}`,
        style: {
          position: "absolute", top: "0", bottom: "0", background: colorFor(s.label),
          left: `${(100 * (s.start_s || 0)) / duration}%`,
          width: `${(100 * Math.max(0, (s.end_s || 0) - (s.start_s || 0))) / duration}%`,
        },
      })), head);
      timeline = el("div", { class: "timeline-wrap" },
        el("div", { class: "timeline-label" }, el("span", { text: "Surgical phases (click to jump)" }), el("span", { text: fmtTime(duration) })),
        bar);
      phaseNow = el("div", { class: "phase-now", text: "Play the video to see the current phase." });
      if (media.tagName === "VIDEO") {
        media.addEventListener("timeupdate", () => {
          const t = media.currentTime;
          head.style.left = `${Math.min(100, (100 * t) / duration)}%`;
          const cur = segs.filter((s) => t >= s.start_s && t < s.end_s).map((s) => s.label);
          phaseNow.replaceChildren(el("span", { text: `${fmtTime(t)}  ·  ` }), el("b", { text: cur.length ? cur.join(" + ") : "—" }));
        });
      }
    }
    const meta = (doc.metadata || []).filter((m) => m && m.label && m.value !== "" && m.value !== null);
    const about = el("details", { class: "about" },
      el("summary", { text: "About this video" }),
      el("div", { class: "about-body" },
        doc.video_summary ? el("p", { text: doc.video_summary }) : null,
        meta.length ? el("dl", { class: "kv" }, meta.map((m) => [el("dt", { text: m.label }), el("dd", { text: String(m.value) })])) : null,
        doc.timeline_note ? el("p", { class: "orig", text: doc.timeline_note }) : null));
    return el("section", { class: "panel video-panel" }, el("div", { class: "video-box" }, media), timeline, phaseNow, about);
  }

  function correctOptionIndex(options, answer) {
    if (!options || !options.length || !answer) return -1;
    const m = String(answer).trim().match(/^\(?([A-Ha-h])[\).:\s]/);
    if (m) {
      const letter = m[1].toUpperCase();
      const k = options.findIndex((o) => String(o).trim().toUpperCase().startsWith(letter + ".") ||
        String(o).trim().toUpperCase().startsWith(letter + ")"));
      if (k >= 0) return k;
    }
    const a = String(answer).trim().toLowerCase();
    return options.findIndex((o) => {
      const body = String(o).replace(/^[A-Ha-h][\).:]\s*/, "").trim().toLowerCase();
      return body && (a === body || a.startsWith(body) || a.includes(body.slice(0, 60)));
    });
  }

  // edits: question (+ options) and answer are edited separately; the GPT original is always kept
  function saveEdit(sid, qid, q, field, values) {
    const r = qAnn(sid, qid, true);
    const origQ = String(q.question || "").trim(), origA = String(q.answer || "").trim();
    const origO = (q.options || []).map((s) => String(s).trim());
    if (field === "question") {
      const nq = String(values.question || "").trim();
      r.question_edit = nq && nq !== origQ ? nq : "";
      if (values.options !== undefined) {
        const no = String(values.options || "").split("\n").map((s) => s.trim()).filter(Boolean);
        r.options_edit = JSON.stringify(no) !== JSON.stringify(origO) ? no : null;
      }
    } else {
      const na = String(values.answer || "").trim();
      r.answer_edit = na && na !== origA ? na : "";
    }
    r.edited = !!(r.question_edit || r.answer_edit || r.options_edit);
    delete state.editing[qid];
    touch(sid);
    renderCase();
    return field === "question" ? !!(r.question_edit || r.options_edit) : !!r.answer_edit;
  }
  function restoreField(sid, qid, field) {
    const r = qAnn(sid, qid, true);
    if (field === "question") { r.question_edit = ""; r.options_edit = null; } else { r.answer_edit = ""; }
    r.edited = !!(r.question_edit || r.answer_edit || r.options_edit);
    delete state.editing[qid];
    touch(sid);
    renderCase();
    toast(`Original ${field} restored`);
  }

  function editForm(sid, qid, q, field, fin) {
    const isQ = field === "question";
    const main = el("textarea", { rows: isQ ? 4 : 3, value: isQ ? fin.question : fin.answer, "aria-label": isQ ? "Edit question" : "Edit answer" });
    const hasOptions = isQ && ((fin.options && fin.options.length) || q.answer_type === "multiple_choice");
    const opts = hasOptions ? el("textarea", { rows: Math.max(3, (fin.options || []).length + 1), value: (fin.options || []).join("\n"), "aria-label": "Edit options" }) : null;
    const orig = isQ ? q.question : q.answer;
    setTimeout(() => main.focus(), 0);
    return el("div", { class: "edit-box" },
      el("label", null, isQ ? "Question" : "Answer", main),
      opts ? el("label", null, "Answer options, one per line", opts) : null,
      el("div", { class: "orig" }, el("b", { text: `GPT original ${field}: ` }), orig || ""),
      el("div", { class: "edit-row" },
        el("button", { class: "btn subtle small", type: "button", text: "Restore original", onclick: () => restoreField(sid, qid, field) }),
        el("button", { class: "btn ghost small", type: "button", text: "Cancel", onclick: () => { delete state.editing[qid]; renderCase(); } }),
        el("button", {
          class: "btn primary small", type: "button", text: `Save ${field}`,
          onclick: () => {
            const changed = saveEdit(sid, qid, q, field, isQ ? { question: main.value, options: opts ? opts.value : undefined } : { answer: main.value });
            toast(changed ? `Edited ${field} saved · the original is kept too` : "No changes");
          },
        })));
  }

  function renderQuestion(c, q, k) {
    const sid = c.id, qid = q.qid || `q${k + 1}`;
    const qa = qAnn(sid, qid);
    const fin = finalOf(q, qa);
    const selected = !!(qa && qa.selected);
    const editing = state.editing[qid] || "";
    const qEdited = !!(qa && (qa.question_edit || qa.options_edit));
    const aEdited = !!(qa && qa.answer_edit);

    const star = el("button", {
      class: `star-btn${selected ? " on" : ""}`, type: "button", "aria-pressed": String(selected),
      title: `Mark as one of the best questions (key ${k + 1})`,
      onclick: () => toggleSelect(sid, qid),
    }, el("span", { class: "s", text: selected ? "★" : "☆" }), selected ? "Best" : "Mark best");

    const tags = el("div", { class: "q-tags" },
      el("span", { class: "q-index", text: `Q${k + 1}` }),
      SA[q.sa_level] ? el("span", { class: `pill ${SA[q.sa_level][1]}`, text: SA[q.sa_level][0] })
        : q.category ? el("span", { class: "pill accent", text: humanize(q.category) }) : null,
      q.answer_type ? el("span", { class: "pill", text: humanize(q.answer_type) + (LENGTH[q.answer_length] ? ` · ${LENGTH[q.answer_length]}` : "") }) : null,
      q.difficulty ? el("span", { class: "pill", text: q.difficulty === "very_hard" ? "Very hard" : humanize(q.difficulty) }) : null,
      qa && qa.edited ? el("span", { class: "pill warn", text: "Edited" }) : null);

    // question block
    let questionBlock;
    if (editing === "question") {
      questionBlock = editForm(sid, qid, q, "question", fin);
    } else {
      const ci = correctOptionIndex(fin.options, fin.answer);
      questionBlock = el("div", null,
        q.family ? el("div", { class: "q-family", text: prettyFamily(q.family) }) : null,
        el("div", { class: "q-text", text: fin.question }),
        qEdited && fin.question !== q.question ? el("div", { class: "orig-line" }, el("b", { text: "Original question: " }), q.question || "") : null,
        fin.options && fin.options.length ? [el("div", { class: "label", text: "Options" }),
          el("ul", { class: "options" }, fin.options.map((o, j) => el("li", { class: j === ci ? "correct" : "", text: o })))] : null,
        qa && qa.options_edit ? el("div", { class: "orig-line" }, el("b", { text: "Original options: " }), (q.options || []).join("  |  ")) : null);
    }

    // answer block
    let answerBlock;
    if (editing === "answer") {
      answerBlock = editForm(sid, qid, q, "answer", fin);
    } else {
      answerBlock = el("div", null,
        el("div", { class: "label", text: aEdited ? "Answer (edited)" : "GPT answer" }),
        el("div", { class: "answer", text: fin.answer }),
        aEdited ? el("div", { class: "orig-line" }, el("b", { text: "Original answer: " }), q.answer || "") : null);
    }

    const ev = (q.evidence_timestamps || []).filter((e) => e && e.start_s !== undefined);
    const evidence = ev.length ? [el("div", { class: "label", text: "Evidence in the video" }),
      el("div", { class: "evidence" }, ev.map((e) => el("button", {
        class: "ev-chip", type: "button", title: "Jump to this moment", onclick: () => seekTo(e.start_s),
      }, el("span", { class: "t", text: `▶ ${fmtTime(e.start_s)}–${fmtTime(e.end_s)}` }), e.observation || "")))] : null;

    const reasoning = el("details", { class: "reason" },
      el("summary", { text: "Reasoning and why it's hard" }),
      el("div", { class: "reason-body" },
        q.answer_rationale ? [el("b", { text: "Reasoning: " }), q.answer_rationale, "\n\n"] : null,
        q.why_hard ? [el("b", { text: "Why it's hard: " }), q.why_hard, "\n\n"] : null,
        q.why_a_surgeon_cares ? [el("b", { text: "Why a surgeon cares: " }), q.why_a_surgeon_cares, "\n\n"] : null,
        q.likely_agent_failure ? [el("b", { text: "Likely agent failure: " }), q.likely_agent_failure, "\n\n"] : null,
        q.clinical_use ? [el("b", { text: "Clinical use: " }), humanize(q.clinical_use), "\n"] : null,
        (q.agentic_skills || []).length ? [el("b", { text: "Skills needed: " }), q.agentic_skills.map(humanize).join(", "), "\n"] : null,
        (q.tool_plan || []).length ? [el("b", { text: "Agent steps: " }), q.tool_plan.map((s, j) => `${j + 1}. ${s}`).join("  "), "\n"] : null,
        q.confidence !== undefined ? [el("b", { text: "GPT confidence: " }), `${Math.round(100 * Number(q.confidence))}%`] : null));

    const corr = qa ? qa.correctness : "";
    const seg = el("div", { class: "seg-ctl", role: "group", "aria-label": "Is the answer correct?" },
      CORRECTNESS.map(([v, label]) => el("button", {
        type: "button", "data-v": v, class: corr === v ? "on" : "", "aria-pressed": String(corr === v),
        text: label,
        onclick: () => { const r = qAnn(sid, qid, true); r.correctness = r.correctness === v ? "" : v; touch(sid); renderCase(); },
      })));
    const review = el("div", { class: "q-review" },
      el("div", null, el("div", { class: "label", style: { marginTop: "0" }, text: "Is the answer correct?" }), seg),
      el("div", { class: "q-actions" },
        editing === "question" ? null : el("button", { class: "btn ghost small", type: "button", text: "✎ Edit question",
          onclick: () => { state.editing[qid] = "question"; renderCase(); } }),
        editing === "answer" ? null : el("button", { class: "btn ghost small", type: "button", text: "✎ Edit answer",
          onclick: () => { state.editing[qid] = "answer"; renderCase(); } })));
    const comment = el("textarea", {
      class: "comment", rows: 1, placeholder: "Note on this question (optional)", value: (qa && qa.comment) || "",
      onchange: (e) => { const r = qAnn(sid, qid, true); r.comment = e.target.value; touch(sid); },
    });

    return el("article", { class: `panel qcard${selected ? " selected" : ""}`, "data-qid": qid },
      el("div", { class: "q-top" }, tags, star), questionBlock, answerBlock, evidence, reasoning, review, comment);
  }

  function toggleSelect(sid, qid) {
    const r = qAnn(sid, qid, true);
    if (!r.selected && selectedCount(sid) >= CFG.SELECT_MAX) {
      toast(`You can mark up to ${CFG.SELECT_MAX} best question${CFG.SELECT_MAX === 1 ? "" : "s"} per video`);
      return;
    }
    r.selected = !r.selected;
    touch(sid);
    renderCase();
  }

  function renderCase() {
    const c = state.cases[state.current];
    const root = $("#case");
    if (!c) { root.replaceChildren(); return; }
    const doc = c.doc;
    const a = annFor(c.id);
    const st = statusOf(c.id);
    const nSel = selectedCount(c.id);
    const head = el("div", { class: "case-head" },
      el("div", null,
        el("h2", { text: doc.procedure || c.id }),
        el("div", { class: "meta-chips" },
          el("span", { class: "pill", text: doc.dataset_title || doc.dataset || "" }),
          doc.video && doc.video.duration_s ? el("span", { class: "pill", text: fmtDuration(doc.video.duration_s) }) : null,
          doc.procedure_category ? el("span", { class: "pill", text: humanize(doc.procedure_category) }) : null,
          el("span", { class: "pill", text: `Video ${state.current + 1} of ${state.cases.length}` }),
          el("span", { class: "pill time", id: "time-here", text: `⏱ ${fmtActive(state.activity.by_sample[c.id] || 0)} on this video` }),
          st === "done" ? el("span", { class: "pill good", text: "✓ Done" }) : st === "in_progress" ? el("span", { class: "pill star", text: "In progress" }) : null)),
      el("div", { class: "nav-btns" },
        el("button", { class: "btn ghost", type: "button", text: "← Previous", disabled: state.current === 0, onclick: () => openCase(state.current - 1) }),
        el("button", { class: "btn ghost", type: "button", text: "Next →", disabled: state.current >= state.cases.length - 1, onclick: () => openCase(state.current + 1) })));

    const questions = (doc.questions || []).map((q, k) => renderQuestion(c, q, k));
    const overall = el("textarea", {
      class: "comment", rows: 1, placeholder: "Comment on this video (optional): video quality, wrong labels, …",
      value: (a && a.comment) || "",
      onchange: (e) => { annFor(c.id, true).comment = e.target.value; touch(c.id); },
    });
    const doneBtn = el("button", {
      class: "btn primary lg", type: "button", text: st === "done" ? "Saved ✓  Next video →" : "Done & next →",
      onclick: () => markDone(c),
    });
    const msg = el("div", { class: "msg", text: nSel >= CFG.SELECT_MIN ? `${nSel} best question${nSel === 1 ? "" : "s"} marked. Progress saves automatically.`
      : `Mark at least ${CFG.SELECT_MIN} best question${CFG.SELECT_MIN === 1 ? "" : "s"} with ★ to finish this video.` });
    const doneBar = el("section", { class: "panel done-bar" }, overall, el("div", { style: { display: "grid", gap: "6px", justifyItems: "end" } }, doneBtn, msg));

    const hint = el("div", { class: "q-hint" },
      el("span", null, "Pick the ", el("b", { text: `best ${CFG.SELECT_MAX > 1 ? `1–${CFG.SELECT_MAX}` : "1"}` }), " question(s) with ★, check the answer, edit if needed."),
      el("span", { class: "pill star", text: `★ ${nSel} / ${CFG.SELECT_MAX}` }));

    // keep the same video player while working on the same video (re-creating it would restart playback)
    const existing = root.dataset.caseId === c.id ? root.querySelector(".video-panel") : null;
    const videoPanel = existing || renderVideoPanel(c);
    root.replaceChildren(el("div", { class: "case-inner" }, head,
      el("div", { class: "grid" }, videoPanel, el("div", { class: "q-col" }, hint, questions, doneBar))));
    root.dataset.caseId = c.id;
  }

  function markDone(c) {
    if (!requireReviewer()) return;
    if (selectedCount(c.id) < CFG.SELECT_MIN) {
      toast(`Mark at least ${CFG.SELECT_MIN} best question with ★ first`);
      const first = $(".star-btn");
      if (first) first.focus();
      return;
    }
    const a = annFor(c.id, true);
    a.status = "done";
    a.updated_at = nowIso();
    persist();
    updateProgress();
    const next = state.cases.findIndex((x, i) => i > state.current && statusOf(x.id) !== "done");
    const any = next >= 0 ? next : state.cases.findIndex((x) => statusOf(x.id) !== "done");
    if (any >= 0) { openCase(any); toast("Saved. Next video"); }
    else { renderCase(); renderList(); toast("All videos done. Use Export to download your review.", 4200); }
  }

  // ------------------------------------------------------------------ save / export / import
  function buildExport() {
    const samples = {};
    const byId = new Map(state.cases.map((c) => [c.id, c]));
    for (const [sid, a] of Object.entries(state.ann)) {
      const c = byId.get(sid);
      if (!c) {   // a saved review for a video not loaded in this session: keep it as it is
        samples[sid] = a;
        continue;
      }
      const qs = {};
      (c.doc.questions || []).forEach((q, k) => {
        const qid = q.qid || `q${k + 1}`;
        const r = a.questions[qid] || {};
        qs[qid] = {
          selected: !!r.selected, correctness: r.correctness || "", edited: !!(r.question_edit || r.answer_edit || r.options_edit),
          comment: r.comment || "", question_edit: r.question_edit || "", answer_edit: r.answer_edit || "",
          options_edit: r.options_edit || null, relevance: null, difficulty: null, agentic: null, clarity: null,
          original: {
            question: q.question || "", answer: q.answer || "", options: q.options || [], category: q.category || "",
            sa_level: q.sa_level || "", family: q.family || "", clinical_use: q.clinical_use || "",
            answer_type: q.answer_type || "", answer_length: q.answer_length || "", difficulty: q.difficulty || "",
            answer_rationale: q.answer_rationale || "", evidence_timestamps: q.evidence_timestamps || [],
            why_a_surgeon_cares: q.why_a_surgeon_cares || "", likely_agent_failure: q.likely_agent_failure || "",
            confidence: q.confidence ?? null,
          },
          final: finalOf(q, r),
        };
      });
      samples[sid] = {
        time_spent_s: Number(state.activity.by_sample[sid]) || 0,
        status: a.status, updated_at: a.updated_at, batch: c.doc.batch ?? null, dataset: c.doc.dataset || "",
        procedure: c.doc.procedure || "", comment: a.comment || "", flags: a.flags || [],
        selected_qids: Object.keys(qs).filter((k) => qs[k].selected), questions: qs,
      };
    }
    return {
      ui_version: UI_VERSION, tool: TOOL, annotator: state.reviewer, exported_at: nowIso(),
      n_loaded: state.cases.length, n_done: Object.values(samples).filter((s) => s.status === "done").length,
      activity: {
        total_active_s: totalActive(), n_sessions: state.activity.sessions.length,
        by_sample: state.activity.by_sample, sessions: state.activity.sessions,
      },
      samples,
    };
  }

  function download(name, text, type) {
    const blob = new Blob([text], { type });
    const url = URL.createObjectURL(blob);
    const a = el("a", { href: url, download: name });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  async function saveProgressNow() {
    if (!requireReviewer()) return;
    persist();
    clearTimeout(folderTimer);
    if (state.dirHandle && state.canWrite && await writeProgressToFolder()) {
      toast(`Progress saved in your batch folder (${progressFileName()}). It loads automatically next time.`, 4200);
      return;
    }
    download(progressFileName(), JSON.stringify(buildExport(), null, 2), "application/json");
    state.lastSaved = { where: "as file", at: new Date() };
    setSaveStatus();
    toast("Progress file downloaded. Put it in your batch folder: it loads automatically next time.", 5200);
  }

  function exportJson() {
    if (!requireReviewer()) return;
    const data = buildExport();
    if (!Object.keys(data.samples).length) { toast("Nothing reviewed yet"); return; }
    download(`catagent_review_${safeName(state.reviewer)}_${stamp()}.json`, JSON.stringify(data, null, 2), "application/json");
    toast(`Exported ${Object.keys(data.samples).length} video(s)`);
  }

  function csvCell(v) {
    const s = Array.isArray(v) ? v.join(" | ") : v === null || v === undefined ? "" : String(v);
    return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  }
  function exportCsv() {
    if (!requireReviewer()) return;
    const data = buildExport();
    const cols = ["annotator", "sample_id", "dataset", "procedure", "status", "qid", "sa_level", "family", "category",
      "selected", "correctness", "edited", "original_question", "final_question", "original_answer", "final_answer",
      "original_options", "final_options", "question_comment", "video_comment", "time_spent_s", "updated_at"];
    const rows = [cols.join(",")];
    for (const [sid, s] of Object.entries(data.samples)) {
      for (const [qid, q] of Object.entries(s.questions || {})) {
        const o = q.original || {}, f = q.final || {};
        rows.push([data.annotator, sid, s.dataset, s.procedure, s.status, qid, o.sa_level, o.family, o.category,
          q.selected ? "yes" : "no", q.correctness, q.edited ? "yes" : "no", o.question, f.question, o.answer, f.answer,
          o.options, f.options, q.comment, s.comment, s.time_spent_s ?? "", s.updated_at].map(csvCell).join(","));
      }
    }
    if (rows.length === 1) { toast("Nothing reviewed yet"); return; }
    download(`catagent_review_${safeName(state.reviewer)}_${stamp()}.csv`, "﻿" + rows.join("\r\n"), "text/csv");
    toast(`Exported ${rows.length - 1} question rows`);
  }

  async function importJson(file) {
    try {
      const data = JSON.parse(await file.text());
      if (!isProgressFile(data)) throw new Error("not a progress file");
      const owner = String(data.annotator || "").trim();
      if (!state.reviewer && owner) setReviewer(owner);
      if (!requireReviewer()) return;
      if (owner && owner.toLowerCase() !== state.reviewer.toLowerCase() &&
          !window.confirm(`This progress file belongs to "${owner}". Load it into your review as "${state.reviewer}"?`)) return;
      const n = mergeAnn(data.samples);
      mergeActivity(data.activity);
      saveActivity();
      persist();
      if (state.cases.length) { renderCase(); renderList(); updateProgress(); }
      toast(`Loaded ${n} saved video review(s)${state.cases.length ? "" : ". Now open your batch folder."}`, 3800);
    } catch (e) {
      toast("That file is not a CAT-Agent progress or export file");
    }
  }

  // ------------------------------------------------------------------ wiring
  function updateStartButtons() {
    const ok = !!state.reviewer;
    for (const id of ["#btn-folder", "#btn-files"]) { const b = $(id); if (b) b.disabled = !ok; }
    const hint = $("#start-hint");
    if (hint) hint.textContent = ok ? `Reviewing as ${state.reviewer}. Your progress is saved under this name.`
      : "Enter your name or ID to start. Nothing is saved without it.";
  }

  function init() {
    try { state.reviewer = (localStorage.getItem("catagent.reviewer") || "").trim(); } catch (e) { /* ignore */ }
    loadStore();
    loadActivity();
    const markInput = () => { state.lastInput = Date.now(); };
    for (const ev of ["mousemove", "mousedown", "keydown", "wheel", "touchstart", "input"]) {
      document.addEventListener(ev, markInput, { passive: true, capture: true });
    }
    setInterval(tick, TICK_S * 1000);
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "hidden" && state.reviewer && state.cases.length) { saveActivity(); persist(); }
    });
    for (const input of document.querySelectorAll(".reviewer-input")) {
      input.value = state.reviewer;
      input.addEventListener("input", () => {
        input.classList.remove("needs");
        if (input.id === "reviewer-start") { // live-enable the start buttons while typing
          const b = !!input.value.trim();
          for (const id of ["#btn-folder", "#btn-files"]) { const x = $(id); if (x) x.disabled = !b; }
        }
      });
      input.addEventListener("change", () => {
        setReviewer(input.value);
        if (state.cases.length) { renderCase(); renderList(); updateProgress(); }
        if (state.reviewer) toast(`Saving progress as ${state.reviewer}`);
      });
      input.addEventListener("keydown", (e) => { if (e.key === "Enter") input.blur(); });
    }
    updateStartButtons();

    const inFolder = $("#input-folder"), inFiles = $("#input-files"), inImport = $("#input-import");
    const commitStartName = () => { const v = $("#reviewer-start"); if (v && v.value.trim() && v.value.trim() !== state.reviewer) setReviewer(v.value); };
    $("#btn-folder").addEventListener("click", () => { commitStartName(); openFolder(); });
    $("#btn-files").addEventListener("click", () => { commitStartName(); if (requireReviewer()) { state.dirHandle = null; state.canWrite = false; inFiles.click(); } });
    $("#btn-open").addEventListener("click", () => openFolder());
    $("#btn-save").addEventListener("click", () => saveProgressNow());
    inFolder.addEventListener("change", () => { state.dirHandle = null; state.canWrite = false; loadFiles(inFolder.files); inFolder.value = ""; });
    inFiles.addEventListener("change", () => { loadFiles(inFiles.files); inFiles.value = ""; });
    inImport.addEventListener("change", () => { if (inImport.files[0]) importJson(inImport.files[0]); inImport.value = ""; });

    const dz = $("#dropzone");
    ["dragenter", "dragover"].forEach((ev) => document.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => document.addEventListener(ev, (e) => {
      if (ev === "dragleave" && e.relatedTarget) return;
      dz.classList.remove("over");
    }));
    document.addEventListener("drop", (e) => {
      e.preventDefault();
      dz.classList.remove("over");
      commitStartName();
      onDrop(e.dataTransfer);
    });
    dz.addEventListener("keydown", (e) => { if ((e.key === "Enter" || e.key === " ") && e.target === dz) { e.preventDefault(); commitStartName(); openFolder(); } });

    const exBtn = $("#btn-export"), menu = $("#export-menu");
    exBtn.disabled = false;
    exBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      const open = menu.classList.toggle("hidden") === false;
      exBtn.setAttribute("aria-expanded", String(open));
    });
    document.addEventListener("click", () => { menu.classList.add("hidden"); exBtn.setAttribute("aria-expanded", "false"); });
    menu.addEventListener("click", (e) => {
      const b = e.target.closest("[data-export]");
      if (!b) return;
      const kind = b.getAttribute("data-export");
      if (kind === "json") exportJson();
      else if (kind === "csv") exportCsv();
      else inImport.click();
    });

    $("#search").addEventListener("input", (e) => { state.search = e.target.value; renderList(); });
    document.querySelectorAll("[data-filter]").forEach((b) => b.addEventListener("click", () => {
      state.filter = b.getAttribute("data-filter");
      document.querySelectorAll("[data-filter]").forEach((x) => x.classList.toggle("active", x === b));
      renderList();
    }));

    document.addEventListener("keydown", (e) => {
      const tag = (e.target && e.target.tagName) || "";
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") { e.preventDefault(); if (state.cases.length) saveProgressNow(); return; }
      if (["INPUT", "TEXTAREA", "SELECT"].includes(tag) || e.metaKey || e.ctrlKey || e.altKey) return;
      if (!state.cases.length) return;
      if (e.key === "ArrowRight" || e.key === "j") openCase(state.current + 1);
      else if (e.key === "ArrowLeft" || e.key === "k") openCase(state.current - 1);
      else if (["1", "2", "3"].includes(e.key)) {
        const c = state.cases[state.current];
        const q = c && (c.doc.questions || [])[Number(e.key) - 1];
        if (q) toggleSelect(c.id, q.qid || `q${e.key}`);
      }
    });

    window.addEventListener("beforeunload", (e) => {
      if (state.unsaved) { e.preventDefault(); e.returnValue = ""; }
      else if (state.dirHandle && state.canWrite && folderTimer) writeProgressToFolder();
    });
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { fmtTime, fmtDuration, stem, correctOptionIndex, csvCell, humanize, safeName, prettyFamily };
  } else {
    document.addEventListener("DOMContentLoaded", init);
  }
})();
