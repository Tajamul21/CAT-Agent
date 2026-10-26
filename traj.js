/* CAT-Agent trajectory review (vanilla JS, no build step, no network calls).
 *
 * The annotator enters a name/ID and opens a trajectory folder laid out like
 * catagent_cvpr/opencompass/data/rhexis_trajectories:
 *   samples_open/<key>.json   one GTA sample {"<key>": {tools, files, dialogs, gt_answer}}
 *   clips/<case>.mp4          the video named in files[0]
 *   frames_1fps/<case>/...    frames used as tool inputs (shown with their bbox)
 *   trajectories_open.jsonl   optional run log: vqid and GPT's own concerns
 *   source_open.json          optional: vqid and duration per key
 * For each trajectory the annotator marks the question, every thought, tool call and tool output as
 * correct / flag / incorrect (with a note), and the final answer as correct / partly / incorrect / can't tell.
 *
 * Saving works as in the question review app: browser storage, plus a progress file
 * (catagent_traj_progress_<id>.json) written into the folder in Chrome/Edge and reloaded from it.
 */
(function () {
  "use strict";

  const UI_VERSION = 1;
  const TOOL = "CAT-Agent trajectory review";
  const VIDEO_EXT = ["mp4", "m4v", "mov", "webm", "mkv", "ogv"];
  const IMAGE_EXT = ["jpg", "jpeg", "png", "webp"];
  const SKIP_DIRS = ["old_runs", "__pycache__", "prompts", "examples", "node_modules"];
  const ITEM_VERDICTS = [["ok", "✓ Correct"], ["flag", "⚑ Flag"], ["wrong", "✗ Incorrect"]];
  const FINAL_VERDICTS = [
    ["correct", "Correct"], ["partial", "Partly"], ["incorrect", "Incorrect"], ["cannot_verify", "Can't tell"],
  ];
  const PARTS = [["thought", "Thought"], ["call", "Tool call"], ["output", "Tool output"]];
  const FOLDER_SAVE_DELAY_MS = 1500;
  const TICK_S = 5;
  const IDLE_MS = 120000;

  // ------------------------------------------------------------------ state
  const state = {
    reviewer: "",
    cases: [],          // [{id, sample, traj, video: File|null, meta}]
    frames: new Map(),  // "case_xxxx/frame_nnn.jpg" -> File
    current: -1,
    filter: "all",
    search: "",
    ann: {},            // key -> annotation record
    videoUrl: null,
    imageUrls: [],
    dirHandle: null,
    canWrite: false,
    unsaved: false,
    lastSaved: null,
    activity: { by_sample: {} },
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
    const m = Math.floor(t / 60), s = Math.floor(t % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  }
  function fmtActive(sec) {
    sec = Math.round(Number(sec) || 0);
    if (sec < 60) return `${sec} s`;
    const h = Math.floor(sec / 3600), m = Math.round((sec % 3600) / 60);
    return h ? `${h} h ${String(m).padStart(2, "0")} min` : `${m} min`;
  }
  const nowIso = () => new Date().toISOString();
  const clock = (d) => d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const base = (p) => String(p || "").split("/").pop();
  const stem = (p) => base(p).replace(/\.[^.]+$/, "");
  const ext = (name) => (String(name).split(".").pop() || "").toLowerCase();
  const frameKey = (p) => String(p || "").split("/").filter(Boolean).slice(-2).join("/").toLowerCase();
  const safeName = (s) => String(s || "reviewer").trim().replace(/[^A-Za-z0-9_.-]+/g, "_");
  const stamp = () => new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-");

  let toastTimer = null;
  function toast(msg, ms = 2600) {
    const t = $("#toast");
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove("show"), ms);
  }

  /** Text with every "12 s", "25-37s" or "34–37 s" turned into a button that seeks the video. */
  function withTimes(text) {
    const out = [];
    const re = /(\d{1,4}(?:\.\d+)?)(?:\s*[–-]\s*(\d{1,4}(?:\.\d+)?))?\s?s\b/g;
    let last = 0, m;
    text = String(text ?? "");
    while ((m = re.exec(text))) {
      if (m.index > last) out.push(text.slice(last, m.index));
      const t = Number(m[1]);
      out.push(el("button", { class: "t-link", type: "button", title: `Jump to ${fmtTime(t)}`, text: m[0],
        onclick: () => seekTo(t) }));
      last = re.lastIndex;
    }
    if (last < text.length) out.push(text.slice(last));
    return out;
  }

  // ------------------------------------------------------------------ trajectory parsing
  /** GTA dialogs -> {question, steps: [{thought, calls, outputs}], final}. */
  function parseDialogs(dialogs) {
    const traj = { question: "", steps: [], final: null };
    let cur = null;
    for (const m of dialogs || []) {
      if (m.role === "user" && !traj.question) traj.question = String(m.content || "");
      else if (m.role === "assistant" && (m.tool_calls || []).length) {
        cur = { thought: m.thought || "", calls: m.tool_calls.map((c) => ({
          name: (c.function && c.function.name) || c.name || "?",
          args: (c.function && c.function.arguments) || c.arguments || {},
        })), outputs: [] };
        if (typeof cur.calls[0].args === "string") {
          try { cur.calls.forEach((c) => { c.args = typeof c.args === "string" ? JSON.parse(c.args) : c.args; }); } catch (e) { /* keep text */ }
        }
        traj.steps.push(cur);
      } else if (m.role === "tool") {
        if (cur) cur.outputs.push({ name: m.name || "", content: m.content });
      } else if (m.role === "assistant") {
        traj.final = { thought: m.thought || "", answer: String(m.content ?? "") };
      }
    }
    return traj;
  }
  const isSample = (v) => v && typeof v === "object" && Array.isArray(v.dialogs) &&
    v.dialogs.some((m) => m && m.role === "assistant");
  function goldOf(gt) {
    if (Array.isArray(gt)) return { accepted: gt.map(String), rejected: [] };
    if (gt && typeof gt === "object") {
      const flat = (x) => (x || []).map((a) => (Array.isArray(a) ? a.join(" / ") : String(a)));
      return { accepted: flat(gt.whitelist), rejected: flat(gt.blacklist) };
    }
    return { accepted: gt ? [String(gt)] : [], rejected: [] };
  }

  // ------------------------------------------------------------------ storage (only with a name/ID)
  const storeKey = () => `catagent.traj.v1.${state.reviewer.toLowerCase()}`;
  const activityKey = () => `catagent.traj.activity.v1.${state.reviewer.toLowerCase()}`;
  const progressFileName = () => `catagent_traj_progress_${safeName(state.reviewer)}.json`;
  let folderTimer = null;

  const blankItem = () => ({ v: "", note: "" });
  function normItem(x) { return { v: (x && x.v) || "", note: (x && x.note) || "" }; }
  function normRecord(s) {
    const steps = {};
    for (const [k, st] of Object.entries((s && s.steps) || {})) {
      steps[k] = { thought: normItem(st.thought), call: normItem(st.call), output: normItem(st.output) };
    }
    return { status: (s && s.status) || "in_progress", updated_at: (s && s.updated_at) || nowIso(),
      comment: (s && s.comment) || "", question: normItem(s && s.question), final: normItem(s && s.final), steps };
  }
  /** Merge records into state.ann; the newer `updated_at` wins. Returns the number taken. */
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
  function mergeActivity(a) {
    for (const [sid, sec] of Object.entries((a && a.by_sample) || {})) {
      state.activity.by_sample[sid] = Math.max(Number(state.activity.by_sample[sid]) || 0, Number(sec) || 0);
    }
  }
  function loadStore() {
    state.ann = {};
    state.activity = { by_sample: {} };
    if (!state.reviewer) return;
    try {
      const raw = localStorage.getItem(storeKey());
      if (raw) mergeAnn(JSON.parse(raw));
      const act = localStorage.getItem(activityKey());
      if (act) mergeActivity(JSON.parse(act));
    } catch (e) { /* unreadable store: start empty */ }
  }
  function saveActivity() {
    if (!state.reviewer) return;
    try { localStorage.setItem(activityKey(), JSON.stringify(state.activity)); } catch (e) { /* ignore */ }
  }
  const totalActive = () => Object.values(state.activity.by_sample).reduce((a, b) => a + (Number(b) || 0), 0);
  function tick() {
    if (!state.reviewer || state.current < 0 || document.visibilityState !== "visible") return;
    const v = $("#player");
    if (!(v && !v.paused && !v.ended) && Date.now() - state.lastInput > IDLE_MS) return;
    const sid = state.cases[state.current].id;
    state.activity.by_sample[sid] = (Number(state.activity.by_sample[sid]) || 0) + TICK_S;
    saveActivity();
    updateTimeLabels();
  }
  function updateTimeLabels() {
    const tot = $("#active-time");
    if (tot) tot.textContent = state.reviewer ? `Active time ${fmtActive(totalActive())}` : "";
    const here = $("#time-here");
    const c = state.cases[state.current];
    if (here && c) here.textContent = `⏱ ${fmtActive(state.activity.by_sample[c.id] || 0)} here`;
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
      s.textContent = state.canWrite ? "Saves to this browser and your folder" : "Saves to this browser";
    }
  }
  async function writeProgressToFolder() {
    folderTimer = null;
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
      state.canWrite = false;
      setSaveStatus();
      return false;
    }
  }
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
    const carry = state.reviewer ? {} : state.ann;
    state.reviewer = next;
    try { localStorage.setItem("catagent.reviewer", next); } catch (e) { /* ignore */ }
    loadStore();
    if (next) { mergeAnn(carry); if (Object.keys(carry).length) persist(); } else state.ann = carry;
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
    if (!state.ann[sid] && create) state.ann[sid] = normRecord(null);
    return state.ann[sid] || null;
  }
  /** The {v, note} record of one item: "question", "final" or ["step", n, part]. */
  function itemOf(sid, ref, create = false) {
    const a = annFor(sid, create);
    if (!a) return null;
    if (typeof ref === "string") return a[ref];
    const [, n, part] = ref;
    if (!a.steps[n] && create) a.steps[n] = { thought: blankItem(), call: blankItem(), output: blankItem() };
    return a.steps[n] ? a.steps[n][part] : null;
  }
  function allRefs(c) {
    const refs = ["question"];
    c.traj.steps.forEach((s, i) => PARTS.forEach(([p]) => refs.push(["step", String(i + 1), p])));
    refs.push("final");
    return refs;
  }
  function counts(c) {
    const out = { total: 0, rated: 0, flag: 0, wrong: 0 };
    for (const r of allRefs(c)) {
      const it = itemOf(c.id, r);
      out.total += 1;
      if (it && it.v) out.rated += 1;
      if (it && it.v === "flag") out.flag += 1;
      if (it && ["wrong", "incorrect", "partial"].includes(it.v)) out.wrong += 1;
      if (it && it.v === "cannot_verify") out.flag += 1;
    }
    return out;
  }
  function touch(sid) {
    const a = annFor(sid, true);
    a.updated_at = nowIso();
    if (a.status !== "done") a.status = "in_progress";
    persist();
    renderList();
    updateProgress();
  }
  const statusOf = (sid) => (state.ann[sid] && state.ann[sid].status) || "todo";

  // ------------------------------------------------------------------ opening the folder
  async function walkHandle(dir, prefix, out) {
    for await (const [name, h] of dir.entries()) {
      if (name.startsWith(".")) continue;
      if (h.kind === "directory") {
        if (!SKIP_DIRS.includes(name)) await walkHandle(h, `${prefix}${name}/`, out);
      } else {
        const f = await h.getFile();
        out.push({ path: prefix + name, file: f });
      }
    }
    return out;
  }
  async function walkEntry(entry, prefix, out) {
    if (entry.isFile) {
      await new Promise((res) => entry.file((f) => { out.push({ path: prefix + entry.name, file: f }); res(); }, () => res()));
    } else if (entry.isDirectory && !SKIP_DIRS.includes(entry.name) && !entry.name.startsWith(".")) {
      const reader = entry.createReader();
      let batch;
      do {
        batch = await new Promise((res) => reader.readEntries(res, () => res([])));
        for (const e of batch) await walkEntry(e, `${prefix}${entry.name}/`, out);
      } while (batch.length);
    }
    return out;
  }
  /** <input webkitdirectory> files -> [{path, file}] with the chosen folder's own name removed. */
  function fromInput(files) {
    return Array.from(files || []).map((f) => {
      const rel = (f.webkitRelativePath || f.name).split("/");
      if (rel.length > 1) rel.shift();
      return { path: rel.join("/"), file: f };
    }).filter((x) => !x.path.split("/").some((seg, i, all) => i < all.length - 1 && (SKIP_DIRS.includes(seg) || seg.startsWith("."))));
  }

  async function openFolder() {
    if (!requireReviewer()) return;
    if (typeof window.showDirectoryPicker === "function") {
      try {
        const dir = await window.showDirectoryPicker({ id: "catagent-traj", mode: "readwrite" });
        state.dirHandle = dir;
        state.canWrite = true;
        await loadEntries(await walkHandle(dir, "", []));
        return;
      } catch (e) {
        if (e && e.name === "AbortError") return;
        state.dirHandle = null;
        state.canWrite = false;
      }
    }
    $("#input-folder").click();
  }

  async function onDrop(dt) {
    if (!requireReviewer()) return;
    const items = dt.items ? Array.from(dt.items) : [];
    const handlePromises = items.map((it) => (typeof it.getAsFileSystemHandle === "function" ? it.getAsFileSystemHandle() : null));
    const entries = items.map((it) => (it.webkitGetAsEntry ? it.webkitGetAsEntry() : null)).filter(Boolean);
    try {
      const handles = (await Promise.all(handlePromises.filter(Boolean))).filter(Boolean);
      const dir = handles.find((h) => h.kind === "directory");
      if (dir) {
        let granted = false;
        try { granted = (await dir.requestPermission({ mode: "readwrite" })) === "granted"; } catch (e) { granted = false; }
        state.dirHandle = granted ? dir : null;
        state.canWrite = granted;
        await loadEntries(await walkHandle(dir, "", []));
        return;
      }
    } catch (e) { /* fall back to the entry API below */ }
    state.dirHandle = null;
    state.canWrite = false;
    const out = [];
    for (const e of entries) {
      if (e.isDirectory) {   // the dropped folder itself is the root: walk its children
        const reader = e.createReader();
        let batch;
        do {
          batch = await new Promise((res) => reader.readEntries(res, () => res([])));
          for (const x of batch) await walkEntry(x, "", out);
        } while (batch.length);
      } else await walkEntry(e, "", out);
    }
    await loadEntries(out);
  }

  const isProgressFile = (d) => d && typeof d === "object" && d.tool === TOOL && d.samples && "annotator" in d;

  async function loadEntries(entries) {
    if (!entries.length) return;
    const videos = new Map();
    const samples = new Map();     // key -> {sample, fromSamplesDir}
    const meta = {};               // key -> {vqid, duration, concerns}
    const progress = [];
    let badJson = 0, otherReviewers = 0;
    const addMeta = (k, m) => { meta[k] = Object.assign(meta[k] || {}, m); };
    for (const { path, file } of entries) {
      const name = base(path), e = ext(name);
      if (VIDEO_EXT.includes(e)) { videos.set(stem(name).toLowerCase(), file); continue; }
      if (IMAGE_EXT.includes(e)) { state.frames.set(frameKey(path), file); continue; }
      if (e === "jsonl" && /^trajectories/i.test(name)) {
        for (const line of (await file.text()).split("\n")) {
          if (!line.trim()) continue;
          try {
            const r = JSON.parse(line);
            if (r && r.key !== undefined) addMeta(String(r.key), { vqid: r.vqid, concerns: r.concerns || [], run_status: r.status });
          } catch (err) { /* skip a broken line */ }
        }
        continue;
      }
      if (e !== "json") continue;
      try {
        const data = JSON.parse(await file.text());
        if (isProgressFile(data)) {
          if (String(data.annotator || "").trim().toLowerCase() === state.reviewer.toLowerCase()) progress.push(data);
          else otherReviewers += 1;
          continue;
        }
        if (!data || typeof data !== "object" || Array.isArray(data)) continue;
        const inSamples = path.split("/").includes("samples_open");
        for (const [k, v] of Object.entries(data)) {
          if (isSample(v)) {
            const prev = samples.get(k);
            if (!prev || inSamples || !prev.inSamples) samples.set(k, { sample: v, inSamples });
          } else if (v && typeof v === "object" && (v.vqid || v.duration)) {
            addMeta(k, { vqid: v.vqid, duration: v.duration });
          }
        }
      } catch (err) { badJson += 1; }
    }
    // samples_open/ is the source of truth when present (dataset*.json may hold other splits)
    const anyInSamples = Array.from(samples.values()).some((s) => s.inSamples);
    let restored = 0;
    for (const p of progress) { restored += mergeAnn(p.samples); mergeActivity(p.activity); }
    if (progress.length) saveActivity();
    if (restored) persist();
    if (!samples.size) {
      toast("No trajectories found. Open the folder that contains samples_open/ (or the .json samples).", 4200);
      return;
    }
    const byId = new Map(state.cases.map((c) => [c.id, c]));
    for (const [k, { sample, inSamples }] of samples) {
      if (anyInSamples && !inSamples) continue;
      const vfile = (sample.files || []).find((f) => f.type === "video") || {};
      const m = Object.assign({}, (byId.get(k) && byId.get(k).meta) || {}, meta[k] || {});
      byId.set(k, { id: k, sample, traj: parseDialogs(sample.dialogs), videoPath: vfile.path || "",
        video: videos.get(stem(vfile.path).toLowerCase()) || (byId.get(k) && byId.get(k).video) || null, meta: m });
    }
    state.cases = Array.from(byId.values()).sort((a, b) => (Number(a.id) - Number(b.id)) || a.id.localeCompare(b.id));
    const withVideo = state.cases.filter((c) => c.video).length;
    showWorkspace();
    const firstTodo = state.cases.findIndex((c) => statusOf(c.id) !== "done");
    openCase(firstTodo >= 0 ? firstTodo : 0);
    const done = state.cases.filter((c) => statusOf(c.id) === "done").length;
    toast(`Loaded ${state.cases.length} trajector${state.cases.length === 1 ? "y" : "ies"}` +
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
      if (state.filter === "flagged") { const n = counts(c); if (!n.flag && !n.wrong) return false; }
      if (!q) return true;
      return `${c.id} ${c.meta.vqid || ""} ${c.traj.question} ${c.videoPath}`.toLowerCase().includes(q);
    });
  }
  function renderList() {
    $("#case-list").replaceChildren(...visibleCases().map(({ c, i }) => {
      const st = statusOf(c.id);
      const n = counts(c);
      const badge = n.wrong ? el("span", { class: "case-flag bad", text: `✗${n.wrong}` })
        : n.flag ? el("span", { class: "case-flag", text: `⚑${n.flag}` }) : el("span", { class: "case-num", text: `#${c.id}` });
      return el("li", {
        class: `case-item${i === state.current ? " active" : ""}`, tabindex: 0, title: c.traj.question,
        onclick: () => openCase(i), onkeydown: (e) => { if (e.key === "Enter") openCase(i); },
      },
      el("span", { class: `dot${st === "done" ? " done" : st === "in_progress" ? " progress-dot" : ""}`, "aria-label": st }),
      el("div", null,
        el("div", { class: "case-title", text: c.traj.question || c.id }),
        el("div", { class: "case-sub", text: [stem(c.videoPath), `${c.traj.steps.length} step${c.traj.steps.length === 1 ? "" : "s"}`,
          n.rated && st !== "done" ? `${n.rated}/${n.total} rated` : ""].filter(Boolean).join(" · ") })),
      badge);
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
    renderCase();
    renderList();
    const active = $(".case-item.active");
    if (active) active.scrollIntoView({ block: "nearest" });
    window.scrollTo(0, 0);
  }
  let stopAt = null;   // end of the window started with "play window"; cleared by any other seek or pause
  function seekTo(t) {
    const v = $("#player");
    if (!v) return;
    stopAt = null;
    v.currentTime = Math.max(0, Number(t) || 0);
    v.play().catch(() => {});
  }
  function seekRange(a, b) {
    seekTo(a);
    if (b !== undefined && b !== null && !isNaN(Number(b))) stopAt = Number(b);
  }

  function renderVideoPanel(c) {
    if (state.videoUrl) { URL.revokeObjectURL(state.videoUrl); state.videoUrl = null; }
    let media;
    if (c.video) {
      state.videoUrl = URL.createObjectURL(c.video);
      media = el("video", { id: "player", src: state.videoUrl, controls: true, preload: "metadata", playsinline: true });
    } else {
      media = el("div", { class: "video-missing" }, el("b", { text: "Video file not loaded" }),
        `Add ${c.videoPath || "the clip"} to the folder (Open folder, top right).`);
    }
    const now = el("div", { class: "phase-now", text: "Times in the trajectory are buttons: click one to jump there." });
    if (media.tagName === "VIDEO") {
      media.addEventListener("pause", () => { stopAt = null; });
      media.addEventListener("timeupdate", () => {
        if (stopAt !== null && media.currentTime >= stopAt) media.pause();
        now.replaceChildren(el("span", { text: "Now at " }), el("b", { text: `${media.currentTime.toFixed(1)} s` }),
          el("span", { text: ` (frame_${String(Math.floor(media.currentTime)).padStart(3, "0")})` }));
      });
    }
    const m = c.meta;
    const about = el("details", { class: "about" },
      el("summary", { text: "About this item" }),
      el("div", { class: "about-body" }, el("dl", { class: "kv" },
        el("dt", { text: "Key" }), el("dd", { text: c.id }),
        m.vqid ? [el("dt", { text: "VQID" }), el("dd", { text: m.vqid })] : null,
        el("dt", { text: "Video" }), el("dd", { text: c.videoPath || "?" }),
        m.duration ? [el("dt", { text: "Duration" }), el("dd", { text: `${m.duration} s` })] : null,
        el("dt", { text: "Tools offered" }), el("dd", { text: (c.sample.tools || []).map((t) => t.name).join(", ") || "?" }))));
    return el("section", { class: "panel video-panel" }, el("div", { class: "video-box" }, media), now, about);
  }

  /** A frame thumbnail with an optional bbox drawn on top; click to enlarge. */
  function frameView(path, bbox) {
    const f = state.frames.get(frameKey(path));
    if (!f) return el("div", { class: "frame-missing", text: `${base(path)} not loaded` });
    const url = URL.createObjectURL(f);
    state.imageUrls.push(url);
    const box = parseBbox(bbox);
    const img = el("img", { src: url, alt: base(path) });
    const wrap = el("button", { class: "frame", type: "button", title: "Click to enlarge", onclick: () => lightbox(url, box, path) }, img);
    if (box) img.addEventListener("load", () => wrap.append(bboxEl(box, img.naturalWidth, img.naturalHeight)), { once: true });
    return el("figure", { class: "frame-fig" }, wrap, el("figcaption", { text: base(path) + (box ? `  ·  bbox ${bbox}` : "") }));
  }
  function parseBbox(b) {
    const n = String(b || "").match(/-?\d+(?:\.\d+)?/g);
    return n && n.length === 4 ? n.map(Number) : null;
  }
  function bboxEl([x1, y1, x2, y2], w, h) {
    return el("span", { class: "bbox", style: {
      left: `${(100 * x1) / w}%`, top: `${(100 * y1) / h}%`, width: `${(100 * (x2 - x1)) / w}%`, height: `${(100 * (y2 - y1)) / h}%`,
    } });
  }
  function lightbox(url, box, path) {
    const img = el("img", { src: url, alt: base(path) });
    const frame = el("div", { class: "lb-frame" }, img);
    if (box) img.addEventListener("load", () => frame.append(bboxEl(box, img.naturalWidth, img.naturalHeight)), { once: true });
    const lb = el("div", { class: "lightbox", role: "dialog", "aria-label": base(path), onclick: () => lb.remove() },
      frame, el("div", { class: "lb-cap", text: `${path}  ·  click anywhere to close` }));
    document.body.append(lb);
  }

  function verdictCtl(sid, ref, options, small) {
    const it = itemOf(sid, ref) || blankItem();
    return el("div", { class: `seg-ctl${small ? " v-ctl" : ""}`, role: "group" },
      options.map(([v, label]) => el("button", {
        type: "button", "data-v": v, class: it.v === v ? "on" : "", "aria-pressed": String(it.v === v), text: label,
        onclick: () => {
          const r = itemOf(sid, ref, true);
          r.v = r.v === v ? "" : v;
          touch(sid);
          renderCase();
          if (["flag", "wrong", "partial", "incorrect", "cannot_verify"].includes(r.v)) {
            const key = refKey(ref);
            setTimeout(() => { const t = document.querySelector(`[data-note="${key}"]`); if (t) t.focus(); }, 0);
          }
        },
      })));
  }
  const refKey = (ref) => (typeof ref === "string" ? ref : ref.join("-"));
  function noteBox(sid, ref) {
    const it = itemOf(sid, ref) || blankItem();
    const needs = ["flag", "wrong", "partial", "incorrect", "cannot_verify"].includes(it.v);
    if (!needs && !it.note) return null;
    return el("textarea", {
      class: "comment note", rows: 1, "data-note": refKey(ref), value: it.note,
      placeholder: it.v === "flag" ? "What looks off? (optional)" : "What is wrong, and what should it say? (optional)",
      onchange: (e) => { itemOf(sid, ref, true).note = e.target.value; touch(sid); },
    });
  }
  function itemRow(sid, ref, label, body, extraClass) {
    const it = itemOf(sid, ref) || blankItem();
    return el("div", { class: `t-row ${extraClass || ""} v-${it.v || "none"}` },
      el("div", { class: "t-row-head" }, el("span", { class: "label", text: label }), verdictCtl(sid, ref, ITEM_VERDICTS, true)),
      el("div", { class: "t-body" }, body),
      noteBox(sid, ref));
  }

  function argsView(c, call) {
    const tool = (c.sample.tools || []).find((t) => t.name === call.name);
    const args = call.args && typeof call.args === "object" ? call.args : { arguments: call.args };
    const rows = [], media = [];
    for (const [k, v] of Object.entries(args)) {
      let val;
      if (k === "video") val = el("code", { text: base(v) });
      else if (k === "image") { val = el("code", { text: base(v) }); media.push(frameView(v, args.bbox)); }
      else if (/time$/.test(k) && !isNaN(Number(v))) val = el("button", { class: "t-link", type: "button", text: `${v} s`, onclick: () => seekTo(v) });
      else val = typeof v === "object" ? JSON.stringify(v) : String(v);
      rows.push(el("dt", { text: k }), el("dd", null, val));
    }
    const range = args.start_time !== undefined ? el("button", {
      class: "ev-chip", type: "button", title: "Play just this window",
      onclick: () => seekRange(args.start_time, args.end_time),
    }, el("span", { class: "t", text: `▶ ${args.start_time}–${args.end_time ?? "end"} s` }), "play window") : null;
    return el("div", { class: "call" },
      el("div", { class: "call-head" }, el("span", { class: "tool-name", title: tool ? tool.description : "", text: call.name }), range),
      el("dl", { class: "kv args" }, rows), media);
  }
  function outputView(o) {
    const ct = o.content;
    if (ct && typeof ct === "object" && ct.type === "image") return frameView(ct.content);
    const text = ct && typeof ct === "object" ? (typeof ct.content === "string" ? ct.content : JSON.stringify(ct.content, null, 2)) : String(ct ?? "");
    return el("div", { class: "out-text" }, withTimes(text));
  }

  function renderStep(c, s, i) {
    const sid = c.id, n = String(i + 1);
    return el("article", { class: "panel step" },
      el("div", { class: "step-head" }, el("span", { class: "step-n", text: n }),
        el("span", { class: "step-title", text: `Step ${n}` }), s.calls.map((cl) => el("span", { class: "pill accent", text: cl.name }))),
      itemRow(sid, ["step", n, "thought"], "Thought", el("div", { class: "thought" }, withTimes(s.thought || "(no thought)"))),
      itemRow(sid, ["step", n, "call"], "Tool call", s.calls.map((cl) => argsView(c, cl))),
      itemRow(sid, ["step", n, "output"], "Tool output",
        s.outputs.length ? s.outputs.map(outputView) : el("div", { class: "orig", text: "(no tool output)" }), "out"));
  }

  function renderFinal(c) {
    const sid = c.id, f = c.traj.final || { answer: "" };
    const it = itemOf(sid, "final") || blankItem();
    const gold = goldOf(c.sample.gt_answer);
    const concerns = (c.meta.concerns || []).filter(Boolean);
    return el("article", { class: `panel step final v-${it.v || "none"}` },
      el("div", { class: "step-head" }, el("span", { class: "step-n done", text: "✓" }), el("span", { class: "step-title", text: "Final answer" })),
      f.thought ? el("div", { class: "thought", style: { marginBottom: "8px" } }, withTimes(f.thought)) : null,
      el("div", { class: "answer" }, f.answer || "(no final answer)"),
      concerns.length ? el("div", { class: "concerns" }, el("b", { text: "GPT's own concerns (parts the frames may not support)" }),
        el("ul", null, concerns.map((x) => el("li", null, withTimes(x))))) : null,
      el("details", { class: "reason" },
        el("summary", { text: `Reference answers from the dataset (${gold.accepted.length}${gold.rejected.length ? ` + ${gold.rejected.length} rejected` : ""})` }),
        el("div", { class: "reason-body" },
          el("ul", { class: "gold" }, gold.accepted.map((g) => el("li", { text: g }))),
          gold.rejected.length ? [el("b", { text: "Rejected: " }), el("ul", { class: "gold rej" }, gold.rejected.map((g) => el("li", { text: g })))] : null)),
      el("div", { class: "q-review" },
        el("div", null, el("div", { class: "label", style: { marginTop: "0" }, text: "Is the final answer correct for this video?" }),
          verdictCtl(sid, "final", FINAL_VERDICTS))),
      noteBox(sid, "final"));
  }

  function renderCase() {
    const c = state.cases[state.current];
    const root = $("#case");
    if (!c) { root.replaceChildren(); return; }
    const a = annFor(c.id);
    const st = statusOf(c.id);
    const n = counts(c);
    const head = el("div", { class: "case-head" },
      el("div", null,
        el("h2", { text: `Trajectory #${c.id}` }),
        el("div", { class: "meta-chips" },
          c.meta.vqid ? el("span", { class: "pill", text: `VQID ${c.meta.vqid}` }) : null,
          el("span", { class: "pill", text: stem(c.videoPath) }),
          el("span", { class: "pill", text: `${c.traj.steps.length} tool step${c.traj.steps.length === 1 ? "" : "s"}` }),
          el("span", { class: "pill", text: `${state.current + 1} of ${state.cases.length}` }),
          el("span", { class: "pill time", id: "time-here", text: `⏱ ${fmtActive(state.activity.by_sample[c.id] || 0)} here` }),
          n.wrong ? el("span", { class: "pill bad", text: `✗ ${n.wrong} incorrect` }) : null,
          n.flag ? el("span", { class: "pill warn", text: `⚑ ${n.flag} flagged` }) : null,
          st === "done" ? el("span", { class: "pill good", text: "✓ Done" }) : st === "in_progress" ? el("span", { class: "pill star", text: "In progress" }) : null)),
      el("div", { class: "nav-btns" },
        el("button", { class: "btn ghost", type: "button", text: "← Previous", disabled: state.current === 0, onclick: () => openCase(state.current - 1) }),
        el("button", { class: "btn ghost", type: "button", text: "Next →", disabled: state.current >= state.cases.length - 1, onclick: () => openCase(state.current + 1) })));

    const qIt = itemOf(c.id, "question") || blankItem();
    const question = el("article", { class: `panel step question v-${qIt.v || "none"}` },
      el("div", { class: "step-head" }, el("span", { class: "step-n q", text: "Q" }), el("span", { class: "step-title", text: "Question" }),
        el("span", { class: "grow" }), verdictCtl(c.id, "question", ITEM_VERDICTS, true)),
      el("div", { class: "q-text" }, c.traj.question),
      noteBox(c.id, "question"));

    const hint = el("div", { class: "q-hint" },
      el("span", null, "Check each ", el("b", { text: "thought, tool call and tool output" }), " against the video, then the final answer."),
      el("span", { class: "pill star", text: `${n.rated} / ${n.total} rated` }));

    const overall = el("textarea", {
      class: "comment", rows: 1, placeholder: "Comment on this trajectory (optional): missing step, better tool, …",
      value: (a && a.comment) || "",
      onchange: (e) => { annFor(c.id, true).comment = e.target.value; touch(c.id); },
    });
    const left = n.total - n.rated;
    const doneBar = el("section", { class: "panel done-bar" }, overall,
      el("div", { style: { display: "grid", gap: "6px", justifyItems: "end" } },
        el("div", { class: "edit-row" },
          left ? el("button", { class: "btn ghost", type: "button", text: `Mark the other ${left} correct`,
            title: "Sets every unrated thought, call, output and the question to ✓ Correct (the final answer too)",
            onclick: () => markRestCorrect(c) }) : null,
          el("button", { class: "btn primary lg", type: "button", text: st === "done" ? "Saved ✓  Next →" : "Done & next →", onclick: () => markDone(c) })),
        el("div", { class: "msg", text: left ? `${left} item${left === 1 ? "" : "s"} not rated yet.` : "Everything rated. Progress saves automatically." })));

    for (const u of state.imageUrls) URL.revokeObjectURL(u);
    state.imageUrls = [];
    const existing = root.dataset.caseId === c.id ? root.querySelector(".video-panel") : null;
    const videoPanel = existing || renderVideoPanel(c);
    const scrollY = window.scrollY;
    root.replaceChildren(el("div", { class: "case-inner" }, head,
      el("div", { class: "grid traj-grid" }, videoPanel,
        el("div", { class: "q-col" }, hint, question, c.traj.steps.map((s, i) => renderStep(c, s, i)), renderFinal(c), doneBar))));
    if (existing) window.scrollTo(0, scrollY);
    root.dataset.caseId = c.id;
  }

  function markRestCorrect(c) {
    for (const r of allRefs(c)) {
      const it = itemOf(c.id, r, true);
      if (!it.v) it.v = r === "final" ? "correct" : "ok";
    }
    touch(c.id);
    renderCase();
  }
  function markDone(c) {
    if (!requireReviewer()) return;
    const n = counts(c);
    if (n.rated < n.total) {
      toast(`Rate every item first (${n.total - n.rated} left), or use “Mark the other ${n.total - n.rated} correct”.`, 3800);
      const first = document.querySelector(".t-row.v-none, .step.v-none");
      if (first) first.scrollIntoView({ block: "center", behavior: "smooth" });
      return;
    }
    const a = annFor(c.id, true);
    a.status = "done";
    a.updated_at = nowIso();
    persist();
    updateProgress();
    const next = state.cases.findIndex((x, i) => i > state.current && statusOf(x.id) !== "done");
    const any = next >= 0 ? next : state.cases.findIndex((x) => statusOf(x.id) !== "done");
    if (any >= 0) { openCase(any); toast("Saved. Next trajectory"); }
    else { renderCase(); renderList(); toast("All trajectories done. Use Export to download your review.", 4200); }
  }

  // ------------------------------------------------------------------ save / export / import
  function buildExport() {
    const samples = {};
    const byId = new Map(state.cases.map((c) => [c.id, c]));
    for (const [sid, a] of Object.entries(state.ann)) {
      const c = byId.get(sid);
      if (!c) { samples[sid] = a; continue; }
      const steps = c.traj.steps.map((s, i) => {
        const r = a.steps[String(i + 1)] || {};
        return { step: i + 1, tools: s.calls.map((x) => x.name), thought: normItem(r.thought), call: normItem(r.call), output: normItem(r.output) };
      });
      const n = counts(c);
      samples[sid] = {
        vqid: c.meta.vqid || "", video: c.videoPath, time_spent_s: Number(state.activity.by_sample[sid]) || 0,
        status: a.status, updated_at: a.updated_at, comment: a.comment || "",
        n_items: n.total, n_rated: n.rated, n_flagged: n.flag, n_incorrect: n.wrong,
        question: normItem(a.question), final: normItem(a.final), final_answer: (c.traj.final && c.traj.final.answer) || "",
        steps: Object.fromEntries(steps.map((s) => [String(s.step), s])),
      };
    }
    return {
      ui_version: UI_VERSION, tool: TOOL, annotator: state.reviewer, exported_at: nowIso(),
      n_loaded: state.cases.length, n_done: Object.values(samples).filter((s) => s.status === "done").length,
      activity: { total_active_s: totalActive(), by_sample: state.activity.by_sample },
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
      toast(`Progress saved in your folder (${progressFileName()}). It loads automatically next time.`, 4200);
      return;
    }
    download(progressFileName(), JSON.stringify(buildExport(), null, 2), "application/json");
    state.lastSaved = { where: "as file", at: new Date() };
    setSaveStatus();
    toast("Progress file downloaded. Put it in your trajectory folder: it loads automatically next time.", 5200);
  }
  function exportJson() {
    if (!requireReviewer()) return;
    const data = buildExport();
    if (!Object.keys(data.samples).length) { toast("Nothing reviewed yet"); return; }
    download(`catagent_traj_review_${safeName(state.reviewer)}_${stamp()}.json`, JSON.stringify(data, null, 2), "application/json");
    toast(`Exported ${Object.keys(data.samples).length} trajector${Object.keys(data.samples).length === 1 ? "y" : "ies"}`);
  }
  function csvCell(v) {
    const s = Array.isArray(v) ? v.join(" | ") : v === null || v === undefined ? "" : String(v);
    return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  }
  function exportCsv() {
    if (!requireReviewer()) return;
    const data = buildExport();
    const byId = new Map(state.cases.map((c) => [c.id, c]));
    const cols = ["annotator", "key", "vqid", "status", "item", "step", "tool", "verdict", "note", "text", "trajectory_comment", "time_spent_s", "updated_at"];
    const rows = [cols.join(",")];
    for (const [sid, s] of Object.entries(data.samples)) {
      const c = byId.get(sid);
      const row = (item, step, tool, it, text) => rows.push([data.annotator, sid, s.vqid, s.status, item, step, tool,
        (it && it.v) || "", (it && it.note) || "", text, s.comment, s.time_spent_s ?? "", s.updated_at].map(csvCell).join(","));
      row("question", "", "", s.question, c ? c.traj.question : "");
      for (const [k, st] of Object.entries(s.steps || {})) {
        const src = c && c.traj.steps[Number(k) - 1];
        row("thought", k, (st.tools || []).join(" | "), st.thought, src ? src.thought : "");
        row("call", k, (st.tools || []).join(" | "), st.call, src ? JSON.stringify(src.calls.map((x) => ({ [x.name]: x.args }))) : "");
        row("output", k, (st.tools || []).join(" | "), st.output,
          src ? src.outputs.map((o) => (o.content && typeof o.content === "object" ? o.content.content : o.content)).join(" | ") : "");
      }
      row("final", "", "", s.final, s.final_answer || "");
    }
    if (rows.length === 1) { toast("Nothing reviewed yet"); return; }
    download(`catagent_traj_review_${safeName(state.reviewer)}_${stamp()}.csv`, "﻿" + rows.join("\r\n"), "text/csv");
    toast(`Exported ${rows.length - 1} rows`);
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
      toast(`Loaded ${n} saved review(s)${state.cases.length ? "" : ". Now open your trajectory folder."}`, 3800);
    } catch (e) {
      toast("That file is not a CAT-Agent trajectory progress or export file");
    }
  }

  // ------------------------------------------------------------------ wiring
  function updateStartButtons() {
    const ok = !!state.reviewer;
    const b = $("#btn-folder");
    if (b) b.disabled = !ok;
    const hint = $("#start-hint");
    if (hint) hint.textContent = ok ? `Reviewing as ${state.reviewer}. Your progress is saved under this name.`
      : "Enter your name or ID to start. Nothing is saved without it.";
  }

  function init() {
    try { state.reviewer = (localStorage.getItem("catagent.reviewer") || "").trim(); } catch (e) { /* ignore */ }
    loadStore();
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
        if (input.id === "reviewer-start") $("#btn-folder").disabled = !input.value.trim();
      });
      input.addEventListener("change", () => {
        setReviewer(input.value);
        if (state.cases.length) { renderCase(); renderList(); updateProgress(); }
        if (state.reviewer) toast(`Saving progress as ${state.reviewer}`);
      });
      input.addEventListener("keydown", (e) => { if (e.key === "Enter") input.blur(); });
    }
    updateStartButtons();

    const inFolder = $("#input-folder"), inImport = $("#input-import");
    const commitStartName = () => { const v = $("#reviewer-start"); if (v && v.value.trim() && v.value.trim() !== state.reviewer) setReviewer(v.value); };
    $("#btn-folder").addEventListener("click", () => { commitStartName(); openFolder(); });
    $("#btn-open").addEventListener("click", () => openFolder());
    $("#btn-save").addEventListener("click", () => saveProgressNow());
    inFolder.addEventListener("change", () => { state.dirHandle = null; state.canWrite = false; loadEntries(fromInput(inFolder.files)); inFolder.value = ""; });
    inImport.addEventListener("change", () => { if (inImport.files[0]) importJson(inImport.files[0]); inImport.value = ""; });

    const dz = $("#dropzone");
    ["dragenter", "dragover"].forEach((ev) => document.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => document.addEventListener(ev, (e) => {
      if (ev === "dragleave" && e.relatedTarget) return;
      dz.classList.remove("over");
    }));
    document.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("over"); commitStartName(); onDrop(e.dataTransfer); });
    dz.addEventListener("keydown", (e) => { if ((e.key === "Enter" || e.key === " ") && e.target === dz) { e.preventDefault(); commitStartName(); openFolder(); } });

    const exBtn = $("#btn-export"), menu = $("#export-menu");
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
      if (e.key === "Escape") { const lb = $(".lightbox"); if (lb) lb.remove(); return; }
      if (["INPUT", "TEXTAREA", "SELECT"].includes(tag) || e.metaKey || e.ctrlKey || e.altKey) return;
      if (!state.cases.length) return;
      if (e.key === "ArrowRight" || e.key === "j") openCase(state.current + 1);
      else if (e.key === "ArrowLeft" || e.key === "k") openCase(state.current - 1);
    });

    window.addEventListener("beforeunload", (e) => {
      if (state.unsaved) { e.preventDefault(); e.returnValue = ""; }
      else if (state.dirHandle && state.canWrite && folderTimer) writeProgressToFolder();
    });
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { parseDialogs, goldOf, parseBbox, frameKey, fmtTime, csvCell };
  } else {
    document.addEventListener("DOMContentLoaded", init);
  }
})();
