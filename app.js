/* CAT-Agent clinician review app (vanilla JS, no build step, no network calls).
 *
 * The clinician opens the batch folder they received (videos + <sample_id>.qa.json files with the
 * GPT-generated questions). Everything is read locally in the browser; progress is autosaved in
 * localStorage and exported as JSON (full record, original + edited) or CSV.
 */
(function () {
  "use strict";

  const CFG = Object.assign({ SELECT_MIN: 1, SELECT_MAX: 2 }, (typeof window !== "undefined" && window.OPHBENCH_CONFIG) || {});
  const UI_VERSION = 2;
  const VIDEO_EXT = ["mp4", "m4v", "mov", "webm", "mkv", "ogv"];
  const CORRECTNESS = [
    ["correct", "Correct"], ["partial", "Partly"], ["incorrect", "Incorrect"], ["cannot_verify", "Can't tell"],
  ];
  const PALETTE = ["#7cc4b8", "#9bb7e3", "#e7b07a", "#c3a6dd", "#e59aa8", "#a8cf8e", "#e6cf72", "#8fc9e0",
    "#d9a3c8", "#b9c08a", "#f0a989", "#a3b0c2"];

  // ------------------------------------------------------------------ state
  const state = {
    reviewer: "",
    cases: [],          // [{id, doc, video: File|null}]
    current: -1,
    filter: "all",
    search: "",
    ann: {},            // sample_id -> annotation record
    editing: {},        // qid -> true while the edit form is open (current case only)
    videoUrl: null,
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
  const humanize = (s) => String(s || "").replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
  const nowIso = () => new Date().toISOString();
  const stem = (name) => String(name || "").split("/").pop().replace(/\.qa\.json$/i, "").replace(/\.[^.]+$/, "");
  const ext = (name) => (String(name).split(".").pop() || "").toLowerCase();
  function colorFor(label) {
    let h = 0;
    for (const ch of String(label)) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
    return PALETTE[h % PALETTE.length];
  }

  let toastTimer = null;
  function toast(msg) {
    const t = $("#toast");
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove("show"), 2200);
  }

  // ------------------------------------------------------------------ storage
  const storeKey = () => `catagent.review.v2.${(state.reviewer || "_").toLowerCase()}`;
  function loadStore() {
    try {
      const raw = localStorage.getItem(storeKey());
      state.ann = raw ? JSON.parse(raw) : {};
    } catch (e) { state.ann = {}; }
  }
  function saveStore() {
    try { localStorage.setItem(storeKey(), JSON.stringify(state.ann)); } catch (e) { toast("Could not save in this browser — export regularly."); }
  }
  function setReviewer(name) {
    const prev = state.reviewer;
    const prevAnn = state.ann;
    state.reviewer = name.trim();
    try { localStorage.setItem("catagent.reviewer", state.reviewer); } catch (e) { /* ignore */ }
    loadStore();
    // first time naming yourself: keep the anonymous work
    if (!prev && state.reviewer && !Object.keys(state.ann).length && Object.keys(prevAnn).length) {
      state.ann = prevAnn;
      saveStore();
    }
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
    saveStore();
    renderList();
    updateProgress();
  }
  function finalOf(q, qa) {
    const edited = qa && qa.edited;
    return {
      question: edited && qa.question_edit ? qa.question_edit : q.question || "",
      answer: edited && qa.answer_edit ? qa.answer_edit : q.answer || "",
      options: edited && Array.isArray(qa.options_edit) ? qa.options_edit : (q.options || []),
    };
  }
  const statusOf = (sid) => (state.ann[sid] && state.ann[sid].status) || "todo";
  const selectedCount = (sid) => {
    const a = state.ann[sid];
    return a ? Object.values(a.questions).filter((q) => q.selected).length : 0;
  };

  // ------------------------------------------------------------------ loading files
  async function filesFromDrop(dt) {
    const out = [];
    const items = dt.items ? Array.from(dt.items) : [];
    const entries = items.map((it) => (it.webkitGetAsEntry ? it.webkitGetAsEntry() : null)).filter(Boolean);
    if (!entries.length) return Array.from(dt.files || []);
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
    return out;
  }

  async function loadFiles(files) {
    files = Array.from(files || []);
    if (!files.length) return;
    const videos = new Map();   // stem -> File
    const docs = new Map();     // sample_id -> doc
    let badJson = 0;
    for (const f of files) {
      const name = f.name || "";
      if (name.startsWith(".")) continue;
      const e = ext(name);
      if (VIDEO_EXT.includes(e)) { videos.set(stem(name).toLowerCase(), f); continue; }
      if (e !== "json") continue;
      try {
        const data = JSON.parse(await f.text());
        const list = Array.isArray(data) ? data : Array.isArray(data.samples) ? data.samples : [data];
        for (const d of list) {
          if (d && Array.isArray(d.questions) && d.questions.length) {
            const sid = String(d.sample_id || stem(name));
            docs.set(sid, Object.assign({ sample_id: sid }, d));
          }
        }
      } catch (err) { badJson += 1; }
    }
    if (!docs.size) {
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
    // videos added on their own (e.g. user picked videos after the questions)
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
    showWorkspace();
    const firstTodo = state.cases.findIndex((c) => statusOf(c.id) !== "done");
    openCase(firstTodo >= 0 ? firstTodo : 0);
    toast(`Loaded ${state.cases.length} video${state.cases.length === 1 ? "" : "s"} with questions` +
      (withVideo < state.cases.length ? ` · ${state.cases.length - withVideo} without a video file` : "") +
      (badJson ? ` · ${badJson} unreadable file(s) skipped` : ""));
  }

  function showWorkspace() {
    $("#start").classList.add("hidden");
    $("#workspace").classList.remove("hidden");
    $("#btn-open").classList.remove("hidden");
    $("#btn-export").disabled = false;
    renderList();
    updateProgress();
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
        class: `case-item${i === state.current ? " active" : ""}`, tabindex: 0,
        title: c.id,
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
    $("#case").scrollTop = 0;
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
    const duration = Number(doc.video && doc.video.duration_s) || Math.max(0, ...phaseSegments(doc).map((s) => s.end_s || 0));
    const segs = phaseSegments(doc);
    let timeline = null, phaseNow = null;
    if (segs.length && duration > 0) {
      const head = el("div", { class: "playhead" });
      const bar = el("div", {
        class: "timeline", title: "Click to jump",
        onclick: (e) => { const r = bar.getBoundingClientRect(); seekTo(((e.clientX - r.left) / r.width) * duration); },
      }, segs.map((s) => el("div", {
        class: "seg", title: `${s.label}  ${fmtTime(s.start_s)}–${fmtTime(s.end_s)}`,
        style: { width: `${(100 * Math.max(0, (s.end_s || 0) - (s.start_s || 0))) / duration}%`, background: colorFor(s.label), marginLeft: "0" },
      })), head);
      // place segments by absolute start (handles gaps/overlaps): use absolute positioning when overlaps exist
      {
        bar.style.display = "block";
        Array.from(bar.querySelectorAll(".seg")).forEach((node, k) => {
          const s = segs[k];
          Object.assign(node.style, { position: "absolute", left: `${(100 * s.start_s) / duration}%`, top: "0", bottom: "0" });
        });
      }
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
      const k = options.findIndex((o) => String(o).trim().toUpperCase().startsWith(letter));
      if (k >= 0) return k;
    }
    const a = String(answer).trim().toLowerCase();
    return options.findIndex((o) => {
      const body = String(o).replace(/^[A-Ha-h][\).:]\s*/, "").trim().toLowerCase();
      return body && a.includes(body.slice(0, 60));
    });
  }

  function renderQuestion(c, q, k) {
    const sid = c.id, qid = q.qid || `q${k + 1}`;
    const qa = qAnn(sid, qid);
    const fin = finalOf(q, qa);
    const selected = !!(qa && qa.selected);
    const editing = !!state.editing[qid];

    const star = el("button", {
      class: `star-btn${selected ? " on" : ""}`, type: "button", "aria-pressed": String(selected),
      title: `Mark as one of the best questions (key ${k + 1})`,
      onclick: () => toggleSelect(sid, qid),
    }, el("span", { class: "s", text: selected ? "★" : "☆" }), selected ? "Best" : "Mark best");

    const tags = el("div", { class: "q-tags" },
      el("span", { class: "q-index", text: `Q${k + 1}` }),
      q.category ? el("span", { class: "pill accent", text: humanize(q.category) }) : null,
      q.difficulty ? el("span", { class: "pill", text: q.difficulty === "very_hard" ? "Very hard" : humanize(q.difficulty) }) : null,
      q.answer_type ? el("span", { class: "pill", text: humanize(q.answer_type) }) : null,
      qa && qa.edited ? el("span", { class: "pill warn", text: "Edited" }) : null);

    let body;
    if (editing) {
      const tq = el("textarea", { rows: 4, value: fin.question });
      const ta = el("textarea", { rows: 3, value: fin.answer });
      const hasOptions = (fin.options && fin.options.length) || q.answer_type === "multiple_choice";
      const to = hasOptions ? el("textarea", { rows: Math.max(4, (fin.options || []).length + 1), value: (fin.options || []).join("\n") }) : null;
      body = el("div", { class: "edit-box" },
        el("label", null, "Question", tq),
        to ? el("label", null, "Options (one per line, e.g. “A. …”)", to) : null,
        el("label", null, "Answer", ta),
        el("div", { class: "edit-row" },
          el("button", { class: "btn subtle small", type: "button", text: "Reset to GPT original",
            onclick: () => { const r = qAnn(sid, qid, true); Object.assign(r, { edited: false, question_edit: "", answer_edit: "", options_edit: null }); delete state.editing[qid]; touch(sid); renderCase(); toast("Restored the original question"); } }),
          el("button", { class: "btn ghost small", type: "button", text: "Cancel", onclick: () => { delete state.editing[qid]; renderCase(); } }),
          el("button", { class: "btn primary small", type: "button", text: "Save changes",
            onclick: () => {
              const r = qAnn(sid, qid, true);
              const nq = tq.value.trim(), na = ta.value.trim();
              const no = to ? to.value.split("\n").map((s) => s.trim()).filter(Boolean) : null;
              const changed = nq !== (q.question || "").trim() || na !== (q.answer || "").trim() ||
                (no !== null && JSON.stringify(no) !== JSON.stringify((q.options || []).map((s) => String(s).trim())));
              Object.assign(r, changed
                ? { edited: true, question_edit: nq !== (q.question || "").trim() ? nq : "", answer_edit: na !== (q.answer || "").trim() ? na : "",
                    options_edit: no !== null && JSON.stringify(no) !== JSON.stringify((q.options || []).map((s) => String(s).trim())) ? no : null }
                : { edited: false, question_edit: "", answer_edit: "", options_edit: null });
              delete state.editing[qid];
              touch(sid);
              renderCase();
              toast(changed ? "Edit saved — the original is kept too" : "No changes");
            } })));
    } else {
      const ci = correctOptionIndex(fin.options, fin.answer);
      const ev = (q.evidence_timestamps || []).filter((e) => e && e.start_s !== undefined);
      body = el("div", null,
        el("div", { class: "q-text", text: fin.question }),
        fin.options && fin.options.length ? [el("div", { class: "label", text: "Options" }),
          el("ul", { class: "options" }, fin.options.map((o, j) => el("li", { class: j === ci ? "correct" : "", text: o })))] : null,
        el("div", { class: "label", text: "GPT answer" + (qa && qa.edited && qa.answer_edit ? " (edited)" : "") }),
        el("div", { class: "answer", text: fin.answer }),
        ev.length ? [el("div", { class: "label", text: "Evidence in the video" }),
          el("div", { class: "evidence" }, ev.map((e) => el("button", {
            class: "ev-chip", type: "button", title: "Jump to this moment", onclick: () => seekTo(e.start_s),
          }, el("span", { class: "t", text: `▶ ${fmtTime(e.start_s)}–${fmtTime(e.end_s)}` }), e.observation || "")))] : null,
        el("details", { class: "reason" },
          el("summary", { text: "Reasoning and why it's hard" }),
          el("div", { class: "reason-body" },
            q.answer_rationale ? [el("b", { text: "Reasoning: " }), q.answer_rationale, "\n\n"] : null,
            q.why_hard ? [el("b", { text: "Why it's hard: " }), q.why_hard, "\n\n"] : null,
            (q.agentic_skills || []).length ? [el("b", { text: "Skills needed: " }), q.agentic_skills.map(humanize).join(", "), "\n"] : null,
            (q.tool_plan || []).length ? [el("b", { text: "Agent steps: " }), q.tool_plan.map((s, j) => `${j + 1}. ${s}`).join("  "), "\n"] : null,
            q.confidence !== undefined ? [el("b", { text: "GPT confidence: " }), `${Math.round(100 * Number(q.confidence))}%`] : null)),
        qa && qa.edited ? el("details", { class: "reason" },
          el("summary", { text: "Show the original GPT version" }),
          el("div", { class: "orig" },
            el("b", { text: "Question: " }), q.question || "", "\n",
            (q.options || []).length ? [el("b", { text: "Options: " }), q.options.join("  |  "), "\n"] : null,
            el("b", { text: "Answer: " }), q.answer || "")) : null);
    }

    const corr = qa ? qa.correctness : "";
    const seg = el("div", { class: "seg-ctl", role: "group", "aria-label": "Is the answer correct?" },
      CORRECTNESS.map(([v, label]) => el("button", {
        type: "button", "data-v": v, class: corr === v ? "on" : "", "aria-pressed": String(corr === v),
        text: label,
        onclick: () => { const r = qAnn(sid, qid, true); r.correctness = r.correctness === v ? "" : v; touch(sid); renderCase(); },
      })));
    const review = el("div", { class: "q-review" },
      el("div", null, el("div", { class: "label", style: { marginTop: "0" }, text: "Is the answer correct?" }), seg),
      editing ? null : el("div", { class: "q-actions" },
        el("button", { class: "btn ghost small", type: "button", text: "✎ Edit", onclick: () => { state.editing[qid] = true; renderCase(); } })));
    const comment = el("textarea", {
      class: "comment", rows: 1, placeholder: "Note on this question (optional)", value: (qa && qa.comment) || "",
      onchange: (e) => { const r = qAnn(sid, qid, true); r.comment = e.target.value; touch(sid); },
    });

    return el("article", { class: `panel qcard${selected ? " selected" : ""}`, "data-qid": qid },
      el("div", { class: "q-top" }, tags, star), body, review, comment);
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
    const msg = el("div", { class: "msg", text: nSel >= CFG.SELECT_MIN ? `${nSel} best question${nSel === 1 ? "" : "s"} marked. Progress is saved automatically.`
      : `Mark at least ${CFG.SELECT_MIN} best question${CFG.SELECT_MIN === 1 ? "" : "s"} with ★ to finish this video.` });
    const doneBar = el("section", { class: "panel done-bar" }, overall, el("div", { style: { display: "grid", gap: "6px", justifyItems: "end" } }, doneBtn, msg));

    const hint = el("div", { class: "q-hint" },
      el("span", null, "Pick the ", el("b", { text: `best ${CFG.SELECT_MAX > 1 ? `1–${CFG.SELECT_MAX}` : "1"}` }), " question(s) with ★, check the answer, edit if needed."),
      el("span", { class: "pill star", text: `★ ${nSel} / ${CFG.SELECT_MAX}` }));

    root.replaceChildren(el("div", { class: "case-inner" }, head,
      el("div", { class: "grid" }, renderVideoPanel(c), el("div", { class: "q-col" }, hint, questions, doneBar))));
  }

  function markDone(c) {
    if (selectedCount(c.id) < CFG.SELECT_MIN) {
      toast(`Mark at least ${CFG.SELECT_MIN} best question with ★ first`);
      const first = $(".star-btn");
      if (first) first.focus();
      return;
    }
    const a = annFor(c.id, true);
    a.status = "done";
    a.updated_at = nowIso();
    saveStore();
    updateProgress();
    const next = state.cases.findIndex((x, i) => i > state.current && statusOf(x.id) !== "done");
    const any = next >= 0 ? next : state.cases.findIndex((x) => statusOf(x.id) !== "done");
    if (any >= 0) { openCase(any); toast("Saved. Next video"); }
    else { renderCase(); renderList(); toast("All videos done — use Export to download your review"); }
  }

  // ------------------------------------------------------------------ export / import
  function requireReviewer() {
    const input = $("#reviewer");
    if (state.reviewer) return true;
    input.classList.add("needs");
    input.focus();
    toast("Please enter your name or ID first (top right)");
    return false;
  }

  function buildExport() {
    const samples = {};
    for (const c of state.cases) {
      const a = state.ann[c.id];
      if (!a) continue;
      const qs = {};
      (c.doc.questions || []).forEach((q, k) => {
        const qid = q.qid || `q${k + 1}`;
        const r = a.questions[qid] || {};
        const fin = finalOf(q, r);
        qs[qid] = {
          selected: !!r.selected, correctness: r.correctness || "", edited: !!r.edited, comment: r.comment || "",
          question_edit: r.question_edit || "", answer_edit: r.answer_edit || "", options_edit: r.options_edit || null,
          relevance: null, difficulty: null, agentic: null, clarity: null,
          original: {
            question: q.question || "", answer: q.answer || "", options: q.options || [], category: q.category || "",
            answer_type: q.answer_type || "", difficulty: q.difficulty || "", answer_rationale: q.answer_rationale || "",
            evidence_timestamps: q.evidence_timestamps || [], confidence: q.confidence ?? null,
          },
          final: fin,
        };
      });
      samples[c.id] = {
        status: a.status, updated_at: a.updated_at, batch: c.doc.batch ?? null, dataset: c.doc.dataset || "",
        procedure: c.doc.procedure || "", comment: a.comment || "", flags: a.flags || [],
        selected_qids: Object.keys(qs).filter((k) => qs[k].selected), questions: qs,
      };
    }
    return {
      ui_version: UI_VERSION, tool: "CAT-Agent clinician review", annotator: state.reviewer, exported_at: nowIso(),
      n_loaded: state.cases.length, n_done: Object.values(samples).filter((s) => s.status === "done").length, samples,
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
  const safeName = (s) => String(s || "reviewer").replace(/[^A-Za-z0-9_.-]+/g, "_");
  const stamp = () => new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-");

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
    const cols = ["annotator", "sample_id", "dataset", "procedure", "status", "qid", "category", "selected", "correctness",
      "edited", "original_question", "final_question", "original_answer", "final_answer", "original_options",
      "final_options", "question_comment", "video_comment", "updated_at"];
    const rows = [cols.join(",")];
    for (const [sid, s] of Object.entries(data.samples)) {
      for (const [qid, q] of Object.entries(s.questions)) {
        rows.push([data.annotator, sid, s.dataset, s.procedure, s.status, qid, q.original.category, q.selected ? "yes" : "no",
          q.correctness, q.edited ? "yes" : "no", q.original.question, q.final.question, q.original.answer, q.final.answer,
          q.original.options, q.final.options, q.comment, s.comment, s.updated_at].map(csvCell).join(","));
      }
    }
    if (rows.length === 1) { toast("Nothing reviewed yet"); return; }
    download(`catagent_review_${safeName(state.reviewer)}_${stamp()}.csv`, "﻿" + rows.join("\r\n"), "text/csv");
    toast(`Exported ${rows.length - 1} question rows`);
  }

  async function importJson(file) {
    try {
      const data = JSON.parse(await file.text());
      if (!data || typeof data.samples !== "object" || Array.isArray(data.samples)) throw new Error("not an export file");
      if (!state.reviewer && data.annotator) { $("#reviewer").value = data.annotator; setReviewer(data.annotator); }
      let n = 0;
      for (const [sid, s] of Object.entries(data.samples)) {
        const cur = state.ann[sid];
        if (cur && cur.updated_at && s.updated_at && cur.updated_at > s.updated_at) continue;
        const qs = {};
        for (const [qid, q] of Object.entries(s.questions || {})) {
          qs[qid] = { selected: !!q.selected, correctness: q.correctness || "", edited: !!q.edited, comment: q.comment || "",
            question_edit: q.question_edit || "", answer_edit: q.answer_edit || "", options_edit: q.options_edit || null };
        }
        state.ann[sid] = { status: s.status || "in_progress", updated_at: s.updated_at || nowIso(), comment: s.comment || "",
          flags: s.flags || [], questions: qs };
        n += 1;
      }
      saveStore();
      if (state.cases.length) { renderCase(); renderList(); updateProgress(); }
      toast(`Imported ${n} video review(s)${state.cases.length ? "" : " — now open the batch folder"}`);
    } catch (e) {
      toast("That file is not a CAT-Agent review export");
    }
  }

  // ------------------------------------------------------------------ wiring
  function init() {
    try { state.reviewer = localStorage.getItem("catagent.reviewer") || ""; } catch (e) { /* ignore */ }
    loadStore();
    const rev = $("#reviewer");
    rev.value = state.reviewer;
    rev.addEventListener("change", () => {
      setReviewer(rev.value);
      rev.classList.toggle("needs", !state.reviewer);
      if (state.cases.length) { renderCase(); renderList(); updateProgress(); }
      if (state.reviewer) toast(`Saving progress as ${state.reviewer}`);
    });

    const inFolder = $("#input-folder"), inFiles = $("#input-files"), inImport = $("#input-import");
    $("#btn-folder").addEventListener("click", () => inFolder.click());
    $("#btn-files").addEventListener("click", () => inFiles.click());
    $("#btn-open").addEventListener("click", () => inFolder.click());
    inFolder.addEventListener("change", () => { loadFiles(inFolder.files); inFolder.value = ""; });
    inFiles.addEventListener("change", () => { loadFiles(inFiles.files); inFiles.value = ""; });
    inImport.addEventListener("change", () => { if (inImport.files[0]) importJson(inImport.files[0]); inImport.value = ""; });

    const dz = $("#dropzone");
    ["dragenter", "dragover"].forEach((ev) => document.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => document.addEventListener(ev, (e) => {
      if (ev === "dragleave" && e.relatedTarget) return;
      dz.classList.remove("over");
    }));
    document.addEventListener("drop", async (e) => {
      e.preventDefault();
      dz.classList.remove("over");
      loadFiles(await filesFromDrop(e.dataTransfer));
    });
    dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); inFolder.click(); } });

    const exBtn = $("#btn-export"), menu = $("#export-menu");
    // Import works before a folder is opened too
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
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { fmtTime, fmtDuration, stem, correctOptionIndex, csvCell, humanize };
  } else {
    document.addEventListener("DOMContentLoaded", init);
  }
})();
