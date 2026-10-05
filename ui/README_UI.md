# ophbench clinician review UI

A static, dependency-free web app (vanilla HTML/CSS/JS, no build step) in which clinicians review
the AI-generated questions for each sampled surgery video, judge and rate them, fix wording,
select the 1–2 questions that go into the benchmark and flag unusable samples. Everything is
saved in the browser (localStorage) and leaves it only when the annotator presses **Submit**
(POST to a configurable endpoint) or **Export JSON**.

```
ui/
  index.html  app.js  styles.css      the app (hash routing: #/ list, #/s/<sample_id> sample view)
  config.js                           deployment settings (MEDIA_BASE_URL, SUBMIT_URL, SELECT_MIN/MAX)
  data/   meta.json index.json assignments.json samples/<sample_id>.json   written by `./ophbench export-ui`
  media/  <sample_id>/contact_sheet.jpg preview.mp4                        written by `./ophbench export-ui`
  submit/Code.gs                      Google Apps Script endpoint for Submit (optional)
```

All fetches and media URLs are relative to `index.html`, so the folder works from any sub-path
(GitHub Pages `https://user.github.io/repo/`, an internal web server, or `scripts/serve_ui.py`).
It does **not** work from `file://` because browsers block `fetch()` there — always serve it.

---

## 1. Generate the data

```bash
cd /mnt/store/tashraf4/projects/ophbench
./ophbench export-ui                       # ui/data/*.json + ui/media/<id>/ (symlinks by default)
./ophbench export-ui --media-mode copy     # real files (needed before deploy_ui.sh commits media)
./ophbench export-ui --annotators alice,bob --overlap 0.1   # personal lists + 10 % shared samples
```

`export-ui` reads the validated QA sets (`data/qa/qa_validated.jsonl`), the prepared samples and
`config/pipeline.yaml` (`ui:` section — `questions_to_select_min/max`, `annotators`,
`overlap_fraction`, `media_base_url`, `submit_url`). `--include-invalid` also exports questions that
failed automatic validation (they are shown with a warning box).

### Offline batches (videos shared as folders)

Clinicians usually receive the videos offline (shared drive / USB) in batches of ~100 and the hosted UI only
carries the small contact sheets:

```bash
./ophbench export-ui --batch 1 --annotators alice,bob      # batch 1 of the benchmark order (rows 0-99 of the manifest)
./ophbench export-ui --batch 2 --annotators alice,bob      # added to the same ui/data (index/meta/assignments are cumulative)
```

Each `--batch K` run (default `--media-mode sheets`: contact sheets copied, no hosted previews) writes
`"batch": K` into every index entry / sample document, `video.local_file = "<sample_id>.mp4"`,
`meta.batches["K"]`, `meta.offline_media = true`, and the offline package `ui_packages/batch_00K/` with one
`<sample_id>.mp4` per sample (`--package-kind preview` = 360p previews, `source` = original videos),
`MANIFEST.csv` and a `README.txt` for the clinicians. Zip that folder and share it together with the UI link.
In the UI the clinician presses **Load videos folder** once and picks the unzipped folder: Chrome/Edge keep the
folder handle (File System Access API, re-confirmed with *Reconnect* after a restart), Firefox/Safari fall back to a
folder picker that must be re-selected per session. Files are read locally and never uploaded. The list page gets a
**Batch** filter and per-batch progress; annotation records carry `batch` per sample.

## 2. Serve locally / on the lab network (VPN)

```bash
python3 scripts/serve_ui.py                       # http://127.0.0.1:8765/
python3 scripts/serve_ui.py --bind 0.0.0.0 --port 8765   # reachable by colleagues on the VPN
python3 scripts/serve_ui.py --ui-dir ui --data-dir data --quiet
```

The server (stdlib only) adds correct MIME types, **HTTP Range** support (video seeking), CORS
headers and two API routes:

* `POST /api/annotations` — stores the submitted record as
  `<data-dir>/annotations/<annotator>_<YYYYmmdd_HHMMSS_mmm>.json` (adds `received_at`). Put
  `SUBMIT_URL: "api/annotations"` in `config.js` to use it from the same server.
* `GET /api/health`, `GET /api/annotations` (list stored files).

`--data-dir` defaults to `$OPHBENCH_DATA_DIR` or `../data` relative to the ui folder.

## 3. Deploy to GitHub Pages

This server has no GitHub credentials, so the script prepares a push-ready repository and prints
the commands; push from a machine that can reach GitHub (or pass `--push` there).

```bash
scripts/deploy_ui.sh user/repo                 # branch gh-pages, media included
scripts/deploy_ui.sh user/repo --no-media      # exclude ui/media (see §4), commit everything else
scripts/deploy_ui.sh user/repo --branch main --ssh --push
GITHUB_REPO=user/repo scripts/deploy_ui.sh     # env var form
```

It: checks the export, refuses symlinked media (git would commit links, not files — re-export with
`--media-mode copy` or use `--no-media`), warns when media exceeds GitHub limits (100 MB/file hard
limit, ~1 GB per Pages site), writes `.nojekyll` and `.gitignore`, `git init`s **inside `ui/`** on
the chosen branch, commits, sets the remote and prints:

1. `git -C ui push -u origin gh-pages`
2. GitHub → **Settings → Pages → Build and deployment → Source: Deploy from a branch → Branch:
   gh-pages / (root) → Save**. The site appears after 1–2 minutes at `https://user.github.io/repo/`.
3. Re-run the script after every new export and push again.

Pages on a **private** repository needs GitHub Pro/Team. If the repo must be public, review the
export first (`ui/data/samples/*.json` metadata and provenance, `ui/media/`) for anything that must
not be published, or host internally with `serve_ui.py` instead.

## 4. Hosting the media elsewhere (`MEDIA_BASE_URL`)

1,500 previews are typically several GB — too much for GitHub. Keep `ui/media/` on any HTTPS web
server (lab server, object storage, a VPN-only host) and set in `ui/config.js`:

```js
MEDIA_BASE_URL: "https://media.example.org/ophbench"   // => .../ophbench/media/<sample_id>/preview.mp4
```

Requirements for the media host:

* **HTTPS** when the UI is served over HTTPS (GitHub Pages) — browsers block mixed content.
* **HTTP Range requests** (`Accept-Ranges: bytes`, 206 responses), otherwise seeking is impossible
  and the timeline/evidence clicks will not work. `scripts/serve_ui.py` already does this; nginx,
  Apache, S3/GCS/Azure static hosting and Cloudflare R2 do as well.
* **CORS is not required** for plain `<video>`/`<img>` playback. Add
  `Access-Control-Allow-Origin: *` (serve_ui.py sends it) only if you later read frames into a
  canvas or fetch media with JavaScript.
* Keep the same layout: `<MEDIA_BASE_URL>/media/<sample_id>/preview.mp4` and `contact_sheet.jpg`
  (copy `ui/media/` as-is). When a video is missing the UI shows the contact sheet with a notice.

## 5. Submissions: Google Apps Script endpoint (`SUBMIT_URL`)

For a GitHub-Pages deployment there is no server to POST to; a free Apps Script web app writes
submissions into a Google Sheet.

1. Create a Google Sheet (e.g. "ophbench annotations"). **Extensions → Apps Script**.
2. Replace the content of `Code.gs` with `ui/submit/Code.gs`; save.
3. Optional: run `testDoPost` once from the editor (▶) to grant permissions and see test rows.
4. **Deploy → New deployment → Select type: Web app**. Description "ophbench", **Execute as: Me**,
   **Who has access: Anyone** (required — the browser posts anonymously). Deploy, authorise, copy
   the Web app URL (`https://script.google.com/macros/s/…/exec`).
5. In `ui/config.js` set `SUBMIT_URL: "https://script.google.com/macros/s/…/exec"`; re-deploy the
   UI (re-run `deploy_ui.sh`, push). A `GET` of the URL in a browser returns `{"ok":true,...}`.
6. After editing `Code.gs`, **Deploy → Manage deployments → ✎ → Version: New → Deploy**; the URL stays.

Sheets written: `questions` (annotator, sample_id, dataset, qid, selected, correctness, relevance,
difficulty, agentic, clarity, edited, question_edit, answer_edit, comment, updated_at, received_at),
`samples` (annotator, sample_id, status, flags, comment, updated_at, received_at) and `submissions`
(audit log). Headers are created automatically. Each Submit appends everything again — that is
intended; `collect_annotations.py` keeps the latest version per (annotator, sample, question).

The UI posts with `Content-Type: text/plain;charset=utf-8` and `redirect: "follow"`; this avoids a
CORS preflight and follows the Apps Script redirect, so any 2xx is treated as success and the
response JSON is shown to the annotator.

## 6. Collecting results

```bash
# JSON files (serve_ui.py submissions and/or files exported by annotators, copied into data/annotations/)
python3 scripts/collect_annotations.py
# plus Google Sheet exports (File > Download > CSV of the "questions" and "samples" sheets)
python3 scripts/collect_annotations.py --sheet-csv questions.csv --samples-csv samples.csv
python3 scripts/collect_annotations.py --only-done --data-dir /other/data --out-dir /tmp/out
```

Outputs in `data/annotations/`: `merged.jsonl` (one row per annotator × sample × question, latest
version only), `per_sample.csv` (selected-question counts, consensus, flags), `summary.json` and
`summary.md` (selection rates per dataset / procedure category / question category / annotator,
correctness distribution, mean ratings, inter-annotator percent agreement and Cohen's κ on
"selected" for samples seen by ≥ 2 annotators, selection disagreements, flagged samples).

---

## 7. Annotation guide for clinicians

**Goal.** Each video comes with three AI-written questions and gold answers. We want a benchmark of
*tough, agentic* questions — ones that require several reasoning steps across the video (seeking,
counting, measuring, ordering phases, weighing a complication) and whose answers a clinician can
verify from the video and its labels. You tell us which questions are correct, how good they are,
fix their wording, and pick the best 1–2 per video.

**Getting started.** Open the link, enter your **annotator ID** (the one we gave you; letters,
digits, `.`, `_`, `-`) and press *Continue*. The list shows your assigned videos with a status chip
(To do / In progress / Done), filters and a progress bar. If you received the videos as a folder, press
**Load videos folder** and pick that folder once (the files stay on your computer; nothing is uploaded) —
then the videos play inside the page. Click a row to open it.

**Sample view.**

* Left: the preview video (or the contact sheet of timestamped frames when the video is missing),
  the **phase timeline** — hover for the name and times, click to jump — small ticks for the frames
  the model saw, the metadata panel and dataset information.
* Right: the three **question cards** and the **sample review** panel.

**For every question:**

| Control | What to do |
|---|---|
| Select for benchmark | Tick for the best 1–2 questions of this video (the exact rule is shown in the help panel). Prefer verifiable, multi-step, clinically meaningful questions. |
| Is the gold answer correct? | *Correct* / *Partially correct* (right in substance, imprecise or incomplete) / *Incorrect* / *Cannot verify* (video or labels don't allow a decision). |
| Clinical relevance 1–5 | 1 trivia … 5 a question a surgical trainer would ask. |
| Difficulty 1–5 | 1 one glance or metadata alone … 5 careful review of several parts of the video. |
| Agentic / multi-step 1–5 | 1 single observation … 5 needs planning several actions (seek, count, measure, compare segments, look things up) and verification. |
| Clarity 1–5 | 1 ambiguous … 5 unambiguous with exactly one defensible answer. |
| Question / Gold answer | Edit freely; your text is what enters the benchmark. An *edited* marker appears; *Reset to original* undoes. |
| Comment | Anything we should know about this question. |

**Sample level:** flags — *Video quality issue*, *Metadata wrong*, *Not suitable* (should not be in
the benchmark at all), *Duplicate* (same video seen under another id) — a comment, and **Mark
complete** (requires the selection rule to be satisfied). After completing, the UI jumps to the
next unfinished sample (toggle *auto-advance*).

**Saving.** Everything is saved instantly in this browser under your ID. Clearing site data or
switching browsers loses unsaved work, so **Submit (or Export JSON) at the end of every session**.
*Import JSON* merges a file exported earlier (newer entries win) — use it to continue on another
computer. Submitting several times is fine; the latest version of each sample is kept.

**Keyboard.** `j` / `k` next / previous sample, `n` next unfinished, `1` `2` `3` toggle selection of
question 1–3, `?` help, `Esc` close. Shortcuts are ignored while typing.

**Troubleshooting.** *Video does not play* → the contact sheet is shown automatically; the
timestamps on the frames still let you verify. *"Submit is not configured"* → use Export JSON and
send us the file. *Nothing loads* → the page must be opened over http(s), not as a local file.

---

## 8. Data shapes (for developers)

* `data/meta.json` — `{generated_at, ui_version: 1, n_samples, datasets: {dataset: count},
  select_min, select_max, annotators: [..], dataset_meta: {dataset: {title, blurb, license, citation}}}`
  plus, for offline batches, `batches: {"K": {n, exported_at, package, package_kind}}`, `batch_size`,
  `n_batches_total`, `offline_media`. Index entries and sample documents then carry `batch`, and
  `video.local_file` (`<sample_id>.mp4`) names the file the UI looks for in the loaded folder; annotation
  records store `batch` per sample.
  `select_min/select_max` override `SELECT_MIN/SELECT_MAX` from `config.js`.
* `data/index.json` — list of `{sample_id, dataset, dataset_title, procedure, procedure_category,
  duration_s, n_questions, thumb, preview, assigned_to, has_issues}`.
* `data/assignments.json` — `{annotators, by_annotator: {annotator: [sample_id]}, overlap}`; empty
  `annotators` means everyone sees everything (an unknown ID also sees everything, with a notice).
* `data/samples/<sample_id>.json` — the sample view payload (video, metadata, segments, frames,
  questions with validation_issues, provenance).
* Annotation record (localStorage key `ophbench.ann.<annotator>`, Export, Submit body):

```json
{"ui_version": 1, "annotator": "alice", "exported_at": "2026-10-05T12:00:00.000Z",
 "samples": {"cataract101__case_269": {"status": "done", "updated_at": "…", "flags": ["video_quality"], "comment": "",
   "questions": {"q1": {"selected": true, "correctness": "correct", "relevance": 5, "difficulty": 4, "agentic": 4, "clarity": 5,
                        "question_edit": "", "answer_edit": "", "edited": false, "comment": ""}}}}}
```

`question_edit` / `answer_edit` are empty strings while the text equals the original; `edited` is
true as soon as one of them differs. A sample absent from `samples` is "To do"; present with
`status: "in_progress"` or `"done"`. Other localStorage keys: `ophbench.annotator` (current ID),
`ophbench.theme`, `ophbench.prefs` (thumbnails, auto-advance, last export/submit time).

Pure helper functions of `app.js` are exported for node (`require("ui/app.js")`) and can be unit
tested without a browser; `node --check ui/app.js` validates the syntax.
