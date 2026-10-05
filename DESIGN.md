# ophbench — Stage‑1 curation pipeline for an agentic ophthalmic‑surgery video benchmark

This document is the build contract. Every module must follow the paths, schemas and CLI
contracts below exactly, so independently written modules fit together.

## 0. Environment facts (verified 2026‑10‑05)

* Project root: `/mnt/store/tashraf4/projects/ophbench` (this folder). Python: the base
  miniconda interpreter `/mnt/store/tashraf4/miniconda3/bin/python3` (Python 3.14). Installed:
  httpx, pyyaml, pandas, numpy, pillow, openpyxl, imageio-ffmpeg, tenacity, tqdm, rich.
  Do **not** use opencv/av/decord; use the ffmpeg CLI only.
* ffmpeg: `imageio_ffmpeg.get_ffmpeg_exe()` → static ffmpeg 7.0.2 (has libx264). There is no
  matching ffprobe; a working ffprobe 4.2.2 exists at
  `/mnt/store/tashraf4/miniconda3/envs/opencompass/bin/ffprobe`. `bench/media.py` must support
  `ffprobe = auto` (use that path if it exists, else fall back to parsing `ffmpeg -i` stderr).
* Datasets root: `/mnt/store/tashraf4/datasets` (NFS, 229 TB pool, 22 TB free). Exact layouts in §3.
* LLM: JHU **WSE AI Gateway**. Key in `/mnt/store/tashraf4/.api_keys.env` as
  `export GPT_ASTRA_KEY=jhu_live_sk_…` (never print or log it). The key is scoped to one model:
  `openai/gpt-6-astra`. Verified routes (base `https://gateway.engineering.jhu.edu/gateway`):
  * `POST {base}/rest/v1/chat/completions` — OpenAI chat‑completions shape, model
    **`openai/gpt-6-astra`** (provider‑prefixed). Works. Supports
    `response_format: {type: json_schema, json_schema: {name, strict, schema}}`,
    `max_completion_tokens`, `reasoning_effort` ∈ {low, medium, high, **xhigh**} — the value
    `max` is rejected with HTTP 400 ("Supported values are: low, medium, high, xhigh"). So
    **"max thinking" == `xhigh`** on this gateway. Usage block includes
    `completion_tokens_details.reasoning_tokens`. Headers carry `cf-aig-request-id`.
  * `POST {base}/codex/openai/v1/responses` — OpenAI Responses shape, model **`gpt-6-astra`**
    (unprefixed; prefixed also accepted). Streams SSE by default; `"stream": false` returns JSON
    (the broker synthesises the object; treat `status` as unreliable and rely on `output` +
    `usage`). Supports `reasoning: {effort}` and `text.format` json_schema. Image input is
    `{"type":"input_image","image_url":"data:image/jpeg;base64,...","detail":"high"}`.
  * `/compat/chat/completions` and `/openai/chat/completions` return an upstream 401 for this key
    (broker does not attach the upstream credential) — do not use.
  * **No video input**: `input_file` with `video/mp4` and `video_url` parts are rejected. The
    pipeline therefore sends **sampled, timestamp‑stamped frames** (JPEG) plus metadata.
  * **Images work only on the Responses route.** On the REST route every image part form
    (`image_url` object, `image_url` string, `file`) is rejected ("Unknown parameter: 'image'" /
    Zod union errors), so the REST route is text‑only. Verified on the Responses route: one
    JPEG frame + `reasoning.effort = xhigh` → correct description, ~3k reasoning tokens, 92 s.
    The default route is therefore **`responses` with `stream: true`** (parse SSE, take the
    `response.completed` event, which also echoes the applied effort); `rest_chat` is kept for
    text‑only utility calls (e.g. JSON repair, probes).
* Network: pypi, github, huggingface reachable. `gh` CLI is **not** installed and the server's
  SSH key is **not** registered with GitHub (`ssh -T git@github.com` → permission denied). The
  UI is prepared as a push‑ready folder; the user pushes from a machine with GitHub access.
* Compute: 104 CPU, 503 GB RAM. Use thread pools for ffmpeg work (default 12 workers).

## 1. Repository layout

```
ophbench/
  DESIGN.md                  this file
  README.md                  how to run (written last; keep in sync with CLI)
  ophbench                   bash wrapper: exec python3 -m bench.cli "$@" from repo root
  requirements.txt  env.example  .gitignore
  config/pipeline.yaml       all tunables (quotas, sampling, media, llm, ui)
  config/prompts/system.md   system prompt
  config/prompts/user_template.md  user prompt template (Python str.format fields)
  resources/ophnet_labels.json      id→name maps (surgery, phase, operation + descriptions)
  resources/procedure_taxonomy.yaml keyword rules for Ophora + OphNet surgery→category map
  bench/
    __init__.py  cli.py  config.py  log.py  schema.py  util.py
    datasets/__init__.py  base.py  cataract101.py  cataract1k.py  cataract_lmm.py
             migs.py  ophnet.py  ophora.py        (lmm_phase/lmm_skill/lmm_raw live in cataract_lmm.py)
    inventory.py  sampler.py  media.py  prepare.py
    llm_client.py  prompts.py  generate.py  validate.py  export_ui.py  status.py
  ui/  index.html  app.js  styles.css  config.js  submit/Code.gs  README_UI.md
       data/ (generated: index.json, meta.json, assignments.json, samples/*.json)
       media/ (generated: <sample_id>/contact_sheet.jpg, preview.mp4)
  scripts/ run_all.sh  smoke_test.sh  serve_ui.py  deploy_ui.sh  collect_annotations.py
  tests/   test_adapters.py  test_sampler.py  test_schema.py  test_prompts.py
  data/    (generated, gitignored)  inventory/ sample/ prepared/ cache/ qa/ annotations/
  logs/    (generated, gitignored)
```

Invocation: `cd /mnt/store/tashraf4/projects/ophbench && ./ophbench <stage> [options]`.
All modules import as `bench.<module>`; no module may rely on the CWD except `cli.py`, which
`os.chdir`s to the repo root (the directory containing `bench/`). `config.py` resolves relative
paths in the YAML against the repo root.

## 2. Shared foundation (already written — read before coding: `bench/schema.py`,
`bench/config.py`, `bench/log.py`, `bench/util.py`)

Key types (dataclasses, JSON via `to_dict()/from_dict()`):

* `Segment(label:str, label_id:int|str|None, start_s:float, end_s:float, kind:str='phase', extra:dict={})`
  — `kind` ∈ phase | operation | step | flag.
* `VideoRecord`: `sample_id, dataset, video_id, source_kind ('file'|'zip_member'|'concat_clips'),
  source_paths:list[str], duration_s, fps, width, height, procedure, procedure_category,
  segments:list[Segment], labels:dict, strata:dict, license, citation, notes:list[str]`.
  * `sample_id = f"{dataset}__{safe(video_id)}"` where `safe()` keeps `[A-Za-z0-9_.-]`, replaces
    others with `_`. Examples: `cataract101__case_269`, `cataract1k__case_2003`,
    `lmm_phase__PH_0001_2931_S2`, `lmm_skill__SK_0130_S1_P03`, `lmm_raw__RV_0001_S1`,
    `migs__284_1`, `ophnet__case_0002`, `ophora__X3jTUYMflCk_26`.
  * `source_paths` for `zip_member` = `["<abs zip path>!<member name>"]`; for `concat_clips` =
    ordered list of absolute clip paths; for `file` = one absolute path.
  * `procedure_category` ∈ cataract | glaucoma | cornea | retina | oculoplastics_strabismus |
    refractive | other_mixed.
* `SampleRecord(VideoRecord)` adds `stratum_key:str, selection_reason:str, sample_index:int`.
* `FrameInfo(idx, t_s, file, label_at_t)` and `PreparedSample` (record + probe + frames + media paths +
  `timeline_note`), serialised to `data/prepared/batch_<KKK>/<sample_id>/sample.json`.
* `QA_OUTPUT_SCHEMA` (strict JSON Schema, §6) and `QAItem/QASet` dataclasses.
* `GenerationLogEntry` fields: sample_id, model, route, reasoning_effort, n_frames,
  prompt_tokens, completion_tokens, reasoning_tokens, latency_s, status (ok|error|parse_error),
  error, request_id, est_cost_usd, started_at, finished_at.

Logging (`bench/log.py`): `get_logger(stage)` → RichHandler console + file
`logs/<stage>_<YYYYmmdd_HHMMSS>.log` and refreshes symlink `logs/<stage>.latest.log`;
`JsonlWriter(path)` with `.write(dict)` (append, fsync‑free); `record_run(stage, args, summary)`
appends to `logs/runs.jsonl`. Every stage logs: start, config snapshot (no secrets), per‑item
events, and an end summary with counts.

## 3. Dataset adapters (`bench/datasets/*.py`)

Each adapter subclasses `base.DatasetAdapter` with `name`, `description`, `license`, `citation`,
`iter_records(cfg) -> Iterator[VideoRecord]`, and `stats(records) -> dict`. Adapters must be
pure metadata readers: no ffmpeg, no extraction (except reading zip tables of contents and
small CSV/JSON/XLSX). Register in `datasets/__init__.py: ADAPTERS = {name: cls}` for names
`cataract101, cataract1k, lmm_phase, lmm_skill, lmm_raw, migs, ophnet, ophora`.

### 3.1 cataract101 → 101 videos
* Root `datasets/cataract-101/`. `videos.csv` (`;` sep: VideoID;Frames;FPS;Surgeon;Experience),
  `annotations.csv` (`;`: VideoID;FrameNo;Phase — start frames only, phase runs to next start),
  `phases.csv` (`;`: Phase;Meaning). Videos `videos/case_<id>.mp4` (720×540, 25 fps, h264).
* segments: for each video, sorted by FrameNo; `end_s` = next start −1 frame, last = Frames.
  Also add a leading `Idle/unannotated` segment [0, first start) if first start > 0.
* labels: `surgeon_id`, `experience` ('low' if 1 else 'high'), `frames`, `fps`, `n_phase_segments`.
* strata: `experience`, `surgeon`, `duration_bin` (bins: `<5min, 5-8min, 8-12min, >12min`).
* duration_s = Frames/FPS. procedure = "Cataract surgery (phacoemulsification)", category cataract.

### 3.2 cataract1k → 1000 videos (303 phase‑annotated)
* Videos `datasets/cataract-1k/cat-1k/case_<id>.mp4` (glob `*.mp4`; 640×360 @ ~60 fps on disk).
  Annotations `datasets/cataract-1k/cataract-1k_annotations/{annotations.csv,phases.csv,videos.csv}`
  (`,` sep). `annotations.csv`: VideoID,FrameNo,Phase at **1 fps**, i.e. FrameNo == seconds;
  each row starts a label until the next row; `videos.csv` Frames == duration in seconds for the
  303 annotated videos. Phase ids per `phases.csv` (0 Idle … 13 Suture; 14/15/16 are usage flags
  "Trypan Blue / Iris Hooks / Malyugin Ring Used").
* segments: phases 0–13 as `kind=phase`; rows with 14/15/16 also recorded as `kind=flag`
  segments **and** collected in `labels.flags` (list of names).
* labels: `annotated` (bool), `flags`, `n_phase_segments`, `has_not_cataract`, `has_suture`.
* Unannotated videos: duration unknown at inventory time → `duration_s=None`;
  `inventory --probe-durations` fills it via ffprobe (thread pool). Without probing, use
  `labels.file_size_mb` for `duration_bin` (`size_bin`).
* strata: `annotated`, `rare_flag` (any of flags/suture/not‑cataract), `duration_bin`/`size_bin`.

### 3.3 Cataract‑LMM (`cataract_lmm.py` → three adapters)
Root `datasets/Cataract-LMM/`. Licence CC BY‑NC‑ND 4.0 (note in records).
* **lmm_phase** (150): `1_Phase_Recognition/annotations_full_video/<PH_xxxx_RRRR_Sx>.csv`, columns
  `Video Name,comment,sec,end_sec,frame,end_frame` (comment = phase name, 13 names incl. Idle).
  Video file inside `1_Phase_Recognition/videos/videos_0NN.zip`; mapping in
  `videos/ZipContentsReport.csv` (`ZipFileName,InternalFileName`, stored/uncompressed zips, fast to
  extract a member). source_kind `zip_member`. labels: `site` (S1 Farabi 720×480@30 / S2 Noor
  1080p@60), `raw_video_id`, `idle_share` (Idle seconds / total), `n_segments`, `phase_names`.
  duration_s = max end_sec. strata: `site`, `duration_bin`, `idle_share_bin` (<0.2, 0.2‑0.35, >0.35).
* **lmm_skill** (170 capsulorhexis clips): `4_Skill_Assessment/annotation/skill_scores.xlsx`
  (sheet1 columns: Video_ID, Phase, Microscope Use, Instrument Handling, Tissue Handling, Motion,
  Commencement of Flap, Circular Completion, Averaged, Adverse Events, Comment). Videos in
  `4_Skill_Assessment/videos/videos_partNNN.zip` via `ZipContentsReport.csv` (deflated zips, ~27 MB
  members). 11 clips also exist unzipped at `datasets/LMM-samples/<Video_ID>.mp4` with
  `manifest.csv` (Video_ID, Adverse Events, Comment) → prefer the loose file (source_kind `file`)
  and merge the manifest comment into `labels.adverse_event_comment`. Tracking annotations
  (kinematics) exist at `3_Object_Tracking/annotations/TR_<same index>.zip` → store path in
  `labels.tracking_annotation_zip` (not parsed). labels: six scores, `averaged`, `adverse_event`
  (0/1), `comment`, `site`. procedure = "Cataract surgery — capsulorhexis phase clip".
  strata: `skill_tertile` (Averaged tertiles over the 170: low/mid/high), `adverse_event`, `site`.
* **lmm_raw** (3000, unannotated): `5_Raw_Videos/videos/videos_metadata.csv`
  (`Filename,Duration (s),File Size (MB),Total Frame Count`), files `5_Raw_Videos/videos/<Filename>`
  (S1 is HEVC 720×480@29.97; S2 h264 1080p@60; some > 50 min). labels: `site`, `file_size_mb`,
  `frame_count`, `in_phase_subset` (raw id appears as RRRR in any PH_xxxx_RRRR_Sx name) — the
  sampler excludes those. strata: `site`, `duration_bin`.

### 3.4 migs → 185 glaucoma (MIGS/goniotomy ± phaco) videos
* Root `datasets/MIGS/MIGS/`. `Task_I_annotation.json`: dict keyed by video id (e.g. "16",
  "284_1") → `{annotations:[{label,label_id,segment:[s,e],"segment(frames)":[f0,f1]}], video_id,
  clear, "operation type", knife, "GT incision num", subset, fps, duration, frame_count}`.
  Segments may overlap (Gonioscopy contains Goniotomy). Labels (8): Carbachol injection, OVDs
  injection, Gonioscopy, Goniotomy, OVDs irrigation/aspiration, Would closure (sic → normalise to
  "Wound closure"), Corneal incision by 3.2 mm keratome, Corneal incision by 15 degree keratome.
* File mapping: loose `*.mp4` in the root (10 files) else inside `MIGS_video_dataset_N.zip`
  (deflated). Normalise `basename without .mp4` with `' ' → '_'` to match ids (e.g. `284 1.mp4` ↔
  `284_1`). All 185 ids resolve this way (verified). 21 extra files have no annotation → ignore.
* labels: `operation_type` (codes: GT120/GT240/GT360 = goniotomy degrees; PEI = phaco‑emulsification
  with IOL implantation; GSL = goniosynechialysis; GATT = gonioscopy‑assisted transluminal
  trabeculotomy; SPI = probably surgical peripheral iridectomy — mark "likely"), `knife`,
  `gt_incision_num`, `clear`, `subset`, `fps`, `frame_count`. duration_s from json.
  strata: `operation_type`, `knife`, `duration_bin`. procedure = "Glaucoma MIGS: <operation type>",
  category glaucoma.

### 3.5 ophnet → 743 cases (14,674 phase clips) with names from `resources/ophnet_labels.json`
* Root `datasets/OphNet2024/`. `OphNet2024_loca_challenge_phase.csv` (utf‑8‑sig; columns
  video_id,start,end,split,surgery_id,phase_id): one row per phase clip; the i‑th row of a case
  (file order) is `OphNet2024_trimmed_phase_extracted/OphNet2024_trimmed_phase/<case>/<case>_<i>.mp4`
  (i is 0‑based row index within the case; verified case_0002 has 14 rows & 14 files). Clips are
  1080p ~60 fps h264, mean 22 s, median 11 s, max 583 s. `OphNet2024_loca_all.csv` adds
  `operation_id` rows (finer). `OphNet2024_surgery.csv` (video_id,surgery) gives comma‑separated
  surgery ids per case, first = primary. Names: `ophnet_labels.json` keys `surgery`, `phase`,
  `operation`, `phase_description`, `operation_description` (string keys = ids). Phase id 51 /
  operation id 106 mean "Others (rare)" in the challenge files.
* Unit (config `sampling.ophnet.unit`): **`case`** (default) → one record per case,
  source_kind `concat_clips` with the ordered clip list; segments in the *concatenated* timeline
  (segment k spans the cumulative sum of previous clip durations, using each clip's `end-start`);
  `labels.original_segments` keeps source‑video times; `labels.clips` lists (file, orig_start,
  orig_end, phase_id, phase_name); `labels.operations` = loca_all rows mapped to names;
  `labels.surgery_ids/surgery_names/primary_surgery`, `split`, `n_clips`, `n_phases`. If total
  duration > `max_case_duration_s` (1200) choose the contiguous run of clips with the most
  distinct phases that fits, and note it. `clip` unit → one record per clip (file source).
* category from `resources/procedure_taxonomy.yaml: ophnet_surgery_category` (id → category).
  strata: `primary_surgery`, `procedure_category`, `duration_bin`, `n_phases_bin`.

### 3.6 ophora → 26,592 clean clips (28k subset) from 3,031 YouTube videos
* Root `datasets/Ophora-160K/`. `ophora28k.csv` (`clip id,instruction`); file
  `clips/<clip id>.mp4` (1080p, ~5.5 s; skip ids whose file is missing — 6 are). Source video id =
  `clip_id.rsplit('_',1)[0]`, clip index = suffix. Licence Apache‑2.0 (YouTube‑sourced).
* labels: `instruction`, `source_video_id`, `clip_index`, `in_filtered_28k=True`,
  `instruction_words`, `category_keywords` (matched terms). category via
  `procedure_taxonomy.yaml: ophora_keyword_rules` (ordered first‑match list of
  `{category, any_of:[...]}`; see file). procedure = short label from matched rule.
  strata: `procedure_category`, `instruction_len_bin` (<12, 12‑20, >20 words).
* Duration unknown without probing (assume 5.5 s; `--probe-durations` fills).

## 4. Inventory stage (`bench/inventory.py`) — `./ophbench inventory [--datasets a,b] [--probe-durations] [--workers N]`
* For each adapter: iterate records, write `data/inventory/<dataset>.jsonl` (one VideoRecord per
  line), log per‑dataset counts, missing files (verify every `file` path and zip member exists;
  record `notes` and skip missing with a WARNING), label histograms; write
  `data/inventory/summary.json` {dataset: {n_records, n_with_duration, total_hours, strata
  histograms, notes}} and `data/inventory/summary.md`. `--probe-durations` runs ffprobe (thread
  pool) for records with `duration_s is None` **and** `source_kind == 'file'`, caching results in
  `data/cache/probe_cache.jsonl` keyed by path+size.

## 5. Sampling stage (`bench/sampler.py`) — `./ophbench sample [--target N] [--seed S] [--dry-run]`
Deterministic (`random.Random(seed)`), fully logged. Config (`config/pipeline.yaml: quotas, sampling`).
1. Load inventories; build each dataset's **eligible pool** using per‑dataset filters:
   * cataract1k: pool = all; target split `annotated_share` (0.87 of quota from the 303 annotated,
     oversampling `rare_flag=True`: include all rare‑flag videos first); remainder unannotated.
   * lmm_skill: include **all** `adverse_event==1` first; then balance `skill_tertile × site`.
   * lmm_raw: exclude `in_phase_subset`; duration in [min_duration_s, max_duration_s]; at least
     `s2_min` S2 videos (all 70 S2 are eligible candidates regardless of duration if needed).
   * migs: every `operation_type` represented at least `min_per_operation_type`; then balance
     `operation_type × knife`.
   * ophnet: one record per case already; ensure every `primary_surgery` with ≥1 case gets ≥1,
     then allocate by sqrt‑proportional weights over `primary_surgery`, prefer cases with
     `n_phases ≥ 3` and duration 60–1200 s.
   * ophora: one clip per `source_video_id`; `min_instruction_words` filter; allocate over
     `procedure_category` with `category_weights` (default up‑weights non‑cataract categories).
   * cataract101 / lmm_phase: quota ≥ pool → take all; else balance strata.
2. Generic allocator `allocate(strata_sizes, quota)`: floor of 1 per non‑empty stratum (if quota
   allows), remaining proportional to `sqrt(size)`, largest‑remainder rounding, then cap by
   stratum size and redistribute leftovers; sample without replacement within stratum.
3. Shortfall: if a dataset's pool < quota, redistribute the deficit to other datasets
   proportionally to their remaining pool; iterate until target met or pools exhausted; log it.
4. Outputs: `data/sample/sample_manifest.jsonl` (SampleRecord), `data/sample/sample_manifest.csv`
   (flat: sample_id,dataset,video_id,procedure,procedure_category,duration_s,stratum_key,
   selection_reason,source_kind,first_source_path), `data/sample/sampling_report.json` and
   `sampling_report.md` (per dataset: pool, quota, selected, per‑stratum table pool→selected,
   redistribution log, seed, config snapshot), `logs/sample*.log`.

## 6. Prepare stage (`bench/prepare.py`, `bench/media.py`) — `./ophbench prepare [--workers 12] [--no-preview] [--ids a,b] [--datasets ..] [--limit N] [--force]`
Per sample dir `data/prepared/batch_<KKK>/<sample_id>/`:
1. **Materialise** `source.mp4`: `file` → symlink; `zip_member` → extract into
   `data/cache/extracted/<dataset>/<member>` (skip if exists & size matches) then symlink;
   `concat_clips` → ffmpeg concat demuxer with `-c copy` (write `concat_list.txt`), fall back to
   re‑encode (`libx264 -crf 20 -preset veryfast`) if copy fails; write `concat_map.json`
   `[{clip, offset_s, duration_s}]`. Verify with ffprobe; actual offsets come from probing each clip.
2. **Probe** → `probe.json` {duration_s, fps, width, height, codec, nb_frames, size_bytes}.
3. **Frames**: `n = clamp(round(duration_s / media.frames.seconds_per_frame), min_frames, max_frames)`
   (defaults 10 s/frame, 12..48). Timestamps = uniform grid at `(i+0.5)*duration/n`, plus
   (when segments exist) each segment start + 1.0 s and each segment midpoint, deduplicated within
   1.5 s and trimmed to `max_frames` by dropping grid points nearest to boundary frames. Extract
   with input seeking: `ffmpeg -ss T -i source.mp4 -frames:v 1 -vf scale='min(768,iw)':-2 -q:v 3`
   (long side ≤ `resize_long_side`). Overlay with PIL (bottom‑left, black box, white text, font
   DejaVuSans if found else default): `t=MM:SS.s  (frame k/n)` and, if known, the phase label at
   t. Save `frames/f_{idx:03d}_{t:07.1f}s.jpg`; `frames_index.json` = list of FrameInfo.
4. **Contact sheet** `contact_sheet.jpg`: grid `cols=6`, tile width 320, with the same overlays.
5. **Preview** `preview.mp4` (unless `--no-preview`): `-vf scale=-2:360 -c:v libx264 -crf 28
   -preset veryfast -pix_fmt yuv420p -an -movflags +faststart`; cap at `media.preview.max_duration_s`
   if set.
6. **sample.json** (PreparedSample): record + probe + frames + `timeline_note` (e.g. "times refer
   to the concatenated case video; original source times in labels.original_segments").
7. Append `data/prepared/_status.jsonl` {sample_id, ok, steps_done, error, elapsed_s, ts};
   skip samples whose `sample.json` exists unless `--force`. Thread pool over samples; ffmpeg
   `-threads 2` per job. Per‑sample `prepare.log` with the exact ffmpeg commands.

## 7. Generation stage (`bench/llm_client.py`, `bench/prompts.py`, `bench/generate.py`) — `./ophbench generate [--concurrency 4] [--limit N] [--ids] [--datasets] [--dry-run] [--force] [--route rest_chat|responses] [--effort xhigh]`
**Client** `GatewayClient(cfg.llm)`:
* `complete(messages_or_input, *, images:list[Path], schema:dict|None, reasoning_effort, max_tokens) -> LLMResult(text, parsed, usage, latency_s, request_id, route, raw)`.
* Route `rest_chat` (text‑only): URL `{base}/rest/v1/chat/completions`, body
  `{model: cfg.model (prefixed), messages, max_completion_tokens, reasoning_effort, response_format}`.
  If images are passed with this route the client must raise `UnsupportedInput` (the gateway
  rejects every image part form on this route).
* Route `responses`: URL `{base}/codex/openai/v1/responses`, body `{model: unprefixed, input:[{role,
  content:[{type:input_text}|{type:input_image,image_url,detail}]}], reasoning:{effort},
  max_output_tokens, text:{format:{type:json_schema,name,strict,schema}}, stream:true}`; parse
  SSE lines `data: {...}` and take the `response.completed` event's `response` (fallback to
  `stream:false` JSON). Text = concatenated `output_text` parts of `message` items.
* Retries (tenacity): up to `max_retries` with exponential backoff + jitter on timeouts, 429, 5xx,
  connection errors; honour `Retry-After`. On HTTP 400 mentioning `reasoning_effort` →
  step down effort (`max→xhigh→high`) once and log a WARNING. On JSON parse failure → one
  repair attempt by re‑asking with the invalid text and "return only valid JSON matching the
  schema". Never log the key; log requests with base64 replaced by `<image N bytes>`.
* `estimate_cost(usage)` using optional `price_per_m_input_usd/output` (None → null).
* `probe()` → models endpoint (`GET {base}/api/models`), a text call, and a 1‑image call on the
  configured route; prints a table and exits non‑zero if the model is unreachable. Exposed as
  `./ophbench probe`.

**Prompt** (`prompts.py` builds from `config/prompts/*.md`):
* System: expert ophthalmic surgeon + benchmark designer for **agentic** video AI. Define agentic
  (multi‑step reasoning across time; tool use such as seeking/zooming/counting/measuring/looking up
  labels or guidelines; planning; verification) and tough (not answerable from one frame, from the
  caption alone or from the metadata alone; ≥3 reasoning steps; clinically meaningful; a
  clinician can verify from video + labels). Require exactly 3 questions with **distinct**
  categories, ≥1 temporally grounded (timestamp/duration/ordering), ≥1 requiring clinical
  judgment (complication, deviation, decision), varied `answer_type`, ≤1 multiple_choice (4–5
  options, exactly one correct, answer gives letter + text). Ground claims in the frames and the
  provided labels; cite times as ranges (frames are sparse); never invent events; keep patient
  anonymity; output JSON only.
* User: dataset blurb + licence; video metadata (procedure, category, duration, resolution,
  site/surgeon/experience, skill scores, adverse events, flags, operation type, instruction);
  label timeline table (name | start | end | duration) in the displayed timeline + phase glossary
  (ophnet descriptions when available); timeline note; then the frames (text part
  `Frame k/n — t=MM:SS.s — label` followed by the image part); final instruction to produce JSON
  per schema. Dry‑run writes the text part + frame list to `data/qa/dryrun/<id>.json`.

**QA_OUTPUT_SCHEMA** (strict, `additionalProperties:false` everywhere):
```
{video_summary: str,
 questions: [3 × {qid: 'q1'|'q2'|'q3', category: enum, question: str, answer: str,
   answer_rationale: str, evidence_timestamps: [{start_s: number, end_s: number, observation: str}],
   agentic_skills: [enum], tool_plan: [str], answer_type: enum, options: [str] (empty if n/a),
   difficulty: 'hard'|'very_hard', why_hard: str, metadata_used: [str], confidence: number}],
 generator_notes: str}
category enum: temporal_grounding, workflow_deviation, complication_detection_management,
  next_step_prediction, skill_assessment_evidence, instrument_anatomy_reasoning,
  quantitative_estimation, counterfactual_decision, guideline_cross_reference, multi_segment_comparison
agentic_skills enum: temporal_localization, counting, measurement_estimation, phase_recognition,
  instrument_recognition, anatomy_recognition, causal_reasoning, planning, external_knowledge_retrieval,
  calculation, comparison_across_segments, anomaly_detection, decision_under_uncertainty, verification
answer_type enum: free_text, timestamp, duration, count, boolean, multiple_choice, list, ranking
```
(Strict mode requires every property in `required`; use `options: []` when not multiple choice.)

**Generate loop**: load manifest ∩ prepared; filter; ThreadPoolExecutor(concurrency); per sample:
build prompt, call client (effort from config, default `xhigh`), save
`data/qa/raw/<id>.request.json` (no base64), `data/qa/raw/<id>.response.json`, parsed
`data/qa/<id>.json` (= QASet + provenance {model, route, effort, n_frames, usage, latency,
generated_at, prompt_version}); append `data/qa/generation_log.jsonl`; rebuild
`data/qa/qa_all.jsonl` at the end; summary (ok/parse_error/error counts, tokens, est cost,
mean latency) to log and `logs/runs.jsonl`. Resumable.

## 8. Validate stage (`bench/validate.py`) — `./ophbench validate`
Checks per QASet: exactly 3 questions; distinct categories; answers non‑empty and not contained
verbatim in the question; evidence timestamps within [0, duration+1]; question dedupe (token
Jaccard < 0.6 within a video); multiple_choice has 4–5 options and the answer matches one;
confidence in [0,1]; flags `low_confidence (<0.5)`. Writes `data/qa/validation_report.json`
(+ `.md`) and `data/qa/qa_validated.jsonl` with `validation: {ok, issues:[...]}` per item.

## 9. Export + UI (`bench/export_ui.py`, `ui/`) — `./ophbench export-ui [--media-mode symlink|copy|none] [--annotators a,b,c] [--overlap 0.1]`
* Writes `ui/data/meta.json` {generated_at, n_samples, datasets, config snapshot (ui section)},
  `ui/data/index.json` [{sample_id, dataset, procedure, procedure_category, duration_s,
  n_questions, thumb: 'media/<id>/contact_sheet.jpg', assigned_to:[...]}],
  `ui/data/samples/<id>.json` {sample_id, dataset, dataset_blurb, citation, video: {preview_url,
  contact_sheet_url, duration_s, width, height}, metadata (clinician‑friendly key/values),
  segments (name,start_s,end_s), frames (t_s, label), questions (from validated QA incl.
  validation issues), timeline_note}, `ui/data/assignments.json` {annotator: [sample_ids]} using
  round‑robin with `overlap` fraction assigned to everyone (for agreement).
* Media: copy/symlink `contact_sheet.jpg` and `preview.mp4` to `ui/media/<id>/`. URLs in JSON are
  **relative** (`media/<id>/preview.mp4`); the UI prefixes `MEDIA_BASE_URL` from `ui/config.js`
  when set (so media can live on another host when the UI is on GitHub Pages).
* UI (vanilla HTML/JS/CSS, no build, works on GitHub Pages at any sub‑path; all fetches relative):
  * Landing: annotator name/ID (localStorage), list of assigned samples with status
    (todo/in‑progress/done), filters (dataset, status), progress bar, Export JSON, Import JSON,
    Submit (POST to `SUBMIT_URL` from config.js when non‑empty; shows result).
  * Sample view (`#/s/<sample_id>`): video player (preview; falls back to contact sheet if the
    file 404s), clickable phase timeline bar under the player (seek on click), metadata panel
    (collapsible), the 3 question cards: question, answer, rationale, evidence chips (click → seek),
    badges (category, difficulty, answer type, agentic skills); per‑question controls:
    **Select for benchmark** (checkbox; enforce `questions_to_select_min..max` before "Mark
    complete"), answer correctness (correct / partially correct / incorrect / cannot verify),
    clinical relevance 1–5, difficulty 1–5, agentic/multi‑step 1–5, clarity 1–5, editable
    question and answer textareas (prefilled; edited flag), comment. Sample‑level: flags
    (video quality issue, metadata wrong, not suitable, duplicate), free comment, "Mark complete",
    prev/next/next‑unfinished; keyboard j/k, 1/2/3 toggles select. Autosave to localStorage key
    `ophbench.ann.<annotator>` as `{sample_id: {...}}` with timestamps and `ui_version`.
  * `ui/config.js`: `window.OPHBENCH_CONFIG = {MEDIA_BASE_URL: "", SUBMIT_URL: "", SELECT_MIN:1, SELECT_MAX:2}`.
  * `ui/submit/Code.gs`: Google Apps Script web app (doPost → append rows to a Sheet, one row per
    question annotation + one per sample) with setup steps in `ui/README_UI.md`.
* `scripts/serve_ui.py`: static server for `ui/` with HTTP Range support (video seeking), CORS,
  and `POST /api/annotations` saving JSON to `data/annotations/<annotator>_<ts>.json` (for
  local/VPN use). `scripts/deploy_ui.sh`: initialises a git repo in `ui/` (or a `gh-pages`
  worktree), commits, and prints the push + Pages instructions (`GITHUB_REPO=user/repo`).
* `scripts/collect_annotations.py`: merges exported/received JSON files (and optionally a Sheet
  CSV) → `data/annotations/merged.jsonl`, computes per‑video selections, selection agreement
  between annotators, per‑dataset/category counts → `data/annotations/summary.md`.

## 10. Status + run‑all
`./ophbench status` prints a table: inventory counts, sampled per dataset, prepared ok/err,
generated ok/parse_error/error, validated ok, exported; tokens + est cost so far.
`scripts/run_all.sh` runs inventory → sample → prepare → generate → validate → export-ui with
`set -euo pipefail`, logging to `logs/run_all_<ts>.log`. `scripts/smoke_test.sh` runs the whole
chain on `--limit 2` per dataset using a temporary `data_dir` (`OPHBENCH_DATA_DIR=data_smoke`).

## 11. Config file (`config/pipeline.yaml`) — authoritative keys
```yaml
seed: 20261005
datasets_root: /mnt/store/tashraf4/datasets
data_dir: data                # override with env OPHBENCH_DATA_DIR
logs_dir: logs
ffmpeg: auto
ffprobe: auto
workers: 12
target_total: 1500
quotas: {cataract101: 101, cataract1k: 300, lmm_phase: 150, lmm_skill: 120, lmm_raw: 80, migs: 160, ophnet: 420, ophora: 169}
sampling:
  duration_bins_s: [300, 480, 720]            # <5, 5-8, 8-12, >12 min
  cataract1k: {annotated_share: 0.87, rare_flag_first: true}
  lmm_skill: {include_all_adverse: true}
  lmm_raw: {min_duration_s: 300, max_duration_s: 1500, s2_min: 30, exclude_phase_subset: true}
  migs: {min_per_operation_type: 1}
  ophnet: {unit: case, max_case_duration_s: 1200, min_case_duration_s: 60, min_phases: 3, max_per_case: 1}
  ophora: {one_clip_per_source_video: true, min_instruction_words: 12,
           category_weights: {cataract: 1.0, glaucoma: 2.0, cornea: 2.0, retina: 1.5, oculoplastics_strabismus: 2.0, refractive: 1.5, other_mixed: 1.0}}
media:
  frames: {seconds_per_frame: 10, min_frames: 12, max_frames: 48, resize_long_side: 768, jpeg_quality: 85, boundary_frames: true}
  contact_sheet: {cols: 6, tile_width: 320}
  preview: {enabled: true, height: 360, crf: 28, max_duration_s: null}
llm:
  base_url: https://gateway.engineering.jhu.edu/gateway
  api_key_env: GPT_ASTRA_KEY
  api_key_file: /mnt/store/tashraf4/.api_keys.env
  route: responses                 # responses (required for image input) | rest_chat (text-only)
  model: openai/gpt-6-astra        # responses route strips the provider prefix automatically
  reasoning_effort: xhigh          # maximum accepted by this model on the gateway
  max_completion_tokens: 32000
  image_detail: high
  stream: true                     # responses route: parse SSE (robust); false = broker-synthesised JSON
  concurrency: 4
  timeout_s: 1200
  max_retries: 5
  questions_per_video: 3
  prompt_version: v1
  price_per_m_input_usd: null
  price_per_m_output_usd: null
ui:
  media_base_url: ""
  submit_url: ""
  questions_to_select_min: 1
  questions_to_select_max: 2
  annotators: []
  overlap_fraction: 0.1
```

## 12. Conventions
* Python 3.11+ syntax, type hints, dataclasses, `pathlib`. No network calls outside `llm_client.py`.
* Every stage: `argparse` sub‑command in `cli.py` delegating to `<module>.main(args, cfg)`; return
  code 0/1; all outputs deterministic given seed; idempotent and resumable.
* Logs are human readable; machine logs are JSONL. Never write secrets to any file.
* Tests must not touch the network and must finish in < 2 min (they may read dataset CSVs).

## 13. Addendum (2026‑10‑05, after user feedback): batches + offline video sharing

The clinicians receive the videos **offline** (shared drive / USB) and annotate in **batches of ~100**.
Media is therefore not hosted; the UI must play local files. This changes §5, §7 and §9 as follows.

### 13.1 Canonical benchmark order and batches (`bench/batches.py`)
* `order_samples(records) -> list[SampleRecord]`: deterministic proportional interleave across datasets.
  For each dataset keep manifest order (then sample_id) and give item *i* of *n* the position
  `(i + 0.5) / n`; sort globally by `(position, dataset, sample_id)`. A batch of 100 then contains roughly
  quota/15 of each dataset. Write the resulting rank into `SampleRecord.sample_index` (0‑based) when the
  sampler runs; downstream stages re‑derive the same order with this function if `sample_index` is missing.
* `select_batch(ordered, batch: int, batch_size: int) -> list` returns items `[(batch-1)*size, batch*size)`
  (1‑based batch numbers). `prepare`, `generate` and `export-ui` accept `--batch K --batch-size N`
  (default batch size from `config: batches.size`, 100); `--limit` keeps working.
* `status` reports progress per batch (prepared / generated / validated / exported).

### 13.2 Export changes (`bench/export_ui.py`)
* `export-ui --batch K` is **additive**: `ui/data/index.json` keeps previously exported batches and each
  entry carries `"batch": K`; `ui/data/meta.json` gains `"batches": {"1": {"n": 100, "exported_at": ...}}`
  and `"offline_media": true`; `ui/data/samples/<id>.json` gains `"batch": K` and
  `"video": {..., "local_file": "<sample_id>.mp4", "preview": null}` (preview stays non‑null only when
  `--media-mode symlink|copy` hosting is requested explicitly). Contact sheets are still copied to
  `ui/media/<id>/contact_sheet.jpg` (small enough for GitHub Pages).
* **Offline package**: unless `--no-package`, write `ui_packages/batch_<K:03d>/` with one
  `<sample_id>.mp4` per sample (`--package-kind preview` (default, 360p) | `source` (the prepared
  `source.mp4`, i.e. original or concatenated case video)), `MANIFEST.csv`
  (sample_id, dataset, procedure, duration_s, file, size_mb, batch) and `README.txt` (what the folder is,
  how to load it in the UI, do not rename files). Log total size. The user zips and shares this folder.

### 13.3 UI changes (`ui/app.js`, `ui/index.html`)
* Landing page: batch filter (from `meta.batches`), per‑batch progress, and a **"Load videos folder"**
  button. Use the File System Access API (`window.showDirectoryPicker`) when available and persist the
  directory handle in IndexedDB (re‑request permission on load); fall back to
  `<input type="file" webkitdirectory multiple>` elsewhere (must be re‑selected per session; say so).
  Index the chosen folder by file name stem → File; show "N of M videos of this batch found".
* Sample view: if a local file matching `video.local_file` (or `<sample_id>` + any of .mp4/.mov/.mkv/.webm)
  is loaded, play it through `URL.createObjectURL` (revoke on navigation); else if a hosted `preview` URL
  exists use it; else show the contact sheet with the hint "load the batch folder to play the video".
  Everything else (timeline bar seeking, evidence chips, question cards, autosave, export, submit) is
  unchanged. Annotation records gain `"batch": K` per sample.
* Nothing is uploaded: local files never leave the browser. State this in the help panel.

### 13.4 Config additions (`config/pipeline.yaml`)
```yaml
batches:
  size: 100
  package_kind: preview        # preview | source
```

### 13.5 Implementation notes (as built)
* The benchmark order is always re‑derived with `order_samples()`; `sample_index` in the manifest is informative.
* `export-ui --batch K` defaults to `--media-mode sheets` (contact sheets only); assignments for batch K are
  seeded with `seed + K` and merged with earlier batches; packages go to `ui_packages/` (override with
  `--packages-dir` or env `OPHBENCH_PACKAGES_DIR`).
* Re‑running `prepare` on an already prepared sample without `--no-preview` only adds the missing preview
  (status rows carry `mode: preview_only`).
* `scripts/smoke_test.sh` defaults to `data_smoke/` (also used by the integration run); pass
  `OPHBENCH_DATA_DIR` for an isolated run. Data facts: Cataract‑1K has 247 (not 303) annotated videos on
  disk (ids 4687–5357 absent) and no flag rows; all Cataract‑LMM skill clips are site S1.
* Prepared samples live in per-batch folders `data/prepared/batch_<KKK>/<sample_id>/` (batch from the
  benchmark order and `config: batches.size`); every stage resolves the folder with
  `bench.batches.prepared_dir()`, which still finds legacy flat `data/prepared/<sample_id>/` folders.
  `data/prepared/_status.jsonl` stays at the root.
