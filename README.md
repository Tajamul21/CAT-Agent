# ophbench — Stage‑1 curation pipeline for an agentic ophthalmic‑surgery video benchmark

Turns the public datasets under `/mnt/store/tashraf4/datasets` into a 1,500‑video benchmark seed:
**inventory → intelligent sampling → media preparation → GPT‑6 Astra question generation (3 tough,
agentic Q&A per video, maximum reasoning) → validation → a static clinician annotation UI** that can be
hosted on GitHub Pages. Every stage writes machine logs (JSONL) and human logs under `logs/`.
Design contract: [`DESIGN.md`](DESIGN.md).

## 1. Prerequisites (already satisfied on this server)

| Item | Where |
|---|---|
| Python 3.14 + packages (`httpx pyyaml pandas numpy pillow openpyxl imageio-ffmpeg tenacity tqdm rich pytest`) | `/mnt/store/tashraf4/miniconda3/bin/python3` (`pip install -r requirements.txt` on another machine) |
| ffmpeg 7 (static) | bundled by `imageio-ffmpeg`; ffprobe from `miniconda3/envs/opencompass/bin/ffprobe` (auto‑detected, see `config/pipeline.yaml: ffmpeg/ffprobe`) |
| Gateway key | `/mnt/store/tashraf4/.api_keys.env` → `export GPT_ASTRA_KEY=jhu_live_sk_…` (WSE AI Gateway key scoped to `openai/gpt-6-astra`) |
| Datasets | `/mnt/store/tashraf4/datasets/{cataract-101,cataract-1k,Cataract-LMM,MIGS,OphNet2024,Ophora-160K,LMM-samples}` |

The `./ophbench` wrapper sources the key file and runs `python3 -m bench.cli`.

## 2. Quick start

```bash
cd /mnt/store/tashraf4/projects/ophbench
./ophbench probe                      # gateway reachable? model id? 1-image call works?
./ophbench inventory                  # scan all datasets -> data/inventory/*.jsonl + summary.md
./ophbench sample                     # 1,500 videos -> data/sample/sample_manifest.{jsonl,csv} + sampling_report.md
./ophbench prepare --workers 12       # frames, contact sheets, previews -> data/prepared/batch_<KKK>/<sample_id>/
./ophbench generate --concurrency 4   # GPT-6 Astra, reasoning_effort=xhigh -> data/qa/<sample_id>.json
./ophbench validate                   # schema/quality checks -> data/qa/qa_validated.jsonl + validation_report.md
./ophbench export-ui --batch 1 --annotators drA,drB   # batch 1 (first 100) -> ui/data, ui/media, ui_packages/batch_001/
./ophbench status                     # progress table for every stage and batch
python3 scripts/serve_ui.py --port 8765                           # preview the UI locally before pushing it
```

### Working in batches of 100 (clinicians get the videos offline)
Samples have a fixed *benchmark order* that interleaves the datasets, so every batch of 100 spans all eight
datasets. Any stage can be restricted to a batch:
```bash
./ophbench prepare  --batch 1            # first 100 in benchmark order
./ophbench generate --batch 1            # 100 real calls
./ophbench validate
./ophbench export-ui --batch 1 --annotators drA,drB
#   -> ui/data/* (hosted UI, cumulative over batches), ui/media/<id>/contact_sheet.jpg (small),
#   -> ui_packages/batch_001/<sample_id>.mp4 + MANIFEST.csv + README.txt  (zip and share offline)
./ophbench export-ui --batch 2 ...       # later batches are added to the same UI
./ophbench export-ui --batch 1 --package-kind source   # ship original-resolution videos instead of 360p previews
./ophbench export-ui --batch 1 --no-package            # UI data only (package already shared)
./ophbench export-ui                     # no --batch: full rebuild of ui/data with hosted previews (symlinks), no package
```
The benchmark order is written into the manifest (`sample_index` = rank), so `--batch K --batch-size N` means
rows `[(K-1)*N, K*N)` of `data/sample/sample_manifest.jsonl` in every stage (`batches.size` in the YAML, default 100).
Offline packages go to `ui_packages/batch_<K>/` (override with `OPHBENCH_PACKAGES_DIR` or `--packages-dir`).
Clinicians open the hosted UI, click **Load videos folder**, pick the shared `batch_001` folder once, and the
videos play from their own disk (nothing is uploaded). Without the folder the UI shows the frame contact sheet.

`scripts/run_all.sh` runs the whole chain; `scripts/smoke_test.sh` runs it on a handful of samples in
`data_smoke/` (dry‑run generation, no API cost; set `OPHBENCH_DATA_DIR` to run it elsewhere — it overwrites the
manifest in that folder). Each stage is **resumable**: re‑running skips finished items (`--force` redoes them),
`prepare` adds missing previews to samples prepared with `--no-preview`, and `--limit/--ids/--datasets` restrict any stage.

### Try it small first
```bash
./ophbench prepare --limit 10 --no-preview
./ophbench generate --dry-run --limit 10          # writes the exact prompts to data/qa/dryrun/ (no API call)
./ophbench generate --limit 3                      # 3 real calls at xhigh effort
./ophbench validate && ./ophbench export-ui && ./ophbench status
```

## 3. What "intelligent sampling" does (`config/pipeline.yaml: quotas, sampling`)

| Dataset | Pool | Quota | Stratification / rules |
|---|---|---|---|
| cataract‑101 | 101 full surgeries | 101 | all (surgeon × experience × duration) |
| Cataract‑1K | 1,000 (303 phase‑annotated) | 300 | 87 % annotated, rare flags first (trypan blue, iris hooks, Malyugin ring, suture, "not cataract"), rest unannotated |
| Cataract‑LMM phase | 150 full surgeries, 2 hospitals | 150 | all (site × duration × idle share) |
| Cataract‑LMM skill | 170 capsulorhexis clips with 6 expert scores | 120 | all 12 adverse‑event clips, then skill tertile × site |
| Cataract‑LMM raw | 3,000 unannotated | 80 | exclude videos already in the phase subset, 5–25 min, ≥ 30 from site S2 (1080p) |
| MIGS | 185 glaucoma surgeries | 160 | every operation type ≥ 1, then operation type × knife |
| OphNet‑2024 | 743 cases (phase clips re‑joined into one case video) | 420 | every surgery type ≥ 1, sqrt‑proportional over surgery type, ≥ 3 phases, 1–20 min |
| Ophora‑160K | 26,592 clean 5‑s YouTube clips | 169 | one clip per source video, up‑weight glaucoma/cornea/retina/oculoplastics vs cataract |

Shortfalls are redistributed across datasets and logged. `data/sample/sampling_report.md` shows pool → selected per stratum.
Change quotas or rules in the YAML and re‑run `sample` (deterministic for a given `seed`).

## 4. What the model receives and returns

* Up to 48 timestamp‑stamped frames (≈ 1 per 10 s, denser at phase boundaries), plus the dataset's labels
  (phase timeline, skill scores, adverse events, operation type, caption…) rendered as text. The gateway
  rejects raw video, so frames are the input; previews are only for the clinician UI.
* Route: Responses API at `…/gateway/codex/openai/v1/responses`, `reasoning.effort = xhigh` (the gateway rejects
  `max`; `xhigh` is the maximum for this model), strict JSON schema (`bench/schema.py: QA_OUTPUT_SCHEMA`).
* Output per video (prompt v2, situation awareness after Endsley 1995): `video_summary` plus exactly 3 questions,
  **q1 L1 Perception, q2 L2 Comprehension, q3 L3 Projection** ("forecast first, then verify"), each on a different
  phase family, with mixed answer formats (at least one one-word / short closed answer and one multi-line answer),
  at most one timing question, closed answer sets with "not determinable" where the view may not settle it. Each
  question carries `sa_level`, `family`, `clinical_use`, `answer_type`, `answer_length`, options, gold answer,
  rationale, evidence time ranges, agentic skills, tool plan, difficulty, why it is hard, why a surgeon cares, the
  likely agent failure and confidence. The v1 prompt is kept in `config/prompts/v1/`.
* Prompts live in `config/prompts/system.md` and `user_template.md`; bump `llm.prompt_version` when you edit them.

**Time and cost.** One xhigh call with ~40 frames takes 2–10 min and 30–80k tokens. With `--concurrency 4`
expect roughly 2–4 days for 1,500 videos; raise concurrency (gateway allows ~60 req/min per key) or use
`--effort high` for a faster first pass. Fill `llm.price_per_m_*_usd` to get cost estimates in `status` and the
generation log. The gateway enforces a monthly spend cap per key; a capped key returns an error that `generate`
logs and skips, so re‑run later to resume.

## 5. Outputs and logs

All data lives **outside the code repo** in `/mnt/store/tashraf4/projects/ophbench_data/` (`data_dir`, `logs_dir` and `packages_dir` in `config/pipeline.yaml`); below, `data/` means that folder, `logs/` is `ophbench_data/logs/` and `ui_packages/` is `ophbench_data/packages/`.

```
data/inventory/<dataset>.jsonl, summary.{json,md}      every candidate video with labels and strata
data/sample/sample_manifest.{jsonl,csv}, sampling_report.{json,md}
data/prepared/batch_<KKK>/<sample_id>/ source.mp4 probe.json frames/ frames_index.json contact_sheet.jpg preview.mp4 sample.json prepare.log
data/prepared/_status.jsonl                            per-sample prepare status
data/qa/<sample_id>.json                               parsed Q&A set with provenance (model, effort, usage, latency)
data/qa/raw/<sample_id>.{request,response}.json        exact request (images redacted) and raw response
data/qa/generation_log.jsonl, qa_all.jsonl, qa_validated.jsonl, validation_report.{json,md}
ui/data/{meta,index,assignments}.json, ui/data/samples/<sample_id>.json, ui/media/<sample_id>/
logs/<stage>_<timestamp>.log (+ <stage>.latest.log), logs/runs.jsonl
```

## 6. Clinician annotation UI (`ui/`)

Static site (no build step). Clinicians enter an annotator ID, pick their batch, see their assigned videos,
load the offline video folder you shared (played locally in the browser; falls back to the contact sheet),
read the 3 generated questions, **select 1–2 per video**, rate correctness / relevance / difficulty /
agentic‑ness / clarity, edit question or answer text, flag problems, and mark complete. Progress is saved in
the browser; **Export JSON** downloads it and **Submit** posts it to an endpoint if one is configured.

Deploy once, then re‑export per batch:
1. `./ophbench export-ui --batch 1 --annotators drA,drB` writes `ui/data`, `ui/media` (contact sheets only,
   ≈ 100–200 KB per video) and the offline package `ui_packages/batch_001/` (≈ 2–4 GB of 360p previews per
   100 videos; `--package-kind source` for the original‑resolution videos).
2. Annotation collection: deploy `ui/submit/Code.gs` as a Google Apps Script web app (Anyone can access) and
   put its URL in `ui/config.js: SUBMIT_URL`, or simply collect the exported JSON files.
   `scripts/collect_annotations.py` merges either into `data/annotations/merged.jsonl` with agreement statistics.
3. `GITHUB_REPO=<user>/<repo> bash scripts/deploy_ui.sh` prepares the git repo in `ui/`; push it from a
   machine with GitHub access (this server has no GitHub credentials), enable Pages, share the link together
   with the zipped `batch_001` folder. For later batches re‑run step 1 with `--batch 2` and push again.
Details and the clinician guide: `ui/README_UI.md`.

## 6b. Verified end to end (2026‑10‑05 integration run on `data_smoke/`)
inventory (31,941 records, 0 missing media) → sample (1,500 = 101/300/150/120/80/160/420/169, no shortfall) →
prepare 2 per dataset (16/16 ok, 12–48 frames, zip members and an OphNet concat case included) → generate dry‑run (16
prompts) and 2 real GPT‑6 Astra calls (`--effort medium`: 60 s and 82 s, 3 valid questions each) → validate (2/2 ok) →
export‑ui → `serve_ui.py` (Range requests, POST annotations) → `smoke_test.sh` PASS; `pytest tests/` green.
Not yet exercised on real data: the full 1,500 prepare/generate run, `--package-kind source` for multi‑GB lmm_raw videos,
and the browser UI (verified headless with a fake‑DOM harness only — do a 10‑minute manual pass in Chrome/Firefox before
sending links to clinicians).

## 7. Stage 2 hook
`data/annotations/merged.jsonl` (clinician selections + edits) joined with `data/qa/qa_validated.jsonl` and
`data/prepared/batch_<KKK>/<id>/sample.json` is the input for stage 2 (e.g. building the final benchmark items / agent evaluation).

## 8. Troubleshooting
* `MODEL_NOT_ALLOWED_FOR_KEY` — enable `openai/gpt-6-astra` on both the project and the key in the gateway UI.
* HTTP 401 "You didn't provide an API key" — you are on the `compat`/`openai` route; use `responses` or `rest_chat` (config `llm.route`).
* `reasoning_effort does not support 'max'` — use `xhigh` (default).
* Preview generation is the slowest step (1080p60 sources). Run `prepare --no-preview` first to unblock generation, then run
  `prepare` again without the flag later; it only adds the missing previews (no `--force` needed).
* Zip extraction cache: `data/cache/extracted/` (Cataract‑LMM and MIGS members); safe to delete after `prepare`.

## 9. Dataset licences and citations
See `bench/schema.py: DATASET_META`. Cataract‑LMM is CC BY‑NC‑ND (no public redistribution of derived clips);
Cataract‑1K requires Synapse registration; Ophora clips are YouTube‑sourced. Share the UI/media with clinicians
privately and cite every dataset in publications.
