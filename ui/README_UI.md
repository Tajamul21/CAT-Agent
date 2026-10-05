# CAT-Agent clinician review app

A single web page (`index.html`, `app.js`, `styles.css`, `config.js`) hosted on GitHub Pages. It contains
**no data**: each clinician opens the batch folder you send them, and everything is read locally in
their browser. Nothing is uploaded.

## What you send each clinician
1. The link: https://tajamul21.github.io/CAT-Agent/
2. The zipped batch folder from `ui_packages/batch_001/` (made by `./ophbench export-ui --batch 1`).
   It holds, for every video, `<sample_id>.mp4` and `<sample_id>.qa.json` (the GPT questions).

## What the clinician does
1. Unzip the folder. Open the link in Chrome or Edge (best: progress is also saved inside the batch folder), or Firefox/Safari.
2. Type a **name or ID**. It is required: nothing is saved without it, and progress is stored under it.
3. Click **Choose batch folder** (or drag the folder onto the page) and pick the unzipped folder. In Chrome/Edge allow
   the site to save changes to the folder when asked.
4. For each video: watch it (time chips jump to the evidence), mark the best question(s) with ★ (one or two), set
   whether the answer is correct, use **Edit question** / **Edit answer** to fix wording (the GPT original is kept and
   shown under the edit), then **Done & next**.
5. Progress saves automatically after every change: in the browser, and in Chrome/Edge also as
   `catagent_progress_<ID>.json` inside the batch folder. **Save progress** (or Ctrl/Cmd+S) saves immediately; when the
   folder is not writable it downloads that file instead: put it into the batch folder and it loads automatically the
   next time the folder is opened with the same name/ID. **Export → Load saved progress** also restores it.
6. When finished: **Export → Export JSON** and send the file back. **Export CSV** gives an Excel table with the original
   and edited question and answer side by side.

Keys: ← / → previous/next video, 1/2/3 mark best, Ctrl/Cmd+S save progress.

## Collecting the results
Put the returned JSON files in `data/annotations/` and run `python3 scripts/collect_annotations.py`.
Each exported question carries `original` (GPT) and `final` (after edits) versions, plus `selected`,
`correctness`, `edited` and the clinician's notes.

## Settings
`config.js`: `SELECT_MIN` / `SELECT_MAX` = how many best questions per video (default 1–2).

## Updating the website
Copy `index.html`, `app.js`, `styles.css` and `config.js` into the `gh-pages` checkout
(`/mnt/store/tashraf4/projects/ophbench_pages`), commit and push. The site never needs data files.
