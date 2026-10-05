// ophbench annotation UI — deployment configuration (plain script, no build step).
// Edit this file after exporting the UI; it is read by app.js at start-up.
//
//   MEDIA_BASE_URL  Prefix for every media URL ("media/<sample_id>/preview.mp4" etc.).
//                   Leave "" when ui/media/ is served next to index.html; set e.g.
//                   "https://media.example.org/ophbench" when the videos live elsewhere
//                   (the host must serve HTTPS and support HTTP Range requests).
//   SUBMIT_URL      Google Apps Script web-app URL (see ui/submit/Code.gs and README_UI.md)
//                   or the local server endpoint "api/annotations" (scripts/serve_ui.py).
//                   Leave "" to disable the Submit button (annotators export JSON instead).
//   SELECT_MIN/MAX  How many of the three questions an annotator must select before a sample
//                   can be marked complete. These are FALLBACKS: data/meta.json (written by
//                   `ophbench export-ui` from config/pipeline.yaml ui.questions_to_select_min/max)
//                   takes precedence whenever it carries select_min/select_max.
window.OPHBENCH_CONFIG = {
  MEDIA_BASE_URL: "",
  SUBMIT_URL: "",
  SELECT_MIN: 1,
  SELECT_MAX: 2
};
