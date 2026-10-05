/**
 * ophbench — Google Apps Script web-app endpoint that receives annotation records from the UI
 * (Submit button) and appends them to a Google Sheet.
 *
 * Setup (details in ui/README_UI.md):
 *   1. Create a Google Sheet, open Extensions > Apps Script, replace Code.gs with this file.
 *   2. Deploy > New deployment > type "Web app"; Execute as: Me; Who has access: Anyone. Authorise.
 *   3. Paste the web-app URL (…/exec) into ui/config.js as SUBMIT_URL. After every code change,
 *      create a NEW deployment version (Deploy > Manage deployments > Edit > Version: New).
 *
 * Sheets created on demand:
 *   questions   one row per (annotator, sample, question)
 *   samples     one row per (annotator, sample)
 *   submissions one row per received payload (audit trail)
 * Every submission appends rows again; scripts/collect_annotations.py de-duplicates by
 * (annotator, sample_id[, qid]) keeping the latest updated_at / received_at.
 *
 * The UI sends:  fetch(SUBMIT_URL, {method: "POST", body: JSON.stringify(record),
 *                 headers: {"Content-Type": "text/plain;charset=utf-8"}, redirect: "follow"})
 * where record = {ui_version, annotator, exported_at, samples: {sample_id: {...}}}.
 */

// Leave empty when this script is container-bound to the spreadsheet (created via Extensions > Apps Script).
// Otherwise paste the spreadsheet id from its URL (…/spreadsheets/d/<ID>/edit).
var SPREADSHEET_ID = "";

var QUESTIONS_SHEET = "questions";
var SAMPLES_SHEET = "samples";
var SUBMISSIONS_SHEET = "submissions";

var Q_HEADERS = ["annotator", "sample_id", "dataset", "qid", "selected", "correctness", "relevance", "difficulty",
                 "agentic", "clarity", "edited", "question_edit", "answer_edit", "comment", "updated_at", "received_at"];
var S_HEADERS = ["annotator", "sample_id", "status", "flags", "comment", "updated_at", "received_at"];
var SUB_HEADERS = ["received_at", "annotator", "exported_at", "ui_version", "n_samples", "n_questions", "bytes"];

var MAX_CELL_CHARS = 49000; // Google Sheets cell limit is 50,000 characters

/** Health check: GET <web-app-url> -> {"ok":true,...} */
function doGet(e) {
  return jsonOut_({
    ok: true,
    service: "ophbench-submit",
    version: 1,
    time: new Date().toISOString(),
    sheets: [QUESTIONS_SHEET, SAMPLES_SHEET, SUBMISSIONS_SHEET]
  });
}

/** Receive one annotation record (JSON body) and append rows. Returns {"ok":true,"rows":n,...}. */
function doPost(e) {
  var lock = LockService.getScriptLock();
  var locked = false;
  try {
    locked = lock.tryLock(30000);
    if (!locked) return jsonOut_({ ok: false, error: "server busy, please retry" });

    var body = (e && e.postData && e.postData.contents) ? e.postData.contents : "";
    if (!body) return jsonOut_({ ok: false, error: "empty request body" });

    var payload;
    try {
      payload = JSON.parse(body);
    } catch (err) {
      return jsonOut_({ ok: false, error: "body is not valid JSON: " + err });
    }
    if (!payload || typeof payload !== "object" || !payload.samples || typeof payload.samples !== "object") {
      return jsonOut_({ ok: false, error: "payload must be an object with a 'samples' object" });
    }

    var receivedAt = new Date().toISOString();
    var annotator = str_(payload.annotator) || "anonymous";
    var qRows = [];
    var sRows = [];
    var sampleIds = Object.keys(payload.samples);
    for (var i = 0; i < sampleIds.length; i++) {
      var sid = sampleIds[i];
      var s = payload.samples[sid] || {};
      var dataset = sid.indexOf("__") > 0 ? sid.split("__")[0] : "";
      var flags = Array.isArray(s.flags) ? s.flags.join(";") : "";
      var updatedAt = str_(s.updated_at);
      sRows.push([annotator, sid, str_(s.status), flags, clip_(s.comment), updatedAt, receivedAt]);
      var qs = (s.questions && typeof s.questions === "object") ? s.questions : {};
      var qids = Object.keys(qs);
      for (var j = 0; j < qids.length; j++) {
        var q = qs[qids[j]] || {};
        qRows.push([
          annotator, sid, dataset, qids[j],
          q.selected === true,
          str_(q.correctness),
          num_(q.relevance), num_(q.difficulty), num_(q.agentic), num_(q.clarity),
          q.edited === true,
          clip_(q.question_edit), clip_(q.answer_edit), clip_(q.comment),
          updatedAt, receivedAt
        ]);
      }
    }

    var ss = openSpreadsheet_();
    appendRows_(getSheet_(ss, QUESTIONS_SHEET, Q_HEADERS), qRows, Q_HEADERS.length);
    appendRows_(getSheet_(ss, SAMPLES_SHEET, S_HEADERS), sRows, S_HEADERS.length);
    appendRows_(getSheet_(ss, SUBMISSIONS_SHEET, SUB_HEADERS),
                [[receivedAt, annotator, str_(payload.exported_at), str_(payload.ui_version), sRows.length, qRows.length, body.length]],
                SUB_HEADERS.length);

    return jsonOut_({ ok: true, rows: qRows.length + sRows.length, questions: qRows.length, samples: sRows.length,
                      annotator: annotator, received_at: receivedAt });
  } catch (err) {
    return jsonOut_({ ok: false, error: String(err && err.message ? err.message : err) });
  } finally {
    if (locked) {
      try { lock.releaseLock(); } catch (e2) { /* ignore */ }
    }
  }
}

// ------------------------------------------------------------------------------ helpers
function openSpreadsheet_() {
  var ss = SPREADSHEET_ID ? SpreadsheetApp.openById(SPREADSHEET_ID) : SpreadsheetApp.getActiveSpreadsheet();
  if (!ss) {
    throw new Error("No spreadsheet: bind this script to a Sheet (Extensions > Apps Script) or set SPREADSHEET_ID");
  }
  return ss;
}

function getSheet_(ss, name, headers) {
  var sheet = ss.getSheetByName(name);
  if (!sheet) sheet = ss.insertSheet(name);
  if (sheet.getLastRow() === 0) {
    sheet.getRange(1, 1, 1, headers.length).setValues([headers]).setFontWeight("bold");
    sheet.setFrozenRows(1);
  }
  return sheet;
}

function appendRows_(sheet, rows, width) {
  if (!rows || !rows.length) return;
  var start = sheet.getLastRow() + 1;
  sheet.getRange(start, 1, rows.length, width).setValues(rows);
}

function jsonOut_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

function str_(v) {
  return (v === null || v === undefined) ? "" : String(v);
}

function num_(v) {
  if (v === null || v === undefined || v === "") return "";
  var n = Number(v);
  return isNaN(n) ? "" : n;
}

function clip_(v) {
  var s = str_(v);
  return s.length > MAX_CELL_CHARS ? s.slice(0, MAX_CELL_CHARS) + "…[truncated]" : s;
}

/** Run this once from the editor to check permissions and see rows appear (Run > testDoPost). */
function testDoPost() {
  var fake = {
    postData: {
      contents: JSON.stringify({
        ui_version: 1, annotator: "test_user", exported_at: new Date().toISOString(),
        samples: {
          "cataract101__case_269": {
            status: "done", updated_at: new Date().toISOString(), flags: ["video_quality"], comment: "test",
            questions: {
              q1: { selected: true, correctness: "correct", relevance: 5, difficulty: 4, agentic: 4, clarity: 5,
                    question_edit: "", answer_edit: "", edited: false, comment: "" },
              q2: { selected: false, correctness: "partial", relevance: 3, difficulty: 3, agentic: 2, clarity: 4,
                    question_edit: "Edited question", answer_edit: "", edited: true, comment: "fixed wording" }
            }
          }
        }
      })
    }
  };
  var out = doPost(fake);
  Logger.log(out.getContent());
}
