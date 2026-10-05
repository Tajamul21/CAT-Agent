"""Offline tests for bench.llm_client: SSE parsing, usage normalisation, request shapes, redaction, retries."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from bench import llm_client as lc  # noqa: E402
from bench.config import Config, load_config  # noqa: E402

FINAL_TEXT = '{"ok": true, "model_self_report": "gpt-6-astra"}'

# A realistic Responses-API SSE transcript (created -> deltas -> completed), as the gateway streams it.
SSE_FIXTURE = "\n".join([
    "event: response.created",
    'data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_abc","object":"response",'
    '"created_at":1759680000,"status":"in_progress","model":"gpt-6-astra","output":[],"reasoning":{"effort":"low"},"usage":null}}',
    "",
    "event: response.in_progress",
    'data: {"type":"response.in_progress","sequence_number":1,"response":{"id":"resp_abc","status":"in_progress"}}',
    "",
    "event: response.output_item.added",
    'data: {"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[]}}',
    "",
    "event: response.output_item.done",
    'data: {"type":"response.output_item.done","sequence_number":3,"output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[]}}',
    "",
    "event: response.output_item.added",
    'data: {"type":"response.output_item.added","sequence_number":4,"output_index":1,"item":{"id":"msg_1","type":"message",'
    '"status":"in_progress","role":"assistant","content":[]}}',
    "",
    "event: response.content_part.added",
    'data: {"type":"response.content_part.added","sequence_number":5,"item_id":"msg_1","output_index":1,"content_index":0,'
    '"part":{"type":"output_text","text":"","annotations":[]}}',
    "",
    "event: response.output_text.delta",
    'data: {"type":"response.output_text.delta","sequence_number":6,"item_id":"msg_1","output_index":1,"content_index":0,'
    '"delta":"{\\"ok\\": true, "}',
    "",
    "event: response.output_text.delta",
    'data: {"type":"response.output_text.delta","sequence_number":7,"item_id":"msg_1","output_index":1,"content_index":0,'
    '"delta":"\\"model_self_report\\": \\"gpt-6-astra\\"}"}',
    "",
    "event: response.output_text.done",
    'data: {"type":"response.output_text.done","sequence_number":8,"item_id":"msg_1","output_index":1,"content_index":0,'
    '"text":"{\\"ok\\": true, \\"model_self_report\\": \\"gpt-6-astra\\"}"}',
    "",
    "event: response.content_part.done",
    'data: {"type":"response.content_part.done","sequence_number":9,"item_id":"msg_1","output_index":1,"content_index":0,'
    '"part":{"type":"output_text","text":"{\\"ok\\": true, \\"model_self_report\\": \\"gpt-6-astra\\"}","annotations":[]}}',
    "",
    "event: response.output_item.done",
    'data: {"type":"response.output_item.done","sequence_number":10,"output_index":1,"item":{"id":"msg_1","type":"message",'
    '"status":"completed","role":"assistant","content":[{"type":"output_text","text":"{\\"ok\\": true, \\"model_self_report\\": \\"gpt-6-astra\\"}","annotations":[]}]}}',
    "",
    "event: response.completed",
    'data: {"type":"response.completed","sequence_number":11,"response":{"id":"resp_abc","object":"response","created_at":1759680000,'
    '"status":"completed","model":"gpt-6-astra","reasoning":{"effort":"low","summary":null},'
    '"output":[{"id":"rs_1","type":"reasoning","summary":[]},{"id":"msg_1","type":"message","status":"completed","role":"assistant",'
    '"content":[{"type":"output_text","text":"{\\"ok\\": true, \\"model_self_report\\": \\"gpt-6-astra\\"}","annotations":[]}]}],'
    '"usage":{"input_tokens":120,"input_tokens_details":{"cached_tokens":0},"output_tokens":45,'
    '"output_tokens_details":{"reasoning_tokens":32},"total_tokens":165}}}',
    "",
    "data: [DONE]",
    "",
])

REST_USAGE = {"prompt_tokens": 210, "completion_tokens": 90, "total_tokens": 300,
              "prompt_tokens_details": {"cached_tokens": 0}, "completion_tokens_details": {"reasoning_tokens": 64}}


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config()


def make_client(cfg: Config, handler=None, **llm_overrides) -> lc.GatewayClient:
    raw = json.loads(json.dumps(cfg.raw))
    raw["llm"].update(llm_overrides)
    c = Config(raw, root=cfg.root)
    transport = httpx.MockTransport(handler) if handler else None
    client = lc.GatewayClient(c, api_key="test-key-not-real", transport=transport)
    client.wait = lambda retry_state: 0.0  # no sleeping in tests
    return client


@pytest.fixture
def jpeg(tmp_path: Path) -> Path:
    p = tmp_path / "f_000_0005.0s.jpg"
    Image.new("RGB", (96, 64), color=(10, 200, 30)).save(p, "JPEG", quality=85)
    return p


# ------------------------------------------------------------------------------------------ SSE
def test_sse_parse_text_and_usage():
    col = lc.parse_sse_text(SSE_FIXTURE)
    assert col.completed is not None and col.completed_type == "response.completed"
    assert col.n_events == 12
    assert col.failed is None
    assert col.delta_text == FINAL_TEXT
    assert lc.extract_output_text(col.completed, "responses") == FINAL_TEXT
    usage = lc.normalise_usage(col.completed["usage"], "responses")
    assert usage["prompt_tokens"] == 120 and usage["completion_tokens"] == 45
    assert usage["reasoning_tokens"] == 32 and usage["total_tokens"] == 165 and usage["cached_tokens"] == 0
    assert lc.applied_effort(col.completed, "responses") == "low"


def test_sse_without_completed_keeps_deltas():
    truncated = SSE_FIXTURE.split("event: response.completed")[0]
    col = lc.parse_sse_text(truncated)
    assert col.completed is None and col.delta_text == FINAL_TEXT


def test_sse_multiline_data_and_noncompliant_lines():
    payloads = list(lc.iter_sse_payloads(['data: {"a":', 'data: 1}', "", ': comment', 'data: {"b":2}', 'data: {"c":3}', "", "data: [DONE]"]))
    assert payloads == [{"a": 1}, {"b": 2}, {"c": 3}]


# ------------------------------------------------------------------------------------------ usage / json helpers
def test_normalise_usage_rest_chat():
    u = lc.normalise_usage(REST_USAGE, "rest_chat")
    assert u == {"prompt_tokens": 210, "completion_tokens": 90, "reasoning_tokens": 64, "total_tokens": 300,
                 "cached_tokens": 0, "route": "rest_chat"}
    assert lc.normalise_usage(None, "rest_chat")["prompt_tokens"] is None
    s = lc.sum_usage(u, lc.normalise_usage({"input_tokens": 10, "output_tokens": 5}, "rest_chat"))
    assert s["prompt_tokens"] == 220 and s["completion_tokens"] == 95 and s["reasoning_tokens"] == 64


def test_extract_json_tolerates_fences_and_prose():
    assert lc.extract_json('```json\n{"a": 1}\n```')[0] == {"a": 1}
    assert lc.extract_json('Sure! {"a": {"b": [1,2]}} trailing')[0] == {"a": {"b": [1, 2]}}
    obj, err = lc.extract_json("not json at all")
    assert obj is None and err
    obj, err = lc.extract_json("[1, 2]")
    assert obj is None and "expected object" in err


def test_schema_issues_detects_missing_and_enum():
    from bench.schema import QA_OUTPUT_SCHEMA
    good = {"video_summary": "s", "generator_notes": "", "questions": [{
        "qid": "q1", "category": "temporal_grounding", "question": "q", "answer": "a", "answer_rationale": "r",
        "evidence_timestamps": [{"start_s": 1, "end_s": 2.5, "observation": "o"}], "agentic_skills": ["counting"],
        "tool_plan": ["seek"], "answer_type": "count", "options": [], "difficulty": "hard", "why_hard": "w",
        "metadata_used": [], "confidence": 0.8}]}
    assert lc.schema_issues(good, QA_OUTPUT_SCHEMA) == []
    bad = json.loads(json.dumps(good))
    del bad["generator_notes"]
    bad["questions"][0]["difficulty"] = "easy"
    bad["questions"][0]["extra"] = 1
    issues = lc.schema_issues(bad, QA_OUTPUT_SCHEMA)
    assert any("generator_notes: missing" in i for i in issues)
    assert any("difficulty" in i and "enum" in i for i in issues)
    assert any("extra: unexpected" in i for i in issues)


def test_retry_after_and_effort_ladder():
    assert lc.parse_retry_after("7") == 7.0
    assert lc.parse_retry_after(None) is None
    assert lc.parse_retry_after("garbage") is None
    assert lc.EFFORT_STEP_DOWN["max"] == "xhigh" and lc.EFFORT_STEP_DOWN["xhigh"] == "high"
    assert lc.strip_provider_prefix("openai/gpt-6-astra") == "gpt-6-astra"
    assert lc.strip_provider_prefix("gpt-6-astra") == "gpt-6-astra"


def test_extract_text_rest_chat():
    resp = {"choices": [{"message": {"role": "assistant", "content": '{"x":1}'}, "finish_reason": "stop"}], "usage": REST_USAGE}
    assert lc.extract_output_text(resp, "rest_chat") == '{"x":1}'
    assert lc.finish_info(resp, "rest_chat") is None
    resp["choices"][0]["finish_reason"] = "length"
    assert lc.finish_info(resp, "rest_chat") == "finish_reason=length"


# ------------------------------------------------------------------------------------------ request shapes
def test_build_request_responses_shape(cfg, jpeg):
    client = make_client(cfg)
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": False}
    parts = [{"type": "text", "text": "intro"}, {"type": "text", "text": "Frame 1/1 - t=00:05.0 - Incision"},
             {"type": "image", "path": str(jpeg), "caption": "Frame 1/1 - t=00:05.0 - Incision"}, {"type": "text", "text": "task"}]
    req = client.build_request(parts, system="SYS", schema=schema, schema_name="qa_output", route="responses",
                               reasoning_effort="xhigh", max_tokens=1234)
    b = req.body
    assert req.url.endswith("/codex/openai/v1/responses")
    assert b["model"] == "gpt-6-astra"  # provider prefix stripped
    assert b["stream"] is True and b["reasoning"] == {"effort": "xhigh"} and b["max_output_tokens"] == 1234
    assert b["text"] == {"format": {"type": "json_schema", "name": "qa_output", "strict": True, "schema": schema}}
    assert b["input"][0] == {"role": "system", "content": [{"type": "input_text", "text": "SYS"}]}
    user = b["input"][1]
    assert user["role"] == "user"
    assert [c["type"] for c in user["content"]] == ["input_text", "input_text", "input_image", "input_text"]
    img = user["content"][2]
    assert img["image_url"].startswith("data:image/jpeg;base64,") and img["detail"] == "high"
    assert req.n_images == 1 and req.image_bytes == jpeg.stat().st_size


def test_build_request_rest_chat_shape_and_unsupported_images(cfg, jpeg):
    client = make_client(cfg)
    schema = {"type": "object", "properties": {}, "additionalProperties": False, "required": []}
    req = client.build_request("hello", system="SYS", schema=schema, route="rest_chat", reasoning_effort="low", max_tokens=50)
    b = req.body
    assert req.url.endswith("/rest/v1/chat/completions")
    assert b["model"] == "openai/gpt-6-astra"  # prefixed on the REST route
    assert b["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hello"}]
    assert b["max_completion_tokens"] == 50 and b["reasoning_effort"] == "low"
    assert b["response_format"]["type"] == "json_schema" and b["response_format"]["json_schema"]["strict"] is True
    with pytest.raises(lc.UnsupportedInput):
        client.build_request([{"type": "image", "path": str(jpeg)}], route="rest_chat")


def test_max_effort_is_sent_as_xhigh(cfg):
    client = make_client(cfg)
    req = client.build_request("x", route="responses", reasoning_effort="max")
    assert req.body["reasoning"] == {"effort": "xhigh"}


# ------------------------------------------------------------------------------------------ redaction
def test_saved_request_has_no_base64_and_no_key(cfg, jpeg, tmp_path):
    client = make_client(cfg)
    parts = [{"type": "text", "text": "intro"}, {"type": "image", "path": str(jpeg), "caption": "Frame 1/1 - t=00:05.0 - x"}]
    req = client.build_request(parts, system="SYS", route="responses", reasoning_effort="low")
    red = req.redacted()
    dumped = json.dumps(red)
    assert "base64," not in dumped
    assert "test-key-not-real" not in dumped
    img = red["body"]["input"][1]["content"][1]
    assert img["type"] == "input_image" and img["image_url"].startswith("<image ") and img["caption"].startswith("Frame 1/1")
    assert red["n_images"] == 1 and red["route"] == "responses"
    # the real body does carry the payload (so the test is meaningful)
    assert "base64," in json.dumps(req.body)
    # bench.log.redact safety net also scrubs a raw body
    from bench.log import redact
    assert "base64," not in json.dumps(redact(req.body))


# ------------------------------------------------------------------------------------------ transport-level behaviour
def _sse_response(text: str = SSE_FIXTURE, **headers) -> httpx.Response:
    return httpx.Response(200, content=text.encode(), headers={"content-type": "text/event-stream",
                                                                "x-request-id": "rid-1", **headers})


def test_send_streaming_parses_completed(cfg):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer test-key-not-real"
        return _sse_response()

    client = make_client(cfg, handler)
    res = client.complete("hi", system="s", schema=lc.PROBE_SCHEMA_TEXT, route="responses", reasoning_effort="low")
    assert res.text == FINAL_TEXT and res.parsed == {"ok": True, "model_self_report": "gpt-6-astra"}
    assert res.usage["prompt_tokens"] == 120 and res.usage["reasoning_tokens"] == 32
    assert res.request_id == "rid-1" and res.route == "responses" and res.effort_applied == "low"
    assert res.attempts == 1 and res.repaired is False and res.schema_issues == []
    assert seen[0]["stream"] is True
    assert "base64" not in json.dumps(res.request_redacted)
    assert res.est_cost_usd is None  # no prices configured


def test_send_retries_on_429_then_succeeds(cfg):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={"error": {"message": "rate limited"}}, headers={"retry-after": "0"})
        return _sse_response()

    client = make_client(cfg, handler, max_retries=5)
    res = client.complete("hi", route="responses", reasoning_effort="low")
    assert calls["n"] == 3 and res.attempts == 3 and res.text == FINAL_TEXT


def test_send_gives_up_after_max_retries(cfg):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream down")

    client = make_client(cfg, handler, max_retries=2)
    with pytest.raises(lc.RetryableError):
        client.complete("hi", route="responses")


def test_effort_step_down_on_400(cfg):
    efforts = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        efforts.append(body["reasoning"]["effort"])
        if body["reasoning"]["effort"] == "xhigh":
            return httpx.Response(400, json={"error": {"message": "Invalid value: 'xhigh'. Supported values are: low, medium, high"}})
        return _sse_response()

    client = make_client(cfg, handler)
    res = client.complete("hi", route="responses", reasoning_effort="xhigh")
    assert efforts == ["xhigh", "high"] and res.effort_requested == "high" and res.attempts == 2
    # the step-down is remembered for later requests in this session
    req2 = client.build_request("again", route="responses", reasoning_effort="xhigh")
    assert req2.body["reasoning"]["effort"] == "high"


def test_non_retryable_400_raises(cfg):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "Unknown parameter: 'foo'"}})

    client = make_client(cfg, handler)
    with pytest.raises(lc.GatewayError) as ei:
        client.complete("hi", route="responses")
    assert ei.value.status == 400 and "Unknown parameter" in str(ei.value)


def test_stream_without_completed_falls_back_to_non_stream(cfg):
    bodies = []
    truncated = "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{}}\n\n"
    final = {"id": "resp_x", "status": "completed", "model": "gpt-6-astra", "reasoning": {"effort": "low"},
             "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": FINAL_TEXT}]}],
             "usage": {"input_tokens": 5, "output_tokens": 7, "output_tokens_details": {"reasoning_tokens": 0}}}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if body["stream"]:
            return _sse_response(truncated)
        return httpx.Response(200, json=final)

    client = make_client(cfg, handler)
    res = client.complete("hi", route="responses", reasoning_effort="low")
    assert [b["stream"] for b in bodies] == [True, False]
    assert res.text == FINAL_TEXT and res.usage["completion_tokens"] == 7 and res.attempts == 2


def test_json_repair_uses_rest_chat_once(cfg):
    urls = []
    broken_sse = SSE_FIXTURE.replace(
        '"content":[{"type":"output_text","text":"{\\"ok\\": true, \\"model_self_report\\": \\"gpt-6-astra\\"}","annotations":[]}]}],',
        '"content":[{"type":"output_text","text":"{\\"ok\\": true, \\"model_self_report\\": ","annotations":[]}]}],')
    assert "model_self_report\\\": \"" in broken_sse

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(request.url.path)
        if request.url.path.endswith("/responses"):
            return _sse_response(broken_sse)
        body = json.loads(request.content)
        assert body["model"] == "openai/gpt-6-astra" and body["reasoning_effort"] == "low"
        assert "Return only valid JSON matching this schema" in body["messages"][0]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": FINAL_TEXT}, "finish_reason": "stop"}],
                                         "usage": REST_USAGE})

    client = make_client(cfg, handler)
    res = client.complete("hi", schema=lc.PROBE_SCHEMA_TEXT, route="responses", reasoning_effort="low")
    assert [u.split("/")[-1] for u in urls] == ["responses", "completions"]
    assert res.parsed == {"ok": True, "model_self_report": "gpt-6-astra"} and res.repaired is True
    assert res.attempts == 2
    assert res.usage["prompt_tokens"] == 120 + 210 and res.usage["reasoning_tokens"] == 32 + 64


def test_estimate_cost_with_prices(cfg):
    client = make_client(cfg, price_per_m_input_usd=2.0, price_per_m_output_usd=10.0)
    assert client.estimate_cost({"prompt_tokens": 1_000_000, "completion_tokens": 100_000}) == pytest.approx(3.0)
    assert client.estimate_cost(None) is None


def test_list_models_shapes(cfg):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/api/models")
        return httpx.Response(200, json={"data": [{"id": "openai/gpt-6-astra"}, {"id": "anthropic/claude-x"}]})

    client = make_client(cfg, handler)
    assert client.list_models() == ["openai/gpt-6-astra", "anthropic/claude-x"]
    # the real WSE gateway shape (verified 2026-10-05): camelCase modelId/modelName inside "data"
    real = {"object": "list", "data": [{"modelId": "openai/gpt-6-astra", "providerId": "openai", "modelName": "gpt-6-astra",
                                        "codexCompatible": True}], "info": {"scope": "key", "count": 1}}
    assert lc._model_ids(real) == ["openai/gpt-6-astra"]
    assert lc._model_ids(["a", "b"]) == ["a", "b"]
    assert lc._model_ids({"models": [{"name": "m1"}]}) == ["m1"]


def test_is_effort_error_matches_gateway_wording():
    # the real gateway message for a rejected effort does not contain the word "effort"
    assert lc.is_effort_error("Invalid value: 'max'. Supported values are: low, medium, high, xhigh", "max")
    assert lc.is_effort_error("reasoning_effort must be one of low|medium|high")
    assert lc.is_effort_error("Unsupported reasoning.effort value", None)
    assert not lc.is_effort_error("Unknown parameter: 'image'", "xhigh")
    assert not lc.is_effort_error("Invalid value: 'video/mp4'.", "xhigh")


# ------------------------------------------------------------------------------------------ generate stage (mocked gateway)
def _qa_payload() -> dict:
    def q(i, cat, atype, opts=None):
        return {"qid": f"q{i}", "category": cat, "question": f"Question {i} about 00:10-00:20?", "answer": f"Answer {i}",
                "answer_rationale": "Because of what is visible at 00:12.", "evidence_timestamps": [{"start_s": 10, "end_s": 20, "observation": "obs"}],
                "agentic_skills": ["temporal_localization", "verification"], "tool_plan": ["seek to 00:10", "zoom"],
                "answer_type": atype, "options": opts or [], "difficulty": "hard", "why_hard": "needs several steps",
                "metadata_used": ["segments.phase"], "confidence": 0.8}
    return {"video_summary": "A short synthetic video.", "generator_notes": "none",
            "questions": [q(1, "temporal_grounding", "timestamp"), q(2, "complication_detection_management", "boolean"),
                          q(3, "instrument_anatomy_reasoning", "multiple_choice", ["A. x", "B. y", "C. z", "D. w"])]}


def _sse_for(text: str, usage: dict, effort: str = "xhigh") -> str:
    resp = {"id": "resp_gen", "status": "completed", "model": "gpt-6-astra", "reasoning": {"effort": effort},
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}], "usage": usage}
    return "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{}}\n\nevent: response.completed\ndata: " \
        + json.dumps({"type": "response.completed", "response": resp}) + "\n\ndata: [DONE]\n\n"


@pytest.fixture
def smoke_data(tmp_path: Path, monkeypatch) -> tuple[Config, str]:
    """A data dir with a manifest and one prepared sample (3 PIL frames)."""
    from bench.schema import FrameInfo, PreparedSample, ProbeInfo, SampleRecord, Segment
    from bench.util import write_json, write_jsonl

    data = tmp_path / "data"
    monkeypatch.setenv("OPHBENCH_DATA_DIR", str(data))
    monkeypatch.setenv("OPHBENCH_LOGS_DIR", str(tmp_path / "logs"))
    c = load_config()
    sid = "cataract101__case_smoke"
    rec = SampleRecord(sample_id=sid, dataset="cataract101", video_id="case_smoke", duration_s=30.0, fps=25.0, width=720, height=540,
                       procedure="Cataract surgery (phacoemulsification)", procedure_category="cataract",
                       segments=[Segment(label="Incision", label_id=1, start_s=0, end_s=15), Segment(label="Capsulorhexis", label_id=2, start_s=15, end_s=30)],
                       labels={"surgeon_id": 1, "experience": "low"}, stratum_key="s", selection_reason="r", sample_index=0)
    write_jsonl(data / "sample" / "sample_manifest.jsonl", [rec.to_dict()])
    # a second manifest record that is NOT prepared (must be skipped with a warning)
    rec2 = SampleRecord.from_dict({**rec.to_dict(), "sample_id": "cataract101__case_unprepared", "video_id": "case_unprepared"})
    write_jsonl(data / "sample" / "sample_manifest.jsonl", [rec.to_dict(), rec2.to_dict()])
    sdir = data / "prepared" / sid
    (sdir / "frames").mkdir(parents=True)
    frames = []
    for i, t in enumerate([5.0, 15.0, 25.0]):
        name = f"frames/f_{i:03d}_{t:07.1f}s.jpg"
        Image.new("RGB", (32, 24), color=(i * 60, 10, 10)).save(sdir / name, "JPEG")
        frames.append(FrameInfo(idx=i, t_s=t, file=name, label_at_t=None))
    ps = PreparedSample(record=rec, probe=ProbeInfo(duration_s=30.0, fps=25.0, width=720, height=540), frames=frames, timeline_note="original timeline")
    write_json(sdir / "sample.json", ps.to_dict())
    return c, sid


def test_generate_stage_end_to_end_with_mocked_gateway(smoke_data, monkeypatch):
    import argparse

    from bench import generate
    from bench.schema import QASet
    from bench.util import read_json, read_jsonl

    c, sid = smoke_data
    calls = {"n": 0}
    usage = {"input_tokens": 5000, "output_tokens": 900, "output_tokens_details": {"reasoning_tokens": 600}}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        body = json.loads(request.content)
        assert body["model"] == "gpt-6-astra" and body["reasoning"] == {"effort": "xhigh"} and body["stream"] is True
        assert body["text"]["format"]["name"] == "qa_output" and body["text"]["format"]["strict"] is True
        user = body["input"][1]["content"]
        assert sum(1 for p in user if p["type"] == "input_image") == 3
        assert user[0]["type"] == "input_text" and "Cataract-101" in user[0]["text"]
        return httpx.Response(200, content=_sse_for(json.dumps(_qa_payload()), usage).encode(),
                              headers={"content-type": "text/event-stream", "cf-aig-request-id": "cf-123"})

    monkeypatch.setattr(generate, "GatewayClient", lambda cfg, log=None: make_client(cfg, handler))

    def run(**kw) -> int:
        ns = argparse.Namespace(concurrency=2, limit=None, ids=None, datasets=None, dry_run=False, force=False,
                                route=None, effort=None, max_tokens=None)
        for k, v in kw.items():
            setattr(ns, k, v)
        return generate.main(ns, c)

    # dry run: no API calls, prompt written
    assert run(dry_run=True) == 0
    dry = read_json(Path(c.paths.qa) / "dryrun" / f"{sid}.json")
    assert dry["n_frames"] == 3 and "Frame 1/3 - t=00:05.0 - Incision" in dry["user_text"] and "base64" not in dry["user_text"]
    assert calls["n"] == 0

    # real (mocked) run
    assert run() == 0
    assert calls["n"] == 1
    qa = QASet.from_dict(read_json(Path(c.paths.qa) / f"{sid}.json"))
    assert len(qa.questions) == 3 and qa.questions[2].answer_type == "multiple_choice" and len(qa.questions[2].options) == 4
    prov = qa.provenance
    assert prov["model"] == "gpt-6-astra" and prov["route"] == "responses" and prov["effort"] == "xhigh"
    assert prov["n_frames"] == 3 and prov["usage"]["prompt_tokens"] == 5000 and prov["usage"]["reasoning_tokens"] == 600
    assert prov["prompt_version"] == "v1" and prov["attempts"] == 1 and prov["request_id"] == "cf-123" and prov["generated_at"]
    req = read_json(Path(c.paths.qa) / "raw" / f"{sid}.request.json")
    assert "base64" not in json.dumps(req) and req["n_images"] == 3
    resp = read_json(Path(c.paths.qa) / "raw" / f"{sid}.response.json")
    assert resp["usage"]["completion_tokens"] == 900 and resp["request_id"] == "cf-123"
    log_rows = read_jsonl(Path(c.paths.qa) / "generation_log.jsonl")
    assert [r["status"] for r in log_rows] == ["ok"] and log_rows[0]["reasoning_tokens"] == 600 and log_rows[0]["n_frames"] == 3
    assert [r["sample_id"] for r in read_jsonl(Path(c.paths.qa) / "qa_all.jsonl")] == [sid]
    runs = read_jsonl(Path(c.paths.logs) / "runs.jsonl")
    gen_runs = [r for r in runs if r["stage"] == "generate"]
    assert gen_runs[-1]["summary"]["counts"] == {"manifest": 2, "filtered": 2, "unprepared": 1, "skipped_existing": 0,
                                                 "selected": 1, "ok": 1, "parse_error": 0, "error": 0, "dry_run": 0}
    assert gen_runs[-1]["summary"]["tokens"] == {"prompt": 5000, "completion": 900, "reasoning": 600}

    # resumable: nothing to do on a re-run; --force regenerates
    assert run() == 0 and calls["n"] == 1
    assert run(force=True) == 0 and calls["n"] == 2
    assert len(read_jsonl(Path(c.paths.qa) / "generation_log.jsonl")) == 2


def test_generate_parse_error_leaves_no_qa_file(smoke_data, monkeypatch):
    import argparse

    from bench import generate
    from bench.util import read_jsonl

    c, sid = smoke_data

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/responses"):
            assert json.loads(request.content)["reasoning"] == {"effort": "high"}
            return httpx.Response(200, content=_sse_for("this is not json", {"input_tokens": 1, "output_tokens": 1}, effort="high").encode(),
                                  headers={"content-type": "text/event-stream"})
        # the repair call (rest_chat) also fails to produce JSON
        return httpx.Response(200, json={"choices": [{"message": {"content": "still not json"}}], "usage": {}})

    monkeypatch.setattr(generate, "GatewayClient", lambda cfg, log=None: make_client(cfg, handler))
    ns = argparse.Namespace(concurrency=1, limit=None, ids=[sid], datasets=None, dry_run=False, force=False, route=None,
                            effort="high", max_tokens=1000)
    assert generate.main(ns, c) == 1  # every attempted sample failed -> non-zero
    assert not (Path(c.paths.qa) / f"{sid}.json").exists()
    rows = read_jsonl(Path(c.paths.qa) / "generation_log.jsonl")
    assert rows[-1]["status"] == "parse_error" and rows[-1]["attempts"] == 2 and rows[-1]["reasoning_effort"] == "high"
    assert (Path(c.paths.qa) / "raw" / f"{sid}.response.json").exists()
