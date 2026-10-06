"""Usage ledger: pricing maths, per-call recording through the client, summaries and backfill."""
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import usage  # noqa: E402
from bench.config import load_config  # noqa: E402


def _cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("OPHBENCH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OPHBENCH_LOGS_DIR", str(tmp_path / "logs"))
    return load_config()


def test_pricing_maths():
    p = usage.Pricing(input=10, cached_input=1, output=50, long_context_threshold=272000, input_long=20, output_long=75)
    std = usage.Pricing(input=10, cached_input=1, output=50)
    # 1M uncached input + 1M output = $10 + $50 (no long-context rule)
    assert std.cost(1_000_000, 0, 1_000_000) == 60.0
    # cached tokens are a subset of prompt tokens
    assert p.cost(200_000, 50_000, 0) == round((150_000 * 10 + 50_000 * 1) / 1e6, 6)
    # long-context prompts switch the whole request to the long rate
    assert p.cost(300_000, 0, 10_000) == round((300_000 * 20 + 10_000 * 75) / 1e6, 6)
    assert usage.Pricing().cost(100, 0, 100) is None and not usage.Pricing().priced


def test_client_records_every_call_with_context(tmp_path, monkeypatch):
    from bench.llm_client import GatewayClient
    cfg = _cfg(tmp_path, monkeypatch)
    monkeypatch.setenv("OPHBENCH_USER", "Dr Ledger")
    body = {"model": "gpt-6-astra", "output": [{"type": "message", "content": [{"type": "output_text", "text": "{\"ok\": true}"}]}],
            "usage": {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 200},
                      "output_tokens": 300, "output_tokens_details": {"reasoning_tokens": 120}, "total_tokens": 1300}}
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"error": {"message": "busy"}}, headers={"retry-after": "0"})
        return httpx.Response(200, json=body, headers={"x-request-id": "rid-usage"})

    client = GatewayClient(cfg, api_key="test-key-not-real", transport=httpx.MockTransport(handler))
    client.wait = lambda rs: 0
    with client.usage_context(stage="generate", sample_id="ds__v1", dataset="ds", batch=1):
        res = client.complete("hi", stream=False, schema=None)
    assert res.text
    rows = usage.read_ledger(cfg)
    assert [r["status"] for r in rows] == ["error", "ok"]          # the failed attempt is recorded too
    ok = rows[-1]
    assert ok["user"] == "Dr Ledger" and ok["stage"] == "generate" and ok["sample_id"] == "ds__v1" and ok["batch"] == 1
    assert ok["prompt_tokens"] == 1000 and ok["cached_tokens"] == 200 and ok["completion_tokens"] == 300
    assert ok["reasoning_tokens"] == 120 and ok["request_id"] == "rid-usage"
    assert ok["key_id"] and "test-key" not in json.dumps(rows)       # only a fingerprint of the key
    assert ok["est_cost_usd"] == usage.Pricing.from_cfg(cfg).cost(1000, 200, 300)
    s = usage.summarize(rows, usage.Pricing.from_cfg(cfg))
    assert s["by"]["user"]["Dr Ledger"]["calls"] == 2 and s["by"]["batch"]["1"]["videos"] == 1
    assert s["total"]["errors"] == 1 and s["total"]["cost_usd"] > 0


def test_backfill_and_report(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch)
    qa = Path(cfg.paths.qa); qa.mkdir(parents=True)
    entry = {"sample_id": "migs__1", "model": "gpt-6-astra", "route": "responses", "reasoning_effort": "xhigh",
             "status": "ok", "prompt_tokens": 20000, "completion_tokens": 12000, "reasoning_tokens": 9000,
             "latency_s": 240.0, "request_id": "rid-a", "started_at": "2026-10-05T19:00:00-04:00",
             "finished_at": "2026-10-05T19:04:00-04:00", "attempts": 1}
    (qa / "generation_log.jsonl").write_text(json.dumps(entry) + "\n")
    assert usage.backfill_from_generation_log(cfg, user="Tajamul") == 1
    assert usage.backfill_from_generation_log(cfg, user="Tajamul") == 0   # idempotent
    import argparse
    assert usage.main(argparse.Namespace(backfill=False, backfill_user=None, only_user=None, since=None), cfg) == 0
    out = Path(cfg.paths.data) / "usage"
    rep = (out / "usage_report.md").read_text()
    assert "Per person" in rep and "Tajamul" in rep and (out / "usage_calls.csv").exists()
    s = json.loads((out / "usage_summary.json").read_text())
    assert s["total"]["videos"] == 1 and s["total"]["cost_usd"] == round((20000 * 10 + 12000 * 50) / 1e6, 4)
