"""Gateway LLM client for the JHU WSE AI Gateway (DESIGN.md §0 and §7).

Two verified routes:

* ``responses`` - ``POST {base}/codex/openai/v1/responses`` (OpenAI Responses shape, SSE streaming by
  default).  This is the **only** route that accepts images, as
  ``{"type": "input_image", "image_url": "data:image/jpeg;base64,...", "detail": "high"}``.
* ``rest_chat`` - ``POST {base}/rest/v1/chat/completions`` (OpenAI chat-completions shape, text only).

Design points
-------------
* ``GatewayClient(cfg)`` takes the key from ``cfg.api_key()`` and base_url/model/route/effort/timeout/
  retries from ``cfg.section("llm")``; every call may override route/effort/max_tokens/stream.
* ``build_request()`` turns ``(system_text, user_parts)`` into a route specific body; ``send()`` performs
  the HTTP call with tenacity retries (timeouts, connection errors, 408/429/5xx with ``Retry-After``),
  a one-time reasoning-effort step-down on HTTP 400 (``max -> xhigh -> high``), a ``stream:false``
  fallback when the SSE stream carries no ``response.completed`` event, and one JSON repair call.
* The API key is never logged and every persisted request has base64 payloads replaced
  (``GatewayRequest.redacted()`` plus ``bench.log.redact`` as a safety net).
* ``probe`` (``./ophbench probe``) lives here as ``add_args``/``main``.
"""
from __future__ import annotations

import argparse
import contextlib
import base64
import copy
import email.utils
import json
import logging
import mimetypes
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import httpx
from tenacity import RetryCallState, Retrying, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_base, wait_exponential_jitter

from bench.config import Config
from bench.log import get_logger, now_iso, record_run, redact
from bench.util import run_cmd

ROUTES: tuple[str, ...] = ("responses", "rest_chat")
EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh")
#: one step down the ladder when the gateway rejects the requested effort
EFFORT_STEP_DOWN: dict[str, str] = {"max": "xhigh", "xhigh": "high", "high": "medium", "medium": "low"}

RESPONSES_PATH = "/codex/openai/v1/responses"
REST_CHAT_PATH = "/rest/v1/chat/completions"
MODELS_PATH = "/api/models"

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 522, 524}
_EFFORT_ERR_RE = re.compile(r"reasoning[._ ]?effort|\beffort\b", re.I)
#: the gateway's message for a rejected effort value does not always say "effort": it lists the enum
_EFFORT_ENUM_RE = re.compile(r"(supported|allowed|valid|expected|must be one of)[^.]*\blow\b[^.]*\bhigh\b", re.I | re.S)
_LOG = logging.getLogger("ophbench.llm")


# ------------------------------------------------------------------------------------------ errors
class GatewayError(RuntimeError):
    """Non-retryable gateway failure (4xx other than 408/429, malformed response, ...)."""

    def __init__(self, message: str, *, status: Optional[int] = None, body: Optional[str] = None,
                 request_id: Optional[str] = None):
        super().__init__(message)
        self.status = status
        self.body = body
        self.request_id = request_id


class RetryableError(GatewayError):
    """Timeouts, connection errors, HTTP 408/429/5xx - retried with backoff."""

    def __init__(self, message: str, *, status: Optional[int] = None, body: Optional[str] = None,
                 request_id: Optional[str] = None, retry_after: Optional[float] = None):
        super().__init__(message, status=status, body=body, request_id=request_id)
        self.retry_after = retry_after


class EffortRejected(GatewayError):
    """HTTP 400 complaining about the reasoning effort value."""


class NoCompletedEvent(GatewayError):
    """The SSE stream ended without a ``response.completed`` event."""


class UnsupportedInput(ValueError):
    """Images were passed to the text-only ``rest_chat`` route."""


# ------------------------------------------------------------------------------------------ helpers
def strip_provider_prefix(model: str) -> str:
    """``openai/gpt-6-astra`` -> ``gpt-6-astra`` (the Responses route wants the bare name)."""
    return model.split("/", 1)[1] if "/" in model else model


def normalise_effort(effort: Optional[str]) -> Optional[str]:
    """Lower-case; ``max`` is kept (the gateway rejects it; ``send`` steps it down to ``xhigh``)."""
    if effort is None:
        return None
    e = str(effort).strip().lower()
    return e or None


def is_effort_error(message: str, effort: Optional[str] = None) -> bool:
    """True when an HTTP 400 message complains about the reasoning effort value.

    Matches "reasoning_effort"/"effort", the enum listing ("Supported values are: low, medium, high, xhigh")
    and "Invalid value: '<effort>'" for the effort that was sent.
    """
    m = message or ""
    if _EFFORT_ERR_RE.search(m) or _EFFORT_ENUM_RE.search(m):
        return True
    if effort and re.search(r"invalid|unsupported|not supported|must be one of", m, re.I) and f"'{effort}'" in m:
        return True
    return False


def normalise_usage(usage: Optional[dict], route: str = "responses") -> dict:
    """Map either route's usage block to prompt/completion/reasoning/total/cached token keys."""
    u = usage if isinstance(usage, dict) else {}
    prompt = u.get("prompt_tokens", u.get("input_tokens"))
    completion = u.get("completion_tokens", u.get("output_tokens"))
    out_details = u.get("completion_tokens_details") or u.get("output_tokens_details") or {}
    in_details = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
    reasoning = out_details.get("reasoning_tokens") if isinstance(out_details, dict) else None
    cached = in_details.get("cached_tokens") if isinstance(in_details, dict) else None
    total = u.get("total_tokens")
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return {
        "prompt_tokens": _as_int(prompt),
        "completion_tokens": _as_int(completion),
        "reasoning_tokens": _as_int(reasoning),
        "total_tokens": _as_int(total),
        "cached_tokens": _as_int(cached),
        "route": route,
    }


def _as_int(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def sum_usage(a: dict, b: dict) -> dict:
    """Element-wise sum of two normalised usage dicts (None-aware)."""
    out = dict(a)
    for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens", "cached_tokens"):
        if a.get(k) is None and b.get(k) is None:
            out[k] = None
        else:
            out[k] = (a.get(k) or 0) + (b.get(k) or 0)
    return out


def extract_output_text(resp: dict, route: str) -> str:
    """Concatenate the model's text output for either route."""
    if route == "rest_chat":
        choices = resp.get("choices") or []
        if not choices:
            return ""
        msg = (choices[0] or {}).get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
        return ""
    parts: list[str] = []
    for item in resp.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for c in item.get("content") or []:
            if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                parts.append(str(c.get("text", "")))
    if not parts and isinstance(resp.get("output_text"), str):
        return resp["output_text"]
    return "".join(parts)


def extract_refusal(resp: dict, route: str) -> Optional[str]:
    if route == "rest_chat":
        choices = resp.get("choices") or []
        msg = (choices[0] or {}).get("message") if choices else {}
        return (msg or {}).get("refusal") or None
    for item in resp.get("output") or []:
        if isinstance(item, dict) and item.get("type") == "message":
            for c in item.get("content") or []:
                if isinstance(c, dict) and c.get("type") == "refusal":
                    return str(c.get("refusal") or "refusal")
    return None


def applied_effort(resp: dict, route: str) -> Optional[str]:
    """The effort echoed by the server (Responses: ``response.reasoning.effort``)."""
    if route == "rest_chat":
        return resp.get("reasoning_effort")
    r = resp.get("reasoning")
    if isinstance(r, dict):
        return r.get("effort")
    return None


def finish_info(resp: dict, route: str) -> Optional[str]:
    """Return a note when the output was cut short (``length`` / ``incomplete``)."""
    if route == "rest_chat":
        choices = resp.get("choices") or []
        fr = (choices[0] or {}).get("finish_reason") if choices else None
        return f"finish_reason={fr}" if fr and fr not in ("stop", None) else None
    if resp.get("status") == "incomplete" or resp.get("incomplete_details"):
        return f"incomplete: {json.dumps(resp.get('incomplete_details'))}"
    return None


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> tuple[Optional[dict], Optional[str]]:
    """Parse the model text as a JSON object; tolerate code fences and leading/trailing prose."""
    if text is None:
        return None, "empty response"
    s = text.strip()
    if not s:
        return None, "empty response"
    candidates = [s]
    m = _FENCE_RE.search(s)
    if m:
        candidates.append(m.group(1).strip())
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j > i:
        candidates.append(s[i:j + 1])
    last_err = "no JSON object found"
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError as e:
            last_err = f"JSONDecodeError: {e}"
            continue
        if isinstance(obj, dict):
            return obj, None
        last_err = f"JSON is {type(obj).__name__}, expected object"
    return None, last_err


def schema_issues(obj: Any, schema: dict, path: str = "$", limit: int = 12) -> list[str]:
    """Minimal JSON-schema checker (type/required/enum/additionalProperties) - enough to decide on repair."""
    issues: list[str] = []
    _check(obj, schema, path, issues, limit)
    return issues[:limit]


def _check(obj: Any, schema: dict, path: str, issues: list[str], limit: int) -> None:
    if len(issues) >= limit or not isinstance(schema, dict):
        return
    t = schema.get("type")
    if t == "object":
        if not isinstance(obj, dict):
            issues.append(f"{path}: expected object, got {type(obj).__name__}")
            return
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in obj:
                issues.append(f"{path}.{k}: missing required property")
        if schema.get("additionalProperties") is False:
            for k in obj:
                if k not in props:
                    issues.append(f"{path}.{k}: unexpected property")
        for k, sub in props.items():
            if k in obj:
                _check(obj[k], sub, f"{path}.{k}", issues, limit)
    elif t == "array":
        if not isinstance(obj, list):
            issues.append(f"{path}: expected array, got {type(obj).__name__}")
            return
        for i, item in enumerate(obj):
            _check(item, schema.get("items", {}), f"{path}[{i}]", issues, limit)
    elif t == "string":
        if not isinstance(obj, str):
            issues.append(f"{path}: expected string, got {type(obj).__name__}")
        elif "enum" in schema and obj not in schema["enum"]:
            issues.append(f"{path}: {obj!r} not in enum")
    elif t == "number":
        if isinstance(obj, bool) or not isinstance(obj, (int, float)):
            issues.append(f"{path}: expected number, got {type(obj).__name__}")
    elif t == "integer":
        if isinstance(obj, bool) or not isinstance(obj, int):
            issues.append(f"{path}: expected integer, got {type(obj).__name__}")
    elif t == "boolean":
        if not isinstance(obj, bool):
            issues.append(f"{path}: expected boolean, got {type(obj).__name__}")


def parse_retry_after(value: Optional[str]) -> Optional[float]:
    """``Retry-After`` header: seconds or an HTTP date -> seconds from now."""
    if not value:
        return None
    v = value.strip()
    try:
        return max(0.0, float(v))
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(v)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    return max(0.0, dt.timestamp() - time.time())


def _error_message(body_text: str) -> str:
    """Pull ``error.message`` (or similar) out of a JSON error body; fall back to the raw text."""
    try:
        data = json.loads(body_text)
    except (TypeError, ValueError):
        return body_text.strip()[:600]
    if isinstance(data, dict):
        err = data.get("error", data)
        if isinstance(err, dict):
            msg = err.get("message") or err.get("detail") or err.get("error")
            if msg:
                code = err.get("code") or err.get("type")
                return f"{msg} ({code})" if code else str(msg)
        if isinstance(err, str):
            return err
        for k in ("message", "detail", "msg"):
            if isinstance(data.get(k), str):
                return data[k]
    return body_text.strip()[:600]


# ------------------------------------------------------------------------------------------ SSE
@dataclass
class SSECollected:
    """Result of consuming a Responses SSE stream."""

    n_events: int = 0
    completed: Optional[dict] = None  # the ``response`` object of response.completed/incomplete
    completed_type: Optional[str] = None
    deltas: list[str] = field(default_factory=list)
    failed: Optional[dict] = None
    event_types: list[str] = field(default_factory=list)

    @property
    def delta_text(self) -> str:
        return "".join(self.deltas)


def iter_sse_payloads(lines: Iterable[str]) -> Iterable[Any]:
    """Yield the JSON payload of each SSE event (``data:`` lines; blank line terminates an event)."""
    buf: list[str] = []

    def flush() -> list[Any]:
        if not buf:
            return []
        joined = "\n".join(buf)
        buf.clear()
        if joined.strip() == "[DONE]":
            return []
        try:
            return [json.loads(joined)]
        except json.JSONDecodeError:
            out = []
            for piece in joined.split("\n"):  # non-compliant servers: one JSON per data line
                piece = piece.strip()
                if not piece or piece == "[DONE]":
                    continue
                try:
                    out.append(json.loads(piece))
                except json.JSONDecodeError:
                    _LOG.debug("skipping unparsable SSE payload: %.120s", piece)
            return out

    for raw in lines:
        line = raw.rstrip("\r")
        if line == "":
            yield from flush()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            buf.append(line[5:].lstrip(" "))
            continue
        # ``event:``, ``id:``, ``retry:`` fields are ignored - the payload carries ``type``.
    yield from flush()


def collect_sse(lines: Iterable[str]) -> SSECollected:
    """Consume Responses SSE events; keep the ``response.completed`` payload and the text deltas."""
    col = SSECollected()
    for payload in iter_sse_payloads(lines):
        if not isinstance(payload, dict):
            continue
        col.n_events += 1
        typ = str(payload.get("type") or "")
        col.event_types.append(typ)
        if typ in ("response.completed", "response.incomplete", "response.done"):
            resp = payload.get("response")
            if isinstance(resp, dict):
                col.completed, col.completed_type = resp, typ
        elif typ in ("response.failed", "error"):
            col.failed = payload
        elif typ == "response.output_text.delta":
            col.deltas.append(str(payload.get("delta", "")))
        elif not typ and isinstance(payload.get("output"), list):
            col.completed, col.completed_type = payload, "response"  # bare response object
    return col


def parse_sse_text(text: str) -> SSECollected:
    """Convenience for tests / recorded transcripts."""
    return collect_sse(text.splitlines())


# ------------------------------------------------------------------------------------------ request
@dataclass
class GatewayRequest:
    """A fully built request. ``body`` holds base64 images; ``body_redacted`` never does."""

    route: str
    url: str
    body: dict
    body_redacted: dict
    reasoning_effort: Optional[str]
    max_tokens: Optional[int]
    schema: Optional[dict]
    schema_name: str
    stream: bool
    n_images: int = 0
    image_bytes: int = 0
    model: str = ""

    def redacted(self) -> dict:
        """Loggable form of the request (no key, no base64)."""
        return redact({
            "route": self.route,
            "url": self.url,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
            "n_images": self.n_images,
            "image_bytes": self.image_bytes,
            "schema_name": self.schema_name if self.schema else None,
            "body": self.body_redacted,
        })

    def with_effort(self, effort: Optional[str]) -> "GatewayRequest":
        req = copy.copy(self)
        req.body = copy.deepcopy(self.body)
        req.body_redacted = copy.deepcopy(self.body_redacted)
        req.reasoning_effort = effort
        for b in (req.body, req.body_redacted):
            if req.route == "rest_chat":
                if effort:
                    b["reasoning_effort"] = effort
                else:
                    b.pop("reasoning_effort", None)
            else:
                if effort:
                    b["reasoning"] = {"effort": effort}
                else:
                    b.pop("reasoning", None)
        return req

    def with_stream(self, stream: bool) -> "GatewayRequest":
        req = copy.copy(self)
        req.body = dict(self.body)
        req.body_redacted = dict(self.body_redacted)
        req.stream = stream
        if req.route == "responses":
            req.body["stream"] = stream
            req.body_redacted["stream"] = stream
        return req


@dataclass
class LLMResult:
    """What ``GatewayClient.send`` returns."""

    text: str
    parsed: Optional[dict]
    usage: dict
    latency_s: float
    request_id: Optional[str]
    route: str
    effort_applied: Optional[str]
    attempts: int
    raw: dict
    model: str = ""
    effort_requested: Optional[str] = None
    parse_error: Optional[str] = None
    schema_issues: list[str] = field(default_factory=list)
    repaired: bool = False
    finish_note: Optional[str] = None
    refusal: Optional[str] = None
    request_redacted: dict = field(default_factory=dict)
    est_cost_usd: Optional[float] = None

    @property
    def ok(self) -> bool:
        return self.parsed is not None

    def summary(self) -> dict:
        return {
            "route": self.route, "model": self.model, "effort_requested": self.effort_requested,
            "effort_applied": self.effort_applied, "attempts": self.attempts, "latency_s": round(self.latency_s, 2),
            "request_id": self.request_id, "usage": self.usage, "parsed": self.parsed is not None,
            "parse_error": self.parse_error, "schema_issues": self.schema_issues, "repaired": self.repaired,
            "finish_note": self.finish_note, "refusal": self.refusal, "est_cost_usd": self.est_cost_usd,
        }


class _WaitRetryAfter(wait_base):
    """Exponential backoff with jitter, raised to the server's ``Retry-After`` when present."""

    def __init__(self, initial: float = 2.0, maximum: float = 90.0, jitter: float = 3.0, cap_retry_after: float = 300.0):
        self.base = wait_exponential_jitter(initial=initial, max=maximum, jitter=jitter)
        self.cap = cap_retry_after

    def __call__(self, retry_state: RetryCallState) -> float:
        w = float(self.base(retry_state))
        exc = retry_state.outcome.exception() if retry_state.outcome and retry_state.outcome.failed else None
        ra = getattr(exc, "retry_after", None)
        if ra:
            w = max(w, min(float(ra), self.cap))
        return w


class _Counter:
    def __init__(self) -> None:
        self.n = 0


# ------------------------------------------------------------------------------------------ client
class GatewayClient:
    """Thread-safe client for the WSE AI Gateway (share one instance across worker threads)."""

    def __init__(self, cfg: Config, *, log: Optional[logging.Logger] = None, api_key: Optional[str] = None,
                 transport: Optional[httpx.BaseTransport] = None):
        llm = cfg.section("llm")
        self.cfg = cfg
        self.log = log or _LOG
        self.base_url: str = str(llm.get("base_url", "https://gateway.engineering.jhu.edu/gateway")).rstrip("/")
        self.model: str = str(llm.get("model", "openai/gpt-6-astra"))
        self.route: str = str(llm.get("route", "responses"))
        self.effort: Optional[str] = normalise_effort(llm.get("reasoning_effort", "xhigh"))
        self.max_tokens: Optional[int] = _as_int(llm.get("max_completion_tokens", 32000))
        self.image_detail: str = str(llm.get("image_detail", "high"))
        self.stream: bool = bool(llm.get("stream", True))
        self.timeout_s: float = float(llm.get("timeout_s", 1200))
        self.max_retries: int = int(llm.get("max_retries", 5))
        self.price_in: Optional[float] = _as_float(llm.get("price_per_m_input_usd"))
        self.price_out: Optional[float] = _as_float(llm.get("price_per_m_output_usd"))
        self.system_mode: str = str(llm.get("responses_system_mode", "message"))  # message | instructions
        if self.route not in ROUTES:
            raise ValueError(f"llm.route must be one of {ROUTES}, got {self.route!r}")
        self._key = api_key if api_key is not None else cfg.api_key()
        self._client = httpx.Client(
            timeout=httpx.Timeout(self.timeout_s, connect=30.0, pool=60.0),
            follow_redirects=True,
            transport=transport,
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
        )
        self._lock = threading.Lock()
        self._effort_overrides: dict[str, str] = {}
        self.wait: wait_base = _WaitRetryAfter()
        # usage ledger: every HTTP call is recorded (who, what for, tokens, cost); context is per thread
        from bench.usage import Ledger, key_id
        self.ledger = Ledger(cfg, log=self.log)
        self.key_id = key_id(self._key)
        self._ctx = threading.local()

    # ---- lifecycle --------------------------------------------------------------------------
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GatewayClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- usage attribution ------------------------------------------------------------------
    @contextlib.contextmanager
    def usage_context(self, **info: Any):
        """Attach stage/sample/dataset/batch to every call made by this thread inside the block."""
        prev = getattr(self._ctx, "info", None)
        self._ctx.info = {**(prev or {}), **info}
        try:
            yield
        finally:
            self._ctx.info = prev

    def _record_call(self, req: "GatewayRequest", *, raw: Optional[dict], hdrs: Optional[dict], latency_s: float,
                     attempt: int, error: Optional[BaseException] = None) -> None:
        info = dict(getattr(self._ctx, "info", None) or {})
        u = normalise_usage((raw or {}).get("usage"), req.route) if raw else {}
        status_code = getattr(error, "status", None) if error is not None else 200
        self.ledger.record(
            stage=info.get("stage", "other"), purpose=getattr(self._ctx, "purpose", None) or info.get("purpose", "main"),
            sample_id=info.get("sample_id"), dataset=info.get("dataset"), batch=info.get("batch"),
            model=str((raw or {}).get("model") or req.model), route=req.route, effort=req.reasoning_effort,
            status="ok" if error is None else "error", http_status=status_code,
            error=None if error is None else f"{type(error).__name__}: {str(error)[:200]}",
            prompt_tokens=u.get("prompt_tokens"), cached_tokens=u.get("cached_tokens"),
            completion_tokens=u.get("completion_tokens"), reasoning_tokens=u.get("reasoning_tokens"),
            total_tokens=u.get("total_tokens"), latency_s=round(latency_s, 2),
            request_id=_rid(hdrs or {}) if hdrs else getattr(error, "request_id", None), attempt=attempt,
            key_id=self.key_id)

    # ---- public API -------------------------------------------------------------------------
    def complete(self, user_parts: str | Sequence[dict], *, system: Optional[str] = None,
                 images: Sequence[str | Path] = (), schema: Optional[dict] = None, schema_name: str = "qa_output",
                 route: Optional[str] = None, reasoning_effort: Optional[str] = None,
                 max_tokens: Optional[int] = None, stream: Optional[bool] = None, repair: bool = True) -> LLMResult:
        """Build and send a request in one go (see ``build_request`` / ``send``)."""
        req = self.build_request(user_parts, system=system, images=images, schema=schema, schema_name=schema_name,
                                 route=route, reasoning_effort=reasoning_effort, max_tokens=max_tokens, stream=stream)
        return self.send(req, repair=repair)

    def build_request(self, user_parts: str | Sequence[dict], *, system: Optional[str] = None,
                      images: Sequence[str | Path] = (), schema: Optional[dict] = None,
                      schema_name: str = "qa_output", route: Optional[str] = None,
                      reasoning_effort: Optional[str] = None, max_tokens: Optional[int] = None,
                      stream: Optional[bool] = None) -> GatewayRequest:
        """Convert ``(system, user_parts)`` into the route specific body.

        ``user_parts`` is a string or an ordered list of ``{"type": "text", "text": ...}`` and
        ``{"type": "image", "path": ..., "caption": ...}`` items (``caption`` is used only for logging).
        Extra ``images`` are appended as image parts.
        """
        route = route or self.route
        if route not in ROUTES:
            raise ValueError(f"route must be one of {ROUTES}, got {route!r}")
        parts = [{"type": "text", "text": user_parts}] if isinstance(user_parts, str) else [dict(p) for p in user_parts]
        parts += [{"type": "image", "path": str(p)} for p in images]
        effort = normalise_effort(reasoning_effort) if reasoning_effort is not None else self.effort
        if effort == "max":
            self.log.warning("reasoning effort 'max' is not accepted by the gateway; sending 'xhigh'")
            effort = "xhigh"
        effort = self._effort_overrides.get(effort or "", effort)
        max_tokens = _as_int(max_tokens) if max_tokens is not None else self.max_tokens
        if route == "rest_chat":
            return self._build_rest_chat(system, parts, schema, schema_name, effort, max_tokens)
        use_stream = self.stream if stream is None else bool(stream)
        return self._build_responses(system, parts, schema, schema_name, effort, max_tokens, use_stream)

    def send(self, req: GatewayRequest, *, repair: bool = True) -> LLMResult:
        """POST with retries, effort step-down, SSE fallback and (optionally) one JSON repair call."""
        t0 = time.monotonic()
        counter = _Counter()
        stepped = False
        while True:
            try:
                raw, headers = self._post_with_retries(req, counter)
                break
            except EffortRejected as e:
                nxt = EFFORT_STEP_DOWN.get(req.reasoning_effort or "")
                if stepped or not nxt:
                    raise
                self.log.warning("gateway rejected reasoning effort %r (%s); stepping down to %r",
                                 req.reasoning_effort, e, nxt)
                with self._lock:
                    self._effort_overrides[req.reasoning_effort or ""] = nxt
                req = req.with_effort(nxt)
                stepped = True
            except NoCompletedEvent as e:
                if not req.stream:
                    raise
                self.log.warning("SSE stream had no response.completed event (%s); retrying with stream=false", e)
                req = req.with_stream(False)
        text = extract_output_text(raw, req.route)
        usage = normalise_usage(raw.get("usage"), req.route)
        refusal = extract_refusal(raw, req.route)
        finish = finish_info(raw, req.route)
        parsed, parse_error = extract_json(text)
        issues = schema_issues(parsed, req.schema) if (parsed is not None and req.schema) else []
        repaired = False
        if req.schema is not None and repair and (parsed is None or issues):
            why = parse_error or f"schema issues: {issues[:3]}"
            self.log.warning("model output is not valid JSON for the schema (%s); attempting one repair call", why)
            fixed, fixed_usage = self._repair_json(text, req.schema, req.schema_name, req.max_tokens, counter)
            if fixed is not None:
                fixed_issues = schema_issues(fixed, req.schema)
                if parsed is None or len(fixed_issues) < len(issues):
                    parsed, issues, parse_error, repaired = fixed, fixed_issues, None, True
            usage = sum_usage(usage, fixed_usage)
        rid = headers.get("x-request-id") or headers.get("cf-aig-request-id")
        result = LLMResult(
            text=text, parsed=parsed, usage=usage, latency_s=time.monotonic() - t0, request_id=rid,
            route=req.route, effort_applied=applied_effort(raw, req.route) or req.reasoning_effort,
            attempts=counter.n, raw=redact(raw), model=str(raw.get("model") or req.model),
            effort_requested=req.reasoning_effort, parse_error=parse_error, schema_issues=issues,
            repaired=repaired, finish_note=finish, refusal=refusal, request_redacted=req.redacted(),
        )
        result.est_cost_usd = self.estimate_cost(usage)
        return result

    def estimate_cost(self, usage: Optional[dict]) -> Optional[float]:
        """USD estimate from ``price_per_m_input_usd``/``price_per_m_output_usd`` (None when unpriced)."""
        if not usage:
            return None
        return self.ledger.pricing.cost(usage.get("prompt_tokens"), usage.get("cached_tokens"),
                                        usage.get("completion_tokens"))

    def list_models(self) -> list[str]:
        """``GET {base}/api/models`` -> model ids (shape-tolerant)."""
        r = self._client.get(self.base_url + MODELS_PATH, headers=self._headers(stream=False))
        if r.status_code >= 400:
            self._raise_http(r)
        data = r.json()
        return _model_ids(data)

    # ---- body builders ----------------------------------------------------------------------
    def _build_rest_chat(self, system: Optional[str], parts: list[dict], schema: Optional[dict], schema_name: str,
                         effort: Optional[str], max_tokens: Optional[int]) -> GatewayRequest:
        if any(p.get("type") == "image" for p in parts):
            raise UnsupportedInput("the rest_chat route is text-only on this gateway; use route='responses' for images")
        text = "\n\n".join(str(p.get("text", "")) for p in parts if p.get("type") == "text")
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": text})
        body: dict[str, Any] = {"model": self.model, "messages": messages}
        if max_tokens:
            body["max_completion_tokens"] = int(max_tokens)
        if effort:
            body["reasoning_effort"] = effort
        if schema is not None:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": schema_name, "strict": True, "schema": schema}}
        return GatewayRequest(route="rest_chat", url=self.base_url + REST_CHAT_PATH, body=body,
                              body_redacted=copy.deepcopy(body), reasoning_effort=effort, max_tokens=max_tokens,
                              schema=schema, schema_name=schema_name, stream=False, model=self.model)

    def _build_responses(self, system: Optional[str], parts: list[dict], schema: Optional[dict], schema_name: str,
                         effort: Optional[str], max_tokens: Optional[int], stream: bool) -> GatewayRequest:
        content, content_red, n_img, n_bytes = self._responses_content(parts)
        model = strip_provider_prefix(self.model)
        body: dict[str, Any] = {"model": model, "input": [], "stream": bool(stream)}
        body_red: dict[str, Any] = {"model": model, "input": [], "stream": bool(stream)}
        if system:
            if self.system_mode == "instructions":
                body["instructions"] = system
                body_red["instructions"] = system
            else:
                sys_item = {"role": "system", "content": [{"type": "input_text", "text": system}]}
                body["input"].append(sys_item)
                body_red["input"].append(copy.deepcopy(sys_item))
        body["input"].append({"role": "user", "content": content})
        body_red["input"].append({"role": "user", "content": content_red})
        if effort:
            body["reasoning"] = {"effort": effort}
            body_red["reasoning"] = {"effort": effort}
        if max_tokens:
            body["max_output_tokens"] = int(max_tokens)
            body_red["max_output_tokens"] = int(max_tokens)
        if schema is not None:
            fmt = {"format": {"type": "json_schema", "name": schema_name, "strict": True, "schema": schema}}
            body["text"] = fmt
            body_red["text"] = copy.deepcopy(fmt)
        return GatewayRequest(route="responses", url=self.base_url + RESPONSES_PATH, body=body, body_redacted=body_red,
                              reasoning_effort=effort, max_tokens=max_tokens, schema=schema, schema_name=schema_name,
                              stream=bool(stream), n_images=n_img, image_bytes=n_bytes, model=model)

    def _responses_content(self, parts: list[dict]) -> tuple[list[dict], list[dict], int, int]:
        content: list[dict] = []
        redacted: list[dict] = []
        n_img = n_bytes = 0
        for p in parts:
            typ = p.get("type")
            if typ == "text":
                item = {"type": "input_text", "text": str(p.get("text", ""))}
                content.append(item)
                redacted.append(dict(item))
            elif typ == "image":
                path = Path(str(p.get("path")))
                data_url, size = image_data_url(path)
                detail = p.get("detail") or self.image_detail
                content.append({"type": "input_image", "image_url": data_url, "detail": detail})
                redacted.append({"type": "input_image", "image_url": f"<image {size} bytes: {path.name}>",
                                 "detail": detail, "caption": p.get("caption")})
                n_img += 1
                n_bytes += size
            else:
                raise ValueError(f"unknown user part type {typ!r} (expected 'text' or 'image')")
        return content, redacted, n_img, n_bytes

    # ---- HTTP -------------------------------------------------------------------------------
    def _headers(self, *, stream: bool) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }

    def _post_with_retries(self, req: GatewayRequest, counter: _Counter) -> tuple[dict, dict]:
        retrying = Retrying(
            retry=retry_if_exception_type(RetryableError),
            stop=stop_after_attempt(max(1, self.max_retries + 1)),
            wait=self.wait,
            before_sleep=self._before_sleep,
            reraise=True,
        )
        return retrying(self._post_once, req, counter)

    def _before_sleep(self, retry_state: RetryCallState) -> None:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        sleep = retry_state.next_action.sleep if retry_state.next_action else 0.0
        self.log.warning("gateway call failed (attempt %d/%d): %s - retrying in %.1fs",
                         retry_state.attempt_number, self.max_retries + 1, exc, sleep)

    def _post_once(self, req: GatewayRequest, counter: _Counter) -> tuple[dict, dict]:
        """One HTTP call; recorded in the usage ledger whether it succeeds or fails."""
        t0 = time.monotonic()
        try:
            raw, hdrs = self._post_once_raw(req, counter)
        except Exception as e:
            self._record_call(req, raw=None, hdrs=None, latency_s=time.monotonic() - t0, attempt=counter.n, error=e)
            raise
        self._record_call(req, raw=raw, hdrs=hdrs, latency_s=time.monotonic() - t0, attempt=counter.n)
        return raw, hdrs

    def _post_once_raw(self, req: GatewayRequest, counter: _Counter) -> tuple[dict, dict]:
        counter.n += 1
        streaming = req.route == "responses" and req.stream
        headers = self._headers(stream=streaming)
        try:
            if streaming:
                with self._client.stream("POST", req.url, json=req.body, headers=headers) as r:
                    if r.status_code >= 400:
                        r.read()
                        self._raise_http(r, req.reasoning_effort)
                    hdrs = {k.lower(): v for k, v in r.headers.items()}
                    col = collect_sse(r.iter_lines())
                return self._resolve_stream(col, hdrs), hdrs
            r = self._client.post(req.url, json=req.body, headers=headers)
            if r.status_code >= 400:
                self._raise_http(r, req.reasoning_effort)
            hdrs = {k.lower(): v for k, v in r.headers.items()}
            try:
                data = r.json()
            except ValueError as e:
                raise GatewayError(f"non-JSON response body: {e}", status=r.status_code,
                                   body=r.text[:500], request_id=_rid(hdrs)) from e
            if not isinstance(data, dict):
                raise GatewayError(f"unexpected JSON body type {type(data).__name__}", status=r.status_code)
            return data, hdrs
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError, httpx.ProxyError) as e:
            raise RetryableError(f"{type(e).__name__}: {e or 'network error'}") from e

    def _resolve_stream(self, col: SSECollected, hdrs: dict) -> dict:
        if col.failed is not None:
            msg = _error_message(json.dumps(col.failed))
            err = col.failed.get("error") if isinstance(col.failed.get("error"), dict) else {}
            code = str((err or {}).get("code") or (err or {}).get("type") or "")
            if re.search(r"rate|server|overload|timeout|unavailable", code + " " + msg, re.I):
                raise RetryableError(f"stream error event: {msg}", request_id=_rid(hdrs))
            raise GatewayError(f"stream error event: {msg}", request_id=_rid(hdrs))
        if col.completed is not None:
            return col.completed
        if col.deltas:
            obj, _ = extract_json(col.delta_text)
            if obj is not None:
                self.log.warning("no response.completed event but the streamed deltas form valid JSON; "
                                 "using them (usage unknown)")
                return {"output": [{"type": "message", "role": "assistant",
                                    "content": [{"type": "output_text", "text": col.delta_text}]}],
                        "usage": None, "synthesised_from_deltas": True}
        raise NoCompletedEvent(f"{col.n_events} SSE events, last types: {col.event_types[-3:]}", request_id=_rid(hdrs))

    def _raise_http(self, r: httpx.Response, effort: Optional[str] = None) -> None:
        status = r.status_code
        body_text = r.text if r.text is not None else ""
        msg = _error_message(body_text)
        rid = r.headers.get("x-request-id") or r.headers.get("cf-aig-request-id")
        if status in _RETRYABLE_STATUS:
            raise RetryableError(f"HTTP {status}: {msg}", status=status, body=body_text[:2000], request_id=rid,
                                 retry_after=parse_retry_after(r.headers.get("retry-after")))
        if status == 400 and is_effort_error(msg, effort):
            raise EffortRejected(f"HTTP 400: {msg}", status=status, body=body_text[:2000], request_id=rid)
        if status in (401, 403):
            raise GatewayError(f"HTTP {status} (authentication/authorisation): {msg}", status=status,
                               body=body_text[:2000], request_id=rid)
        raise GatewayError(f"HTTP {status}: {msg}", status=status, body=body_text[:2000], request_id=rid)

    # ---- JSON repair ------------------------------------------------------------------------
    def _repair_json(self, invalid_text: str, schema: dict, schema_name: str, max_tokens: Optional[int],
                     counter: _Counter) -> tuple[Optional[dict], dict]:
        """One text-only ``rest_chat`` call asking the model to re-emit valid JSON for the schema."""
        prompt = (
            "Return only valid JSON matching this schema. Do not add commentary, markdown or code fences; "
            "preserve the content of the draft as faithfully as possible and fix only structural problems "
            "(missing fields, wrong types, invalid enum values, trailing text).\n\n"
            f"JSON schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
            f"Draft output to repair:\n{invalid_text}"
        )
        prev_purpose = getattr(self._ctx, "purpose", None)
        self._ctx.purpose = "json_repair"
        try:
            req = self._build_rest_chat(None, [{"type": "text", "text": prompt}], schema, schema_name, "low", max_tokens)
            raw, _ = self._post_with_retries(req, counter)
        except GatewayError as e:
            self._ctx.purpose = prev_purpose
            self.log.warning("JSON repair call failed: %s", e)
            return None, normalise_usage(None, "rest_chat")
        self._ctx.purpose = prev_purpose
        fixed, err = extract_json(extract_output_text(raw, "rest_chat"))
        if fixed is None:
            self.log.warning("JSON repair call did not return a JSON object (%s)", err)
        return fixed, normalise_usage(raw.get("usage"), "rest_chat")


def _as_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _rid(hdrs: dict) -> Optional[str]:
    return hdrs.get("x-request-id") or hdrs.get("cf-aig-request-id")


def _model_ids(data: Any) -> list[str]:
    items: Any = data
    if isinstance(data, dict):
        for k in ("data", "models", "items", "results"):
            if isinstance(data.get(k), list):
                items = data[k]
                break
        else:
            items = [data] if any(k in data for k in ("id", "name", "model", "modelId")) else list(data.keys())
    ids: list[str] = []
    for it in items if isinstance(items, list) else []:
        if isinstance(it, str):
            ids.append(it)
        elif isinstance(it, dict):
            v = next((it[k] for k in ("modelId", "model_id", "id", "model", "modelName", "model_name", "name", "slug")
                      if it.get(k)), None)
            if v:
                ids.append(str(v))
    return ids


def image_data_url(path: str | Path) -> tuple[str, int]:
    """Read an image file and return ``(data URL, size in bytes)``."""
    p = Path(path)
    data = p.read_bytes()
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}", len(data)


# ------------------------------------------------------------------------------------------ probe
PROBE_SCHEMA_TEXT: dict = {
    "type": "object", "additionalProperties": False,
    "properties": {"ok": {"type": "boolean"}, "model_self_report": {"type": "string"}},
    "required": ["ok", "model_self_report"],
}
PROBE_SCHEMA_IMAGE: dict = {
    "type": "object", "additionalProperties": False,
    "properties": {"description": {"type": "string"}, "surgery_type_guess": {"type": "string"},
                   "visible_instruments": {"type": "array", "items": {"type": "string"}}},
    "required": ["description", "surgery_type_guess", "visible_instruments"],
}
PROBE_VIDEO_REL = Path("cataract-101") / "videos" / "case_269.mp4"


def add_args(p: argparse.ArgumentParser) -> None:
    """CLI options of ``./ophbench probe``."""
    p.add_argument("--route", choices=list(ROUTES), default=None, help="route for the text probe (default: config)")
    p.add_argument("--effort", choices=list(EFFORTS), default=None, help="reasoning effort for the probes (default: low)")
    p.add_argument("--no-image", action="store_true", help="skip the 1-image call on the responses route")


def extract_probe_frame(cfg: Config, out_path: Path, *, t_s: float = 120.0, long_side: int = 512,
                        video: Optional[Path] = None) -> Path:
    """Grab one frame with ffmpeg and shrink it to a <=512 px JPEG with PIL."""
    from PIL import Image

    src = video or (Path(cfg.paths.datasets_root) / PROBE_VIDEO_REL)
    if not src.exists():
        raise FileNotFoundError(f"probe video not found: {src}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".raw.png")
    cmd = [cfg.ffmpeg(), "-y", "-loglevel", "error", "-ss", f"{t_s:.3f}", "-i", str(src),
           "-frames:v", "1", "-threads", "2", str(tmp)]
    res = run_cmd(cmd, timeout=180)
    if not res.ok or not tmp.exists():
        raise RuntimeError(f"ffmpeg frame extraction failed (rc={res.rc}): {res.err.strip()[:300]}")
    with Image.open(tmp) as im:
        rgb = im.convert("RGB")
        rgb.thumbnail((long_side, long_side))
        rgb.save(out_path, "JPEG", quality=85)
    tmp.unlink(missing_ok=True)
    return out_path


def main(args: argparse.Namespace, cfg: Config) -> int:
    """``./ophbench probe``: models endpoint, a tiny text call, and (unless --no-image) a 1-image call."""
    cfg.ensure_dirs()
    log = get_logger("probe", cfg)
    started = now_iso()
    route = getattr(args, "route", None) or cfg.get("llm.route", "responses")
    effort = getattr(args, "effort", None) or "low"
    no_image = bool(getattr(args, "no_image", False))
    rows: list[tuple[str, str, str]] = []
    summary: dict[str, Any] = {"route": route, "effort": effort, "checks": {}}
    log.info("probe: base=%s model=%s route=%s effort=%s", cfg.get("llm.base_url"), cfg.get("llm.model"), route, effort)
    try:
        client = GatewayClient(cfg, log=log)
        client._ctx.info = {"stage": "probe"}
    except (RuntimeError, ValueError) as e:
        log.error("cannot create client: %s", e)
        rows.append(("client", "FAIL", str(e)))
        _print_probe_table(rows)
        record_run(cfg, "probe", args, {**summary, "ok": False, "error": str(e)}, started_at=started)
        return 1

    # 1. models endpoint -------------------------------------------------------------------------
    model = str(cfg.get("llm.model"))
    try:
        ids = client.list_models()
        bare = strip_provider_prefix(model)
        present = any(bare in i or i in model for i in ids)
        rows.append(("GET /api/models", "ok", f"{len(ids)} model(s); configured model {'listed' if present else 'NOT listed'}"
                     + (f": {', '.join(ids[:6])}" if ids else "")))
        summary["checks"]["models"] = {"ok": True, "n": len(ids), "configured_present": present}
    except Exception as e:  # noqa: BLE001 - probe reports every failure
        rows.append(("GET /api/models", "warn", f"{type(e).__name__}: {str(e)[:160]}"))
        summary["checks"]["models"] = {"ok": False, "error": str(e)[:300]}
        log.warning("models endpoint failed: %s", e)

    # 2. text call ------------------------------------------------------------------------------
    text_ok = False
    try:
        res = client.complete(
            "Reply with JSON: set ok=true and model_self_report to the model name you believe you are.",
            system="You are a connectivity probe. Answer briefly.", schema=PROBE_SCHEMA_TEXT, schema_name="probe_text",
            route=route, reasoning_effort=effort, max_tokens=800, repair=False)
        text_ok = res.parsed is not None
        rows.append((f"text call ({route})", "ok" if text_ok else "warn",
                     f"{res.latency_s:.1f}s, tokens in/out/reasoning={res.usage.get('prompt_tokens')}/"
                     f"{res.usage.get('completion_tokens')}/{res.usage.get('reasoning_tokens')}, "
                     f"effort={res.effort_applied}, attempts={res.attempts}, rid={res.request_id}, "
                     f"text={res.text[:80]!r}"))
        summary["checks"]["text"] = {"ok": text_ok, **res.summary()}
    except Exception as e:  # noqa: BLE001
        rows.append((f"text call ({route})", "FAIL", f"{type(e).__name__}: {str(e)[:200]}"))
        summary["checks"]["text"] = {"ok": False, "error": str(e)[:300]}
        log.error("text probe failed: %s", e)

    # 3. image call -----------------------------------------------------------------------------
    if not no_image:
        frame = Path(cfg.paths.cache) / "probe_frame.jpg"
        try:
            extract_probe_frame(cfg, frame)
            res = client.complete(
                [{"type": "text", "text": "Frame from an ophthalmic surgery video at t=02:00.0. Describe it in one "
                                          "sentence, guess the surgery type and list the visible instruments."},
                 {"type": "image", "path": str(frame), "caption": "probe frame t=02:00.0"}],
                system="You are a connectivity probe for a surgical video pipeline. Answer in JSON only.",
                schema=PROBE_SCHEMA_IMAGE, schema_name="probe_image", route="responses", reasoning_effort=effort,
                max_tokens=2000, repair=False)
            ok = res.parsed is not None
            desc = (res.parsed or {}).get("description", res.text[:100]) if res.parsed else res.text[:100]
            rows.append(("image call (responses)", "ok" if ok else "warn",
                         f"{res.latency_s:.1f}s, tokens in/out/reasoning={res.usage.get('prompt_tokens')}/"
                         f"{res.usage.get('completion_tokens')}/{res.usage.get('reasoning_tokens')}, "
                         f"effort={res.effort_applied}, rid={res.request_id}, frame={frame.name}, desc={str(desc)[:90]!r}"))
            summary["checks"]["image"] = {"ok": ok, "frame": str(frame), **res.summary()}
        except Exception as e:  # noqa: BLE001
            rows.append(("image call (responses)", "FAIL", f"{type(e).__name__}: {str(e)[:200]}"))
            summary["checks"]["image"] = {"ok": False, "error": str(e)[:300]}
            log.error("image probe failed: %s", e)
    else:
        rows.append(("image call (responses)", "skip", "--no-image"))
    client.close()
    _print_probe_table(rows)
    summary["ok"] = text_ok
    record_run(cfg, "probe", args, summary, started_at=started)
    if not text_ok:
        log.error("model unreachable on route %s", route)
        return 1
    log.info("probe ok")
    return 0


def _print_probe_table(rows: list[tuple[str, str, str]]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        t = Table(title="ophbench probe", show_lines=False)
        t.add_column("check")
        t.add_column("status")
        t.add_column("detail", overflow="fold")
        for name, status, detail in rows:
            style = {"ok": "green", "FAIL": "bold red", "warn": "yellow", "skip": "dim"}.get(status, "")
            t.add_row(name, f"[{style}]{status}[/{style}]" if style else status, detail)
        Console().print(t)
    except Exception:  # pragma: no cover - rich is installed
        for name, status, detail in rows:
            print(f"{name:28s} {status:6s} {detail}")


if __name__ == "__main__":  # standalone: python -m bench.llm_client [--route R] [--effort E] [--no-image]
    import os
    import sys

    from bench.config import load_config

    _p = argparse.ArgumentParser(description="probe the WSE AI Gateway")
    add_args(_p)
    _p.add_argument("--config", default=None)
    _p.add_argument("--data-dir", default=None)
    _a = _p.parse_args()
    if _a.data_dir:
        os.environ["OPHBENCH_DATA_DIR"] = _a.data_dir
    sys.exit(main(_a, load_config(_a.config)))
