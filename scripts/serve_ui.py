#!/usr/bin/env python3
"""Local/VPN server for the ophbench annotation UI (stdlib only).

Serves the ``ui/`` folder with

* correct MIME types (mp4, json, js, css, jpg, ...),
* HTTP Range requests (206 Partial Content, ``Content-Range``, ``Accept-Ranges``) so the browser can
  seek inside preview videos,
* permissive CORS headers (``Access-Control-Allow-Origin: *``), useful when the UI is hosted on
  GitHub Pages and only the media is served from here,
* ``POST /api/annotations`` which stores the JSON body (an ANNOTATION RECORD submitted by the UI)
  as ``<data_dir>/annotations/<annotator>_<timestamp>.json`` and answers ``{"ok": true, ...}``,
* ``GET /api/health`` and ``GET /api/annotations`` (list stored files).

Usage::

    python3 scripts/serve_ui.py [--port 8765] [--bind 127.0.0.1] [--ui-dir ui] [--data-dir data] [--quiet]

Point ``ui/config.js`` ``SUBMIT_URL`` at ``api/annotations`` (relative) or
``http://<host>:<port>/api/annotations`` to make the Submit button store files here.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
from datetime import datetime, timezone
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_UI_DIR = REPO_ROOT / "ui"
MAX_BODY_BYTES = 64 * 1024 * 1024
_ANNOTATOR_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")

EXTRA_TYPES = {
    ".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
    ".json": "application/json", ".jsonl": "application/x-ndjson", ".js": "text/javascript",
    ".mjs": "text/javascript", ".css": "text/css", ".html": "text/html", ".htm": "text/html",
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif",
    ".svg": "image/svg+xml", ".webp": "image/webp", ".ico": "image/x-icon", ".md": "text/markdown",
    ".txt": "text/plain", ".gs": "text/plain", ".csv": "text/csv", ".woff2": "font/woff2",
}
NO_CACHE_EXT = {".json", ".html", ".htm", ".js", ".css", ".md"}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class _LimitedReader:
    """File-like wrapper that yields at most ``length`` bytes (for 206 responses)."""

    def __init__(self, fh, length: int):
        self._fh = fh
        self._left = max(0, length)

    def read(self, n: int = -1) -> bytes:
        if self._left <= 0:
            return b""
        if n is None or n < 0 or n > self._left:
            n = self._left
        data = self._fh.read(n)
        self._left -= len(data)
        return data

    def close(self) -> None:
        self._fh.close()


class UIRequestHandler(SimpleHTTPRequestHandler):
    """Static files + Range + CORS + the annotations API."""

    protocol_version = "HTTP/1.1"
    server_version = "ophbench-ui/1.0"

    def __init__(self, *args: Any, directory: str, data_dir: Path, quiet: bool = False, **kwargs: Any):
        self.data_dir = Path(data_dir)
        self.quiet = quiet
        super().__init__(*args, directory=directory, **kwargs)

    # ---------------------------------------------------------------- plumbing
    def guess_type(self, path: str) -> str:  # type: ignore[override]
        ext = os.path.splitext(str(path))[1].lower()
        if ext in EXTRA_TYPES:
            return EXTRA_TYPES[ext] + ("; charset=utf-8" if ext in (".js", ".mjs", ".css", ".html", ".htm", ".md", ".txt", ".gs", ".csv", ".json", ".jsonl") else "")
        return super().guess_type(path)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Range")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Accept-Ranges, Content-Length")
        if self.command in ("GET", "HEAD"):
            self.send_header("Accept-Ranges", "bytes")
            ext = os.path.splitext(urlsplit(self.path).path)[1].lower()
            if ext in NO_CACHE_EXT or urlsplit(self.path).path.endswith("/"):
                self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        if not self.quiet:
            super().log_message(fmt, *args)

    # ---------------------------------------------------------------- verbs
    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path.startswith("/api/"):
            self._api_get(path)
            return
        try:
            super().do_GET()
        except (BrokenPipeError, ConnectionResetError):
            pass  # browsers abort media requests all the time

    def do_HEAD(self) -> None:  # noqa: N802
        try:
            super().do_HEAD()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if path != "/api/annotations":
            self._drain_body(length)  # keep the keep-alive stream in sync
            self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown endpoint; POST /api/annotations"})
            return
        if length <= 0:
            self.close_connection = True
            self._json(HTTPStatus.LENGTH_REQUIRED, {"ok": False, "error": "Content-Length required"})
            return
        if length > MAX_BODY_BYTES:
            self.close_connection = True
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"ok": False, "error": f"body larger than {MAX_BODY_BYTES} bytes"})
            return
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"invalid JSON: {exc}"})
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("samples"), dict):
            self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "payload must be an object with a 'samples' object"})
            return
        try:
            info = self._store_annotations(payload)
        except OSError as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": f"could not write file: {exc}"})
            return
        self._json(HTTPStatus.OK, info)

    # ---------------------------------------------------------------- Range support
    def send_head(self):  # type: ignore[override]
        """Like the base implementation but honours a single ``Range: bytes=a-b`` header."""
        range_header = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not range_header or os.path.isdir(path) or not os.path.isfile(path):
            return super().send_head()
        m = _RANGE_RE.match(range_header.strip())
        if not m:
            return super().send_head()  # ignore unsupported/multi-range requests -> full 200
        try:
            fh = open(path, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None
        try:
            st = os.fstat(fh.fileno())
            size = st.st_size
            first, last = m.group(1), m.group(2)
            if first == "" and last == "":
                fh.close()
                return super().send_head()
            if first == "":  # suffix range: last N bytes
                n = int(last)
                start = max(0, size - n)
                end = size - 1
            else:
                start = int(first)
                end = int(last) if last else size - 1
                end = min(end, size - 1)
            if size == 0 or start >= size or start > end:
                fh.close()
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Last-Modified", self.date_time_string(int(st.st_mtime)))
            self.end_headers()
            fh.seek(start)
            return _LimitedReader(fh, end - start + 1)
        except Exception:
            fh.close()
            raise

    # ---------------------------------------------------------------- API
    def _api_get(self, path: str) -> None:
        if path == "/api/health":
            self._json(HTTPStatus.OK, {"ok": True, "service": "ophbench-ui", "ui_dir": self.directory,
                                       "data_dir": str(self.data_dir), "time": now_iso()})
        elif path == "/api/annotations":
            out_dir = self.data_dir / "annotations"
            files = []
            if out_dir.is_dir():
                for p in sorted(out_dir.glob("*.json")):
                    st = p.stat()
                    files.append({"file": p.name, "bytes": st.st_size,
                                  "modified": datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds")})
            self._json(HTTPStatus.OK, {"ok": True, "dir": str(out_dir), "files": files})
        else:
            self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown endpoint"})

    def _store_annotations(self, payload: dict) -> dict:
        annotator = _ANNOTATOR_RE.sub("_", str(payload.get("annotator") or "anonymous")).strip("_")[:64] or "anonymous"
        out_dir = self.data_dir / "annotations"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        out = out_dir / f"{annotator}_{stamp}.json"
        payload = dict(payload)
        payload["received_at"] = now_iso()
        tmp = out.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, out)
        samples = payload["samples"]
        n_q = sum(len(s.get("questions") or {}) for s in samples.values() if isinstance(s, dict))
        if not self.quiet:
            self.log_message("stored annotations from %s: %s (%d samples, %d questions)", annotator, out.name, len(samples), n_q)
        return {"ok": True, "file": out.name, "path": str(out), "annotator": annotator,
                "samples": len(samples), "questions": n_q, "rows": len(samples) + n_q, "received_at": payload["received_at"]}

    def _drain_body(self, length: int) -> None:
        """Read and discard an unread request body (bounded) so the next keep-alive request parses."""
        if length > MAX_BODY_BYTES:
            self.close_connection = True
            return
        while length > 0:
            chunk = self.rfile.read(min(length, 65536))
            if not chunk:
                break
            length -= len(chunk)

    def _json(self, status: HTTPStatus, obj: dict) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8765, help="TCP port (0 = pick a free port); default 8765")
    p.add_argument("--bind", default="127.0.0.1", help="address to bind; use 0.0.0.0 to allow other machines (VPN); default 127.0.0.1")
    p.add_argument("--ui-dir", default=str(DEFAULT_UI_DIR), help="folder to serve (default: <repo>/ui)")
    p.add_argument("--data-dir", default=None,
                   help="where <data-dir>/annotations/*.json are written (default: $OPHBENCH_DATA_DIR or ../data relative to the ui folder)")
    p.add_argument("--quiet", action="store_true", help="do not log every request")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    ui_dir = Path(args.ui_dir).resolve()
    if not (ui_dir / "index.html").is_file():
        print(f"error: {ui_dir} does not contain index.html (use --ui-dir)", file=sys.stderr)
        return 1
    data_dir = Path(args.data_dir or os.environ.get("OPHBENCH_DATA_DIR") or (ui_dir.parent / "data")).resolve()
    if not (ui_dir / "data" / "index.json").is_file():
        print(f"warning: {ui_dir / 'data' / 'index.json'} is missing - run `./ophbench export-ui` first", file=sys.stderr)

    handler = partial(UIRequestHandler, directory=str(ui_dir), data_dir=data_dir, quiet=args.quiet)
    try:
        server = ThreadingHTTPServer((args.bind, args.port), handler)
    except OSError as exc:
        print(f"error: cannot bind {args.bind}:{args.port}: {exc}", file=sys.stderr)
        return 1
    server.daemon_threads = True
    host, port = server.server_address[0], server.server_address[1]
    shown_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    print(f"ophbench UI:   http://{shown_host}:{port}/")
    if host in ("0.0.0.0", "::"):
        print(f"               (also reachable as http://{socket.gethostname()}:{port}/ from other machines)")
    print(f"serving:       {ui_dir}")
    print(f"annotations:   POST /api/annotations -> {data_dir / 'annotations'}")
    print("press Ctrl+C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
