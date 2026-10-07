#!/usr/bin/env python3
"""Upload files or folders from this server to the JHU SharePoint folder (config: sharepoint.*).

    python3 scripts/sp_upload.py                      # menu: pick what to upload from the packages folder
    python3 scripts/sp_upload.py PATH [PATH ...]      # upload these files / folders
    python3 scripts/sp_upload.py batch_001 --zip      # zip a folder first, then upload the zip
    python3 scripts/sp_upload.py FILE --dest sub/dir  # put it in a subfolder of the SharePoint folder
    python3 scripts/sp_upload.py --list               # show what is already in the SharePoint folder

Relative names are looked up in the current directory first, then in the packages folder. Folders are
uploaded with their structure. Files that already exist in SharePoint with the same size are skipped (so an
interrupted upload can simply be re-run) unless --force. Every upload is verified by size and logged in
<data_dir>/logs/uploads.jsonl.

Uploads go straight to the shared folder by its ID through Microsoft Graph (upload sessions in 60 MiB
chunks). The sign-in is the rclone remote made by scripts/sharepoint_setup.sh; rclone keeps it fresh. The
token is read in memory only and never printed.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bench.batches import packages_dir  # noqa: E402
from bench.config import load_config  # noqa: E402

GRAPH = "https://graph.microsoft.com/v1.0"
CHUNK = 192 * 327680  # 60 MiB, a multiple of 320 KiB as Graph requires


# ------------------------------------------------------------------------------------ sign-in
class Remote:
    def __init__(self, rclone: str, remote: str, require_folder: bool = True):
        self.rclone, self.remote = rclone, remote
        self.require_folder = require_folder
        self._token_at = 0.0
        self._access = ""
        self.drive_id = self.folder_id = ""
        self.refresh()

    def refresh(self) -> None:
        """Let rclone renew the token if needed (any API call does), then read it from the rclone config."""
        subprocess.run([self.rclone, "lsf", f"{self.remote}:", "--max-depth", "1"], capture_output=True, timeout=120)
        out = subprocess.run([self.rclone, "config", "dump"], capture_output=True, text=True, timeout=60)
        cfg = json.loads(out.stdout or "{}").get(self.remote)
        if not cfg:
            sys.exit(f"Not signed in: no rclone remote '{self.remote}'. Run: bash scripts/sharepoint_setup.sh")
        self._access = json.loads(cfg["token"])["access_token"]
        self.drive_id, self.folder_id = cfg.get("drive_id", ""), cfg.get("root_folder_id", "")
        if not self.drive_id or (self.require_folder and not self.folder_id):
            sys.exit("The SharePoint folder is not configured yet. Run: bash scripts/sharepoint_setup.sh")
        self._token_at = time.time()

    def call(self, method: str, url: str, body: bytes | None = None, headers: dict | None = None,
             auth: bool = True, timeout: int = 120) -> dict:
        if auth and time.time() - self._token_at > 40 * 60:
            self.refresh()
        h = dict(headers or {})
        if auth:
            h["Authorization"] = "Bearer " + self._access
        for attempt in range(6):
            req = urllib.request.Request(url, data=body, method=method, headers=h)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    raw = r.read()
                    return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise FileNotFoundError(url) from None
                if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                    time.sleep(float(e.headers.get("Retry-After") or 2 ** attempt))
                    continue
                if e.code == 401 and auth and attempt == 0:
                    self.refresh(); h["Authorization"] = "Bearer " + self._access
                    continue
                raise RuntimeError(f"HTTP {e.code} for {method} {url.split('?')[0][:90]}: {e.read()[:300]!r}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt < 5:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"network error: {e}") from None
        raise RuntimeError("giving up after retries")

    # paths are relative to the shared folder
    def item_url(self, rel: str) -> str:
        rel = rel.strip("/")
        base = f"{GRAPH}/drives/{self.drive_id}/items/{self.folder_id}"
        return base if not rel else f"{base}:/{urllib.parse.quote(rel)}:"

    def stat(self, rel: str) -> dict | None:
        try:
            return self.call("GET", self.item_url(rel) + "?$select=name,size,folder,webUrl")
        except FileNotFoundError:
            return None

    def listdir(self, rel: str = "") -> list[dict]:
        url = self.item_url(rel) + "/children?$select=name,size,folder,lastModifiedDateTime&$top=500"
        out = []
        while url:
            d = self.call("GET", url)
            out += d.get("value", [])
            url = d.get("@odata.nextLink")
        return out

    def upload(self, local: Path, rel: str) -> dict:
        size = local.stat().st_size
        sess = self.call("POST", self.item_url(rel) + "/createUploadSession",
                         json.dumps({"item": {"@microsoft.graph.conflictBehavior": "replace"}}).encode(),
                         {"Content-Type": "application/json"})
        up = sess["uploadUrl"]   # pre-authenticated: chunk uploads need no token
        t0, sent, last = time.time(), 0, None
        with open(local, "rb") as fh:
            while sent < size or size == 0:
                chunk = fh.read(CHUNK)
                end = sent + len(chunk) - 1
                hdr = {"Content-Length": str(len(chunk)), "Content-Range": f"bytes {sent}-{end}/{size}"} if size else \
                      {"Content-Length": "0", "Content-Range": "bytes */0"}
                last = self.call("PUT", up, chunk, hdr, auth=False, timeout=600)
                sent += len(chunk)
                rate = sent / max(time.time() - t0, 1e-6) / 1e6
                print(f"\r    {sent / 1e6:8.1f} / {size / 1e6:.1f} MB  ({100 * sent / max(size, 1):5.1f}%)  {rate:5.1f} MB/s",
                      end="", flush=True)
                if size == 0:
                    break
        print()
        return last or {}


# ------------------------------------------------------------------------------------ helpers
def zip_folder(folder: Path) -> Path:
    z = folder.with_suffix(".zip")
    newest = max((p.stat().st_mtime for p in folder.rglob("*") if p.is_file()), default=0)
    if z.exists() and z.stat().st_mtime >= newest:
        print(f"  using existing {z.name} (up to date)")
        return z
    print(f"  zipping {folder.name} -> {z.name} ...")
    tmp = z.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(folder.parent))
    os.replace(tmp, z)
    return z


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1000 or u == "GB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1000
    return f"{n:.1f} GB"


def resolve(arg: str, pkg: Path) -> Path:
    p = Path(arg).expanduser()
    for cand in (p, pkg / arg):
        if cand.exists():
            return cand.resolve()
    sys.exit(f"Not found: {arg} (looked in the current directory and {pkg})")


def menu(pkg: Path) -> list[str]:
    items = sorted([p for p in pkg.iterdir() if not p.name.startswith(".")], key=lambda p: p.name) if pkg.is_dir() else []
    print(f"What do you want to upload? (packages folder: {pkg})")
    for i, p in enumerate(items, 1):
        kind = "folder" if p.is_dir() else human(p.stat().st_size)
        print(f"  {i:2d}. {p.name}  [{kind}]")
    print("Type numbers (e.g. 1 3), or a path to any file/folder on this server; Enter to cancel.")
    ans = input("> ").strip()
    if not ans:
        sys.exit("cancelled")
    picks = []
    for tok in ans.split():
        if tok.isdigit() and 1 <= int(tok) <= len(items):
            picks.append(str(items[int(tok) - 1]))
        else:
            picks.append(tok)
    return picks


def log_upload(data_dir: Path, rec: dict) -> None:
    (data_dir / "logs").mkdir(parents=True, exist_ok=True)
    with open(data_dir / "logs" / "uploads.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


# ------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="*", help="files or folders to upload (none: choose from a menu)")
    ap.add_argument("--dest", default="", help="subfolder inside the SharePoint folder")
    ap.add_argument("--zip", action="store_true", help="zip folders first and upload the zip")
    ap.add_argument("--force", action="store_true", help="re-upload files that already exist with the same size")
    ap.add_argument("--list", action="store_true", help="list the SharePoint folder and exit")
    ap.add_argument("--pin-folder", action="store_true",
                    help="(setup) look up config sharepoint.folder and pin the rclone remote to that folder's ID")
    ap.add_argument("--selftest", action="store_true", help="(setup) upload, verify and delete a tiny test file")
    args = ap.parse_args()

    cfg = load_config()
    sp = cfg.section("sharepoint")
    pkg = packages_dir(cfg)
    rclone, rname = sp.get("rclone") or "rclone", sp.get("remote") or "jhu-wound"
    label = sp.get("folder_label") or sp.get("folder") or "the SharePoint folder"

    if args.pin_folder:
        r = Remote(rclone, rname, require_folder=False)
        path = str(sp.get("folder") or "").strip("/")
        url = f"{GRAPH}/drives/{r.drive_id}/root:/{urllib.parse.quote(path)}?$select=id,name,folder"
        try:
            d = r.call("GET", url)
        except FileNotFoundError:
            sys.exit(f"Folder '{path}' not found in the SharePoint library (check sharepoint.folder in config/pipeline.yaml)")
        if "folder" not in d:
            sys.exit(f"'{path}' is not a folder")
        subprocess.run([rclone, "config", "update", rname, "root_folder_id", d["id"], "--non-interactive"],
                       check=True, capture_output=True)
        print(f"connection pinned to folder '{path}'")
        return 0

    remote = Remote(rclone, rname)
    if args.selftest:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "ophbench_upload_test.txt"
            f.write_text(f"ophbench upload test {datetime.now().isoformat()}\n")
            remote.upload(f, f.name)
            ok = (remote.stat(f.name) or {}).get("size") == f.stat().st_size
            remote.call("DELETE", remote.item_url(f.name))
        print("upload test passed: this server can write to " + label if ok else "upload test FAILED")
        return 0 if ok else 1

    if args.list:
        rows = remote.listdir(args.dest)
        print(f"{label}/{args.dest}".rstrip("/") + f": {len(rows)} item(s)")
        for r in rows:
            print(f"  {'[folder]' if 'folder' in r else human(r.get('size', 0)):>10}  {r['name']}")
        return 0

    paths = args.paths or menu(pkg)
    plan: list[tuple[Path, str]] = []
    for a in paths:
        p = resolve(a, pkg)
        if p.is_dir() and args.zip:
            p = zip_folder(p)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    plan.append((f, "/".join(x for x in (args.dest, p.name, str(f.relative_to(p))) if x)))
        else:
            plan.append((p, "/".join(x for x in (args.dest, p.name) if x)))
    total = sum(f.stat().st_size for f, _ in plan)
    print(f"Uploading {len(plan)} file(s), {human(total)}, to {label}" + (f"/{args.dest}" if args.dest else ""))

    done = skipped = failed = 0
    for i, (f, rel) in enumerate(plan, 1):
        size = f.stat().st_size
        have = remote.stat(rel)
        if have and not args.force and have.get("size") == size and "folder" not in have:
            print(f"  [{i}/{len(plan)}] {rel}: already there ({human(size)}), skipped")
            skipped += 1
            continue
        print(f"  [{i}/{len(plan)}] {rel} ({human(size)})")
        try:
            remote.upload(f, rel)
            got = remote.stat(rel) or {}
            if got.get("size") != size:
                raise RuntimeError(f"size check failed: local {size} bytes, SharePoint {got.get('size')}")
            done += 1
            log_upload(Path(cfg.paths.data), {
                "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
                "user": os.environ.get("OPHBENCH_USER") or getpass.getuser(), "file": str(f), "dest": rel,
                "bytes": size, "folder": label})
        except Exception as e:  # keep going with the other files
            failed += 1
            print(f"    FAILED: {e}")
    print(f"Done: {done} uploaded and verified, {skipped} already there, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
