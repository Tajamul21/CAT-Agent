#!/usr/bin/env bash
# One-time sign-in so this server can upload batch zips to the JHU SharePoint folder (config: sharepoint.*).
#
#   bash scripts/sharepoint_setup.sh            # sign in through a link (run it in the VS Code terminal)
#   bash scripts/sharepoint_setup.sh --paste    # paste a token made on your Mac with: rclone authorize "onedrive"
#   bash scripts/sharepoint_setup.sh --reset    # throw away the saved sign-in and start again
#
# The sign-in (an OAuth refresh token) is stored only in ~/.config/rclone/rclone.conf (permissions 600) and
# renews itself; it is never printed or logged.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_sp_config.sh"
MODE="browser"; RESET=0
for a in "$@"; do case "$a" in --paste) MODE="paste";; --reset) RESET=1;; -h|--help) sed -n 2,9p "$0"; exit 0;; esac; done

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
if [ "$RESET" = 1 ]; then "$RCLONE" config delete "$REMOTE" 2>/dev/null || true; fi
if "$RCLONE" listremotes 2>/dev/null | grep -qx "$REMOTE:"; then
  if "$RCLONE" lsf "$REMOTE:$FOLDER" --max-depth 1 >/dev/null 2>&1; then
    say "Already set up: '$REMOTE:$FOLDER' is reachable. (Use --reset to sign in again.)"; exit 0
  fi
  say "A saved sign-in exists but the folder is not reachable; signing in again."; "$RCLONE" config delete "$REMOTE"
fi

TMP="$(mktemp)"; trap 'rm -f "$TMP"' EXIT; chmod 600 "$TMP"
if [ "$MODE" = "browser" ]; then
  say "Step 1/3: sign in with your JHU account"
  echo "Click the http://127.0.0.1:53682/... link printed below (VS Code opens it on your computer and forwards"
  echo "the port), sign in with your JHU account and accept. If the browser later shows 'site can't be reached'"
  echo "on localhost:53682, stop with Ctrl-C and use:  bash scripts/sharepoint_setup.sh --paste"
  "$RCLONE" authorize "onedrive" --auth-no-open-browser 2>&1 | tee "$TMP" | grep -v -E '^\{|access_token' || true
  TOKEN="$("$PY" - "$TMP" <<'PYEOF'
import re, sys
t = open(sys.argv[1]).read()
m = re.search(r"--->\s*(\{.*?\})\s*<---", t, re.S) or re.search(r"(\{\"access_token\".*?\})\s*$", t, re.S | re.M)
print(m.group(1).strip() if m else "")
PYEOF
)"
else
  say "Step 1/3: paste the token"
  echo "On your Mac run:  rclone authorize \"onedrive\"   (install with: brew install rclone)"
  echo "sign in, then copy the {...} text between '--->' and '<---End paste' and paste it here (input is hidden):"
  read -r -s TOKEN; echo
fi
[ -n "$TOKEN" ] || { echo "No token received - sign-in did not finish." >&2; exit 1; }
if ! printf '%s' "$TOKEN" | "$PY" -c 'import json,sys; json.load(sys.stdin)' 2>/dev/null; then
  echo "That does not look like the token JSON." >&2; exit 1
fi

say "Step 2/3: finding the '$LIBRARY' library of $SITE_URL"
DRIVE_ID="$(printf '%s' "$TOKEN" | SITE_URL="$SITE_URL" LIBRARY="$LIBRARY" "$PY" -c '
import json, os, sys, urllib.parse, urllib.request
tok = json.load(sys.stdin)["access_token"]
u = urllib.parse.urlparse(os.environ["SITE_URL"])
def get(url):
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + tok})
    return json.load(urllib.request.urlopen(req, timeout=30))
site = get(f"https://graph.microsoft.com/v1.0/sites/{u.netloc}:{u.path}?$select=id,webUrl")
sid = site["id"]
drives = get(f"https://graph.microsoft.com/v1.0/sites/{sid}/drives?$select=id,name,webUrl")["value"]
want = os.environ["LIBRARY"].lower()
pick = [d for d in drives if d["name"].lower() == want] or [d for d in drives if d["webUrl"].rstrip("/").endswith("Shared%20Documents")]
print(pick[0]["id"] if pick else "")
')" || { echo "Could not read the SharePoint site with this sign-in (no access to the site, or JHU blocked the app)." >&2; exit 1; }
[ -n "$DRIVE_ID" ] || { echo "Library '$LIBRARY' not found on $SITE_URL." >&2; exit 1; }

"$RCLONE" config create "$REMOTE" onedrive token "$TOKEN" drive_id "$DRIVE_ID" drive_type documentLibrary --non-interactive >/dev/null
chmod 600 "$("$RCLONE" config file | tail -1)"
unset TOKEN

say "Step 3/3: checking that this server can write to $FOLDER"
TEST=".ophbench_write_test_$$.txt"
echo "ophbench upload test $(date -Is)" | "$RCLONE" rcat "$REMOTE:$FOLDER/$TEST"
"$RCLONE" deletefile "$REMOTE:$FOLDER/$TEST"
say "Done. Upload a batch with:  bash scripts/upload_batch.sh 1"
