#!/usr/bin/env bash
# Zip a clinician batch package (if needed) and upload it to the SharePoint folder in config: sharepoint.folder.
#
#   bash scripts/upload_batch.sh 1          # uploads <packages_dir>/batch_001.zip
#
# Run scripts/sharepoint_setup.sh once first. Each upload is appended to <data_dir>/logs/uploads.jsonl.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_sp_config.sh"
K="${1:?usage: upload_batch.sh <batch number>}"
B="$(printf 'batch_%03d' "$K")"
read -r PKG DATA < <(cd "$REPO" && "$PY" -c '
import sys; sys.path.insert(0, ".")
from bench.config import load_config; from bench.batches import packages_dir
c = load_config(); print(packages_dir(c), c.paths.data)')
ZIP="$PKG/$B.zip"
[ -d "$PKG/$B" ] || { echo "No package folder $PKG/$B - run: ./ophbench export-ui --batch $K" >&2; exit 1; }
"$RCLONE" listremotes | grep -qx "$REMOTE:" || { echo "Not signed in yet - run: bash scripts/sharepoint_setup.sh" >&2; exit 1; }

if [ ! -f "$ZIP" ] || [ -n "$(find "$PKG/$B" -newer "$ZIP" -print -quit)" ]; then
  echo "zipping $PKG/$B ..."; (cd "$PKG" && rm -f "$B.zip" && zip -0 -r -q "$B.zip" "$B")
fi
"$PY" -c "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); bad=z.testzip(); sys.exit('zip is damaged: '+bad if bad else 0)" "$ZIP"
SIZE=$(stat -c %s "$ZIP")
echo "uploading $B.zip ($((SIZE / 1000000)) MB) to $REMOTE:$FOLDER ..."
"$RCLONE" copyto "$ZIP" "$REMOTE:$FOLDER/$B.zip" --progress --onedrive-chunk-size 100M --retries 5 --low-level-retries 20
RSIZE=$("$RCLONE" lsjson "$REMOTE:$FOLDER/$B.zip" | "$PY" -c 'import json,sys; d=json.load(sys.stdin); print(d[0]["Size"] if d else -1)')
[ "$RSIZE" = "$SIZE" ] || { echo "Size check failed: local $SIZE bytes, SharePoint $RSIZE bytes" >&2; exit 1; }
mkdir -p "$DATA/logs"
printf '{"ts": "%s", "user": "%s", "batch": %d, "file": "%s.zip", "bytes": %s, "remote": "%s", "folder": "%s"}\n' \
  "$(date -Is)" "${OPHBENCH_USER:-$(whoami)}" "$K" "$B" "$SIZE" "$REMOTE" "$FOLDER" >> "$DATA/logs/uploads.jsonl"
echo "Uploaded and verified: $B.zip ($SIZE bytes) is in $SITE_URL -> $LIBRARY/$FOLDER"
