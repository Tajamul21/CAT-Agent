#!/usr/bin/env bash
# Upload a clinician batch package to the SharePoint folder (config: sharepoint.*), zipped.
#
#   bash scripts/upload_batch.sh 1          # zips <packages_dir>/batch_001 if needed, uploads batch_001.zip, verifies it
#
# For anything else (other files, folders, a menu to choose from) use:  python3 scripts/sp_upload.py
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_sp_config.sh"
K="${1:?usage: upload_batch.sh <batch number>}"
B="$(printf 'batch_%03d' "$K")"
cd "$REPO" && exec "$PY" scripts/sp_upload.py "$B" --zip "${@:2}"
