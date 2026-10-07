# shared by sharepoint_setup.sh / upload_batch.sh: read the sharepoint section of config/pipeline.yaml
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${OPHBENCH_PYTHON:-/mnt/store/tashraf4/miniconda3/bin/python3}"
_sp() { "$PY" - "$1" <<'PYEOF'
import sys, yaml
import os
cfg = yaml.safe_load(open(os.environ["SP_CFG"])) or {}
print((cfg.get("sharepoint") or {}).get(sys.argv[1], ""))
PYEOF
}
export SP_CFG="${OPHBENCH_CONFIG:-$REPO/config/pipeline.yaml}"
RCLONE="${RCLONE:-$(_sp rclone)}"; RCLONE="${RCLONE:-rclone}"
REMOTE="${SHAREPOINT_REMOTE:-$(_sp remote)}"
SITE_URL="${SHAREPOINT_SITE_URL:-$(_sp site_url)}"
LIBRARY="${SHAREPOINT_LIBRARY:-$(_sp library)}"
FOLDER="${SHAREPOINT_FOLDER:-$(_sp folder)}"
