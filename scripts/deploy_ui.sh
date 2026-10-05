#!/usr/bin/env bash
# Prepare the ophbench annotation UI (ui/) as a push-ready GitHub Pages repository.
#
# Usage:
#   GITHUB_REPO=user/repo scripts/deploy_ui.sh [options]
#   scripts/deploy_ui.sh user/repo [options]
#
# Options:
#   --branch NAME     branch to publish from: gh-pages (default) or main
#   --push            actually run `git push` (by default only the exact command is printed)
#   --no-media        do not commit ui/media/ (host the videos elsewhere and set MEDIA_BASE_URL in ui/config.js)
#   --ssh             use git@github.com:user/repo.git instead of https://github.com/user/repo.git
#   --remote NAME     git remote name (default: origin)
#   --ui-dir PATH     UI folder (default: <repo>/ui)
#   --message MSG     commit message (default: "ophbench UI export <timestamp>")
#   -h, --help        this help
#
# What it does: checks the export, warns about GitHub size limits (100 MB per file, ~1 GB per
# site), refuses symlinked media (git would commit the links, not the files), writes .nojekyll and a
# .gitignore, initialises a git repository INSIDE ui/ on the chosen branch, commits everything, adds
# the remote and prints the push command plus the GitHub Pages settings. It never pushes unless
# --push is given (this server has no GitHub credentials; push from a machine that has).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
UI_DIR="$REPO_ROOT/ui"
BRANCH="gh-pages"
REMOTE="origin"
DO_PUSH=0
NO_MEDIA=0
USE_SSH=0
MESSAGE=""
GITHUB_REPO="${GITHUB_REPO:-}"

usage() { sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
die() { echo "error: $*" >&2; exit 1; }
warn() { echo "warning: $*" >&2; }
info() { echo "-- $*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --branch) BRANCH="${2:?--branch needs a value}"; shift 2 ;;
    --push) DO_PUSH=1; shift ;;
    --no-media) NO_MEDIA=1; shift ;;
    --ssh) USE_SSH=1; shift ;;
    --remote) REMOTE="${2:?--remote needs a value}"; shift 2 ;;
    --ui-dir) UI_DIR="$(cd "${2:?--ui-dir needs a value}" && pwd)"; shift 2 ;;
    --message) MESSAGE="${2:?--message needs a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*) die "unknown option $1 (see --help)" ;;
    *) if [ -z "$GITHUB_REPO" ]; then GITHUB_REPO="$1"; shift; else die "unexpected argument $1"; fi ;;
  esac
done

[ -n "$GITHUB_REPO" ] || { usage; echo; die "GITHUB_REPO=user/repo is required (env or first argument)"; }
[[ "$GITHUB_REPO" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || die "GITHUB_REPO must look like user/repo (got '$GITHUB_REPO')"
GITHUB_REPO="${GITHUB_REPO%.git}"
GH_USER="${GITHUB_REPO%%/*}"
GH_NAME="${GITHUB_REPO##*/}"
case "$BRANCH" in gh-pages|main) ;; *) warn "unusual branch '$BRANCH'; GitHub Pages can publish any branch, remember to pick it in Settings > Pages" ;; esac
command -v git >/dev/null 2>&1 || die "git is not installed"

# ------------------------------------------------------------------ sanity checks on the export
[ -f "$UI_DIR/index.html" ] || die "$UI_DIR/index.html not found (use --ui-dir)"
for f in app.js styles.css config.js; do [ -f "$UI_DIR/$f" ] || die "$UI_DIR/$f missing"; done
if [ ! -f "$UI_DIR/data/index.json" ]; then
  warn "$UI_DIR/data/index.json is missing - run ./ophbench export-ui first (deploying an empty UI)"
else
  N_SAMPLES=$(python3 - "$UI_DIR/data/index.json" <<'PY' 2>/dev/null || echo "?"
import json, sys
print(len(json.load(open(sys.argv[1]))))
PY
)
  info "export contains $N_SAMPLES samples ($(ls "$UI_DIR/data/samples" 2>/dev/null | wc -l | tr -d ' ') sample files)"
fi

# ------------------------------------------------------------------ media checks
MEDIA_DIR="$UI_DIR/media"
if [ -d "$MEDIA_DIR" ] && [ "$NO_MEDIA" -eq 0 ]; then
  N_LINKS=$(find "$MEDIA_DIR" -type l 2>/dev/null | wc -l | tr -d ' ')
  if [ "$N_LINKS" -gt 0 ]; then
    die "ui/media contains $N_LINKS symlinks (export-ui --media-mode symlink). Git would commit the links, not the files.
       Either re-run:  ./ophbench export-ui --media-mode copy
       or deploy without media:  $0 $GITHUB_REPO --no-media   (and set MEDIA_BASE_URL in ui/config.js)"
  fi
  MEDIA_KB=$(du -sk "$MEDIA_DIR" 2>/dev/null | cut -f1)
  MEDIA_MB=$(( MEDIA_KB / 1024 ))
  N_BIG=$(find "$MEDIA_DIR" -type f -size +100M 2>/dev/null | wc -l | tr -d ' ')
  N_LARGE=$(find "$MEDIA_DIR" -type f -size +50M 2>/dev/null | wc -l | tr -d ' ')
  info "ui/media is ${MEDIA_MB} MB ($(find "$MEDIA_DIR" -type f | wc -l | tr -d ' ') files)"
  if [ "$N_BIG" -gt 0 ]; then
    die "$N_BIG media files exceed GitHub's hard limit of 100 MB per file; GitHub will reject the push.
       Use --no-media and host ui/media on another server (set MEDIA_BASE_URL in ui/config.js), or
       re-export with a smaller preview (media.preview.max_duration_s / crf in config/pipeline.yaml)."
  fi
  [ "$N_LARGE" -gt 0 ] && warn "$N_LARGE media files are larger than 50 MB (GitHub warns above 50 MB)"
  if [ "$MEDIA_MB" -gt 1024 ]; then
    warn "ui/media is ${MEDIA_MB} MB. GitHub Pages sites are limited to ~1 GB and repositories should stay under 1-5 GB."
    warn "Strongly consider --no-media and serving the videos from another host (MEDIA_BASE_URL), e.g. a lab web server"
    warn "or an object store with HTTPS + HTTP Range support. The UI falls back to contact sheets when a video is missing."
  elif [ "$MEDIA_MB" -gt 300 ]; then
    warn "ui/media is ${MEDIA_MB} MB; pushes of this size are slow and GitHub may throttle. --no-media + MEDIA_BASE_URL is the robust option."
  fi
fi

# ------------------------------------------------------------------ files GitHub Pages needs
touch "$UI_DIR/.nojekyll"
GI="$UI_DIR/.gitignore"
{
  echo "# generated by scripts/deploy_ui.sh"
  echo "data/annotations/"
  echo "*.tmp"
  echo ".DS_Store"
  echo "Thumbs.db"
  [ "$NO_MEDIA" -eq 1 ] && echo "media/"
} > "$GI"

# ------------------------------------------------------------------ git repository inside ui/
if [ ! -d "$UI_DIR/.git" ]; then
  git init -q "$UI_DIR"
  info "initialised git repository in $UI_DIR"
fi
g() { git -C "$UI_DIR" "$@"; }

# Publish the current working tree onto $BRANCH without touching the files: point HEAD at the branch
# (created on first commit when it does not exist yet) and re-sync the index. This never fails on
# untracked/ignored files the way `git checkout` can when media tracking differs between branches.
g symbolic-ref HEAD "refs/heads/$BRANCH"
if g rev-parse --verify -q HEAD >/dev/null 2>&1; then g reset -q; fi

if [ -z "$(g config user.email || true)" ]; then
  g config user.name "ophbench"
  g config user.email "ophbench@localhost"
  warn "git identity not configured; using 'ophbench <ophbench@localhost>' for this repository only"
fi

if [ "$NO_MEDIA" -eq 1 ] && g ls-files --error-unmatch media >/dev/null 2>&1; then
  g rm -r -q --cached media
  info "removed previously committed media/ from the index (--no-media)"
fi

g add -A
if g diff --cached --quiet; then
  info "nothing new to commit"
else
  [ -n "$MESSAGE" ] || MESSAGE="ophbench UI export $(date '+%Y-%m-%d %H:%M')"
  g commit -q -m "$MESSAGE"
  info "committed: $MESSAGE"
fi

if [ "$USE_SSH" -eq 1 ]; then REMOTE_URL="git@github.com:$GITHUB_REPO.git"; else REMOTE_URL="https://github.com/$GITHUB_REPO.git"; fi
if g remote get-url "$REMOTE" >/dev/null 2>&1; then
  g remote set-url "$REMOTE" "$REMOTE_URL"
else
  g remote add "$REMOTE" "$REMOTE_URL"
fi

PUSH_CMD="git -C \"$UI_DIR\" push -u $REMOTE $BRANCH"
PAGES_URL="https://$GH_USER.github.io/$GH_NAME/"
[ "$GH_NAME" = "$GH_USER.github.io" ] && PAGES_URL="https://$GH_USER.github.io/"

cat <<EOF

================================================================================
 UI repository ready: $UI_DIR  (branch $BRANCH, remote $REMOTE -> $REMOTE_URL)
 $(g log --oneline | wc -l | tr -d ' ') commit(s), $(g ls-files | wc -l | tr -d ' ') tracked files
================================================================================
 1. Create the repository https://github.com/$GITHUB_REPO (empty, no README) if it does not exist.
 2. Push (from a machine with GitHub access; this server has none):
      $PUSH_CMD
    If the ui/ folder cannot push directly, copy it first, e.g.
      rsync -a --delete "$UI_DIR/" laptop:ophbench-ui/   && cd ophbench-ui && git push -u $REMOTE $BRANCH
 3. GitHub: Settings > Pages > Build and deployment > Source: "Deploy from a branch",
    Branch: "$BRANCH", folder "/ (root)" > Save. The site appears in 1-2 minutes at
      $PAGES_URL
    (private repos need GitHub Pro/Team for Pages; otherwise make the repo public but keep patient
     data out of it, or host on an internal web server with scripts/serve_ui.py instead).
 4. After updating the export, re-run this script and push again.
EOF
if [ "$NO_MEDIA" -eq 1 ]; then
  cat <<EOF
 NOTE: media/ is excluded. Host ui/media/ on an HTTPS server that supports HTTP Range requests and
       set MEDIA_BASE_URL in $UI_DIR/config.js (e.g. "https://media.example.org/ophbench"), then
       re-run this script to commit config.js.
EOF
fi
cat <<EOF
 Submissions: set SUBMIT_URL in config.js to the Apps Script web-app URL (ui/submit/Code.gs) so
       Submit works from GitHub Pages; otherwise annotators use Export JSON.
================================================================================
EOF

if [ "$DO_PUSH" -eq 1 ]; then
  info "pushing to $REMOTE_URL ($BRANCH)"
  g push -u "$REMOTE" "$BRANCH"
  info "pushed. Configure Settings > Pages as described above."
fi
