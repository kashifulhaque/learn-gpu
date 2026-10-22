#!/usr/bin/env bash
#
# Deploys learn-gpu to the VM.
#
# The VM holds a clone of this repository, so a deploy is: push a commit, then
# have the VM fetch that commit and rebuild. What is running is therefore a
# commit you can name, which is the part rsync could never tell you.
#
# The server keeps its own .env, which is not in the repository, and its data
# lives in a Docker volume. Neither is touched here.
#
# The PDF book and the slides PDF are the two things built here rather than on
# the VM, because printing them needs Chrome and the VM has neither the memory
# nor the disk to spare. Each build skips itself when no chapter, deck, or layout file has
# changed. To deploy without touching the PDFs already on the VM, set
# LG_SKIP_BOOK=1.
#
# Usage:
#   scripts/deploy.sh [--host HOST] [--path PATH] [--key KEY] [--branch BRANCH]
#
# First time on a new server, or to convert an rsync deploy to a clone:
#   scripts/deploy.sh --init

set -euo pipefail

HOST="${LG_HOST:-ifkash@vm.ifkash.dev}"
REMOTE_PATH="${LG_REMOTE_PATH:-~/docs/learn-gpu}"
SSH_KEY="${LG_SSH_KEY:-$HOME/.ssh/id_ed25519_kh}"
BRANCH="${LG_BRANCH:-main}"
# The VM pulls over HTTPS: the repository is public, so it needs no deploy key.
REPO_URL="${LG_REPO_URL:-https://github.com/kashifulhaque/learn-gpu.git}"
INIT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)   HOST="$2"; shift 2 ;;
    --path)   REMOTE_PATH="$2"; shift 2 ;;
    --key)    SSH_KEY="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --init)   INIT=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=accept-new)
cd "$REPO_ROOT"

# --- checks before anything is touched on the server ------------------------

if [[ -n "$(git status --porcelain)" ]]; then
  echo "The working tree has uncommitted changes, which a git deploy cannot" >&2
  echo "carry. Commit them, or stash them, then run this again:" >&2
  echo >&2
  git status --short >&2
  exit 1
fi

COMMIT="$(git rev-parse HEAD)"
SHORT="$(git rev-parse --short HEAD)"

echo "==> Checking that $SHORT is on the remote"
git fetch --quiet origin "$BRANCH"
if ! git merge-base --is-ancestor "$COMMIT" "origin/$BRANCH"; then
  echo "$SHORT is not on origin/$BRANCH yet, so the server cannot fetch it." >&2
  echo "Push it first:" >&2
  echo >&2
  echo "  git push origin HEAD:$BRANCH" >&2
  exit 1
fi

if [[ "${LG_SKIP_BOOK:-0}" != "1" ]]; then
  echo "==> Building the PDF book"
  node frontend/scripts/build-book.mjs
  echo "==> Building the slides PDF"
  node frontend/scripts/build-slides.mjs
fi

# --- the server -------------------------------------------------------------

if [[ "$INIT" == "1" ]]; then
  echo "==> Turning $HOST:$REMOTE_PATH into a clone of $REPO_URL"
  ssh "${SSH_OPTS[@]}" "$HOST" "
    set -euo pipefail
    mkdir -p $REMOTE_PATH && cd $REMOTE_PATH
    if [ ! -d .git ]; then
      git init -q -b $BRANCH
      git remote add origin $REPO_URL
    fi
    git remote set-url origin $REPO_URL
  "
fi

echo "==> Fetching $SHORT on $HOST"
ssh "${SSH_OPTS[@]}" "$HOST" "
  set -euo pipefail
  cd $REMOTE_PATH
  if [ ! -d .git ]; then
    echo 'No git checkout at $REMOTE_PATH. Run scripts/deploy.sh --init once.' >&2
    exit 1
  fi
  if [ ! -f .env ]; then
    echo 'No .env at $REMOTE_PATH. Copy .env.example there and fill it in.' >&2
    exit 1
  fi
  git fetch --quiet origin $BRANCH
  # A deploy checkout has no work of its own, so it is reset rather than merged.
  # Untracked files stay behind, which is how .env survives.
  git reset --quiet --hard $COMMIT
  git branch --quiet --set-upstream-to=origin/$BRANCH $BRANCH 2>/dev/null || true
"

# The directory must exist before compose mounts it, or Docker creates it as
# root and the next copy fails.
ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p $REMOTE_PATH/book"
if [[ "${LG_SKIP_BOOK:-0}" != "1" ]]; then
  echo "==> Copying the PDF book and the slides PDF"
  # rsync does not expand a leading ~, and a relative remote path is relative
  # to the home directory anyway.
  rsync -az --delete --exclude .build --exclude .slides-build -e "ssh ${SSH_OPTS[*]}" \
    book/ "$HOST:${REMOTE_PATH#\~/}/book/"
fi

echo "==> Building and starting"
ssh "${SSH_OPTS[@]}" "$HOST" \
  "cd $REMOTE_PATH && GIT_SHA=$SHORT docker compose up -d --build"

echo "==> Waiting for the health check"
for _ in $(seq 1 60); do
  if ssh "${SSH_OPTS[@]}" "$HOST" \
      "curl -sf http://127.0.0.1:8091/api/health > /dev/null"; then
    echo "==> Healthy, running $SHORT"
    ssh "${SSH_OPTS[@]}" "$HOST" "curl -s http://127.0.0.1:8091/api/health"
    echo
    exit 0
  fi
  sleep 3
done

echo "The app did not become healthy. Recent logs:" >&2
ssh "${SSH_OPTS[@]}" "$HOST" "cd $REMOTE_PATH && docker compose logs --tail 60 app" >&2
exit 1
