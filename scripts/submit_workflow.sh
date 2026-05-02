#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-openai/Qwen/Qwen3-Next-80B-A3B-Thinking}"
NEBIUS_API_BASE_URL="${NEBIUS_API_BASE_URL:-https://api.tokenfactory.nebius.com/v1/}"

BRANCH="eesha"
COMMIT_MESSAGE="Submit purple agent 1 for model $MODEL_NAME"
REMOTE="${REMOTE:-origin}"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

echo "Switching to branch: $BRANCH"
git switch "$BRANCH"

if command -v gh >/dev/null 2>&1; then
  echo "Ensuring GitHub Actions secrets are configured for Nebius Token Factory..."
  printf '%s' "$MODEL_NAME" | gh secret set MODEL_NAME --body-file -
  printf '%s' "$NEBIUS_API_BASE_URL" | gh secret set OPENAI_API_BASE --body-file -
  printf '%s' "$NEBIUS_API_BASE_URL" | gh secret set LITELLM_API_BASE_URL --body-file -

  if [ -n "${NEBIUS_API_KEY:-}" ]; then
    printf '%s' "$NEBIUS_API_KEY" | gh secret set NEBIUS_API_KEY --body-file -
  else
    echo "NEBIUS_API_KEY env var is not set locally; leaving existing GitHub secret unchanged."
  fi
else
  echo "gh CLI not found; skipping GitHub secret updates."
  echo "Make sure the repo has a NEBIUS_API_KEY secret before the workflow runs."
fi

# echo "Running local checks..."
# uv sync --extra test
# python3 -m py_compile src/agent.py src/config.py src/roles.py src/server.py
# uv run pytest tests/test_pure_networkx_agent.py

echo "Committing changes..."
git add .
if git diff --cached --quiet; then
  echo "No changes to commit."
else
  git commit -m "$COMMIT_MESSAGE"
fi

echo "Pushing to GitHub..."
git push -u "$REMOTE" "$BRANCH"

echo "Done."
