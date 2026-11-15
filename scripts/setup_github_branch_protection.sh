#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# GitHub Branch Protection Setup für allpass-crossfade-gate
# -----------------------------------------------------------------------------

OWNER="${1:-composite-stack}"
REPO="${2:-dtqw-reverb}"
BRANCH="${3:-main}"

echo "==> Richte Branch Protection für $OWNER/$REPO auf Branch '$BRANCH' ein..."

PAYLOAD=$(cat <<JSON
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "allpass-crossfade-gate"
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": null,
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON
)

echo "Payload:"
echo "$PAYLOAD"

if command -v gh >/dev/null 2>&1; then
    echo "==> Wende Regel via GitHub CLI (gh api) an..."
    echo "$PAYLOAD" | gh api \
        --method PUT \
        -H "Accept: application/vnd.github+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        "/repos/$OWNER/$REPO/branches/$BRANCH/protection" \
        --input -
    echo "✅ Branch Protection Rule für 'allpass-crossfade-gate' erfolgreich aktiviert!"
else
    echo "Info: Ausführung via GitHub Connector oder cURL:"
    echo "PUT /repos/$OWNER/$REPO/branches/$BRANCH/protection"
fi
