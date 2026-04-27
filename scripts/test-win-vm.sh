#!/usr/bin/env bash
# Test the latest CI-built Windows wheel inside the local Windows VM.
# Prereq (one-time): run scripts/win-vm-setup-ssh.ps1 inside the VM as Administrator.
#
# Usage: ./scripts/test-win-vm.sh [win_username]

set -euo pipefail

WIN_USER="${1:-}"
SSH_PORT=2222
REPO="jaywyawhare/GigaVector"

if [[ -z "$WIN_USER" ]]; then
    echo "Usage: $0 <windows_username>"
    exit 1
fi

SSH="ssh -p $SSH_PORT -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no ${WIN_USER}@localhost"
SCP="scp -P $SSH_PORT -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no"

echo "==> Fetching latest Windows wheel artifact from CI..."
RUN_ID=$(gh api "repos/$REPO/actions/runs?branch=master&status=success&per_page=5" \
    --jq '.workflow_runs[] | select(.name=="Publish to PyPI") | .id' | head -1)

if [[ -z "$RUN_ID" ]]; then
    echo "No successful 'Publish to PyPI' run found. Trying 'Build Python wheels'..."
    RUN_ID=$(gh api "repos/$REPO/actions/runs?branch=master&status=success&per_page=5" \
        --jq '.workflow_runs[] | select(.name=="Build Python wheels") | .id' | head -1)
fi

echo "==> Run ID: $RUN_ID"

ARTIFACT_URL=$(gh api "repos/$REPO/actions/runs/$RUN_ID/artifacts" \
    --jq '.artifacts[] | select(.name | test("windows")) | .archive_download_url' | head -1)

echo "==> Downloading Windows wheel artifact..."
TMP_DIR=$(mktemp -d)
gh api "$ARTIFACT_URL" > "$TMP_DIR/windows-wheel.zip"
unzip -q "$TMP_DIR/windows-wheel.zip" -d "$TMP_DIR/wheel/"

WHEEL_FILE=$(ls "$TMP_DIR/wheel/"*win_amd64*.whl 2>/dev/null | head -1)
if [[ -z "$WHEEL_FILE" ]]; then
    echo "ERROR: no win_amd64 wheel found in artifact"
    ls "$TMP_DIR/wheel/"
    exit 1
fi
echo "==> Wheel: $(basename "$WHEEL_FILE")"

echo "==> Copying wheel to Windows VM..."
$SCP "$WHEEL_FILE" "${WIN_USER}@localhost:C:/Users/${WIN_USER}/gigavector_test.whl"

echo "==> Installing and testing in Windows VM..."
$SSH "python -m pip install --force-reinstall C:/Users/${WIN_USER}/gigavector_test.whl && python -c \"
import gigavector
db = gigavector.GigaVectorDB(':memory:')
v = gigavector.Vector([1.0, 0.0, 0.0])
db.insert(v, {'tag': 'test'})
results = db.search([1.0, 0.0, 0.0], top_k=1)
print('PASS: found', len(results), 'result(s)')
print('Result:', results)
\""

echo ""
echo "==> Cleaning up..."
rm -rf "$TMP_DIR"
echo "==> Done."
