#!/usr/bin/env bash
# apply_patch.sh — Clone AEV-PLIG-refined and apply the source-embedding patch
# so that dataset_idx (0=experimental, 1=synthetic) is used during forward().
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/../aev-plig-refined"
PATCH="${SCRIPT_DIR}/patches/aevplig_source_embedding.patch"
UPSTREAM_COMMIT=21c8a6d9699e6a79dae88cc99cf52eab6eeeffaf

if [[ -d "$REPO_DIR" ]]; then
    echo "AEV-PLIG-refined already cloned at $REPO_DIR"
else
    echo "Cloning AEV-PLIG-refined..."
    git clone https://github.com/weitse-hsu/AEV-PLIG-refined.git "$REPO_DIR"
    git -C "$REPO_DIR" checkout -q "$UPSTREAM_COMMIT"
fi

cd "$REPO_DIR"

if [[ "$(git rev-parse HEAD)" != "$UPSTREAM_COMMIT" ]]; then
    echo "ERROR: $REPO_DIR is at $(git rev-parse --short HEAD), expected ${UPSTREAM_COMMIT:0:7}." >&2
    echo "Remove it and re-run, or check out ${UPSTREAM_COMMIT:0:7} there." >&2
    exit 1
fi

# Apply the patch to enable source embedding (synthetic vs experimental).
# The patch adds a dataset_idx embedding, concatenated onto the node features
# before the GNN layers, and widens the first GNN layer accordingly.
echo "Applying source-embedding patch..."
if git apply --check "$PATCH" 2>/dev/null; then
    git apply "$PATCH"
    echo "Patch applied successfully."
elif grep -q "dataset_idx_embedding" aev_plig/model.py; then
    echo "Patch already applied — skipping."
else
    echo "ERROR: patch does not apply cleanly to $(git rev-parse --short HEAD)." >&2
    echo "The patch targets AEV-PLIG-refined ${UPSTREAM_COMMIT:0:7}; regenerate it if you change UPSTREAM_COMMIT." >&2
    exit 1
fi

# Install the package in editable mode
echo "Installing AEV-PLIG-refined in editable mode..."
pip install -e . --quiet

echo "Done. AEV-PLIG-refined is patched and installed."
