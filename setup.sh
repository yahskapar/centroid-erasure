#!/bin/bash
# ============================================================================
# centroid-erasure environment setup
#
# Usage:
#   ./setup.sh conda        # fresh conda environment
#   ./setup.sh uv           # fresh uv virtual environment
# ============================================================================

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 {conda|uv}"
    exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

MODE=$1
ENV_NAME="centroid-erasure"
PYTHON_VERSION="3.10.20"

# Torch 2.6.0 + CUDA 12.4 is the exact combination used for the paper.
TORCH_VERSION="2.6.0"
TORCHVISION_VERSION="0.21.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

conda_setup() {
    echo ""
    echo "Setting up conda environment: ${ENV_NAME}"
    echo "-----------------------------------------------"
    command -v conda >/dev/null 2>&1 || {
        echo "conda not found: https://docs.conda.io/"
        exit 1
    }
    if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
        echo "Environment '${ENV_NAME}' already exists; refusing to delete it."
        echo "Activate it, choose a different name in setup.sh, or remove it explicitly."
        exit 1
    fi
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
    python -m pip install --upgrade pip
    python -m pip install "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" --index-url "${TORCH_INDEX}"
    python -m pip install -r requirements.txt
    python -m pip install --no-deps -e .
    echo ""
    echo "Done. Activate with:  conda activate ${ENV_NAME}"
}

uv_setup() {
    echo ""
    echo "Setting up uv virtual environment"
    echo "-----------------------------------------------"
    command -v uv >/dev/null 2>&1 || { echo "uv not found: https://docs.astral.sh/uv/"; exit 1; }
    if [ -e .venv ]; then
        echo ".venv already exists; refusing to replace it."
        echo "Activate it or remove it explicitly before requesting a fresh environment."
        exit 1
    fi
    uv venv --python "${PYTHON_VERSION}"
    source .venv/bin/activate
    uv pip install "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" --index-url "${TORCH_INDEX}"
    uv pip install -r requirements.txt
    uv pip install --no-deps -e .
    echo ""
    echo "Done. Activate with:  source .venv/bin/activate"
}

case "${MODE}" in
    conda) conda_setup ;;
    uv)    uv_setup ;;
    *)     echo "Usage: $0 {conda|uv}"; exit 1 ;;
esac
