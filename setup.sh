#!/bin/bash
# ============================================================================
# centroid-erasure environment setup
#
# Usage:
#   ./setup.sh conda        # fresh conda environment
#   ./setup.sh uv           # fresh uv virtual environment
# ============================================================================

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 {conda|uv}"
    exit 1
fi

MODE=$1
ENV_NAME="centroid-erasure"
PYTHON_VERSION="3.10"

# Torch 2.6.0 + CUDA 12.4 is the exact combination used for the paper.
TORCH_VERSION="2.6.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

conda_setup() {
    echo ""
    echo "Setting up conda environment: ${ENV_NAME}"
    echo "-----------------------------------------------"
    conda remove --name ${ENV_NAME} --all -y 2>/dev/null || true
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ${ENV_NAME}
    pip install --upgrade pip
    pip install torch==${TORCH_VERSION} --index-url ${TORCH_INDEX}
    pip install -r requirements.txt
    pip install -e .
    echo ""
    echo "Done. Activate with:  conda activate ${ENV_NAME}"
}

uv_setup() {
    echo ""
    echo "Setting up uv virtual environment"
    echo "-----------------------------------------------"
    command -v uv >/dev/null 2>&1 || { echo "uv not found: https://docs.astral.sh/uv/"; exit 1; }
    uv venv --python ${PYTHON_VERSION}
    source .venv/bin/activate
    uv pip install torch==${TORCH_VERSION} --index-url ${TORCH_INDEX}
    uv pip install -r requirements.txt
    uv pip install -e .
    echo ""
    echo "Done. Activate with:  source .venv/bin/activate"
}

case ${MODE} in
    conda) conda_setup ;;
    uv)    uv_setup ;;
    *)     echo "Usage: $0 {conda|uv}"; exit 1 ;;
esac
