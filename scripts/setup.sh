#!/usr/bin/env bash
# Isolated Linux/CUDA environment for the pinned OpenRLHF training API.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  printf 'Training setup requires Linux x86_64 (a GPU pod or WSL2).\n' >&2
  exit 2
fi
python_command="${WORKSHOP_PYTHON:-python3}"
"$python_command" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10, 3.11 or 3.12"'
"$python_command" -m venv .venv
python_command="$PWD/.venv/bin/python"
"$python_command" -m pip install --upgrade pip wheel 'setuptools==78.1.0' packaging ninja psutil py-cpuinfo
"$python_command" -m pip install 'torch==2.8.0' 'torchvision==0.23.0' 'torchaudio==2.8.0' --index-url https://download.pytorch.org/whl/cu129
python_tag="$("$python_command" -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
"$python_command" -m pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-${python_tag}-${python_tag}-linux_x86_64.whl"
"$python_command" scripts/ensure_cuda_compiler.py
if [[ -x "$PWD/.venv/cuda-toolkit/bin/nvcc" ]]; then
  export CUDA_HOME="$PWD/.venv/cuda-toolkit"
fi
# This integration uses torch.optim on one device, so no DeepSpeed CUDA ops
# need compilation. Scope the build settings to pip; runtime still uses CUDA.
DS_BUILD_OPS=0 DS_ACCELERATOR=cpu "$python_command" -m pip install --no-build-isolation -r requirements.txt
"$python_command" -m workshop backend-check
printf 'Setup complete. Run bash scripts/run_b200.sh or bash scripts/run_b300.sh.\n'
