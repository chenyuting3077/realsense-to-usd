# source env.sh — activate the conda env ($RS2USD_ENV, default realsense-to-usd) and point CUDA at the CUDA 13
# toolkit pip-installed into it (gsplat / MonoGS JIT builds, nvblox's NPP libs).
# PYTHONPATH is set to the repo root (replacing e.g. ROS's) so the scripts in recon/ and gs/ can import frames.py and each other.
_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_had_u=0; [[ $- == *u* ]] && _had_u=1 && set +u   # conda's activate scripts use unset vars
eval "$("${CONDA_EXE:-$HOME/miniforge3/bin/conda}" shell.bash hook)"
conda activate "${RS2USD_ENV:-realsense-to-usd}"
((_had_u)) && set -u; unset _had_u
export CUDA_HOME=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)}
export MAX_JOBS=${MAX_JOBS:-16}
export TRITON_PTXAS_PATH=$CUDA_HOME/bin/ptxas   # Triton's bundled ptxas does not know Thor (sm_110a)
export PYTHONPATH=$_root; unset _root
