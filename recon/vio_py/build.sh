#!/bin/bash
# recon/vio_py/vio_py*.so for the realsense-to-usd conda env, after build_deps.sh:
#   bash recon/vio_py/build.sh
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$HOME/miniforge3/envs/realsense-to-usd/bin/python}
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH  # not ROS 2's pybind11 / packages
export PATH=$(echo "$PATH" | tr : '\n' | grep -v /opt/ros | paste -sd:)
cmake -S "$HERE" -B "$HERE/build" -DPython3_EXECUTABLE="$PY" \
    -Dpybind11_DIR="$("$PY" -c 'import pybind11; print(pybind11.get_cmake_dir())')"
cmake --build "$HERE/build" -j"${JOBS:-8}"
