#!/bin/bash
# OpenVINS (no ROS, with Ceres 2.2) and RTAB-Map 0.23.13 (core only) for recon/vio_py, into
# ~/.local/{ceres,open_vins,rtabmap}, against the system's Eigen 3.4 and OpenCV 4.6 (conda's Ceres is built with
# Eigen 5: not to be mixed):
#   git clone --depth 1 --branch 2.2.0 https://github.com/ceres-solver/ceres-solver.git ~/ceres-2.2
#   git clone https://github.com/rpng/open_vins.git ~/open_vins && git -C ~/open_vins checkout 6948812
#     (master: v2.7 still uses Ceres' LocalParameterization, gone in Ceres 2.2)
#   git clone --depth 1 --branch 0.23.13-jazzy https://github.com/introlab/rtabmap.git ~/rtabmap
#   bash recon/vio_py/build_deps.sh
set -e
# a sourced ROS 2 makes OpenVINS build its ROS 2 node and RTAB-Map pick ROS' packages: hide it
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH ROS_DISTRO ROS_VERSION PYTHONPATH
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr : '\n' | grep -v /opt/ros | paste -sd:)
export PATH=$(echo "$PATH" | tr : '\n' | grep -v /opt/ros | paste -sd:)  # CMake searches prefixes of PATH too
JOBS=${JOBS:-8}
OPENCV=/usr/lib/$(uname -m)-linux-gnu/cmake/opencv4

# Ceres for OpenVINS' initializer: miniglog, no gflags, SuiteSparse or CUDA
[ -f $HOME/.local/ceres/lib/cmake/Ceres/CeresConfig.cmake ] || {
cmake -S ~/ceres-2.2 -B ~/ceres-2.2/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/.local/ceres \
    -DMINIGLOG=ON -DGFLAGS=OFF -DSUITESPARSE=OFF -DUSE_CUDA=OFF -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF \
    -DBUILD_BENCHMARKS=OFF -DBUILD_SHARED_LIBS=ON -DEigen3_DIR=/usr/share/eigen3/cmake
cmake --build ~/ceres-2.2/build -j$JOBS
cmake --install ~/ceres-2.2/build
}

cmake -S ~/open_vins/ov_msckf -B ~/open_vins/build -DCMAKE_BUILD_TYPE=Release -DENABLE_ROS=OFF \
    -DCMAKE_DISABLE_FIND_PACKAGE_ament_cmake=ON -DCMAKE_DISABLE_FIND_PACKAGE_catkin=ON \
    -DCMAKE_INSTALL_PREFIX=$HOME/.local/open_vins -DOpenCV_DIR=$OPENCV -DCeres_DIR=$HOME/.local/ceres/lib/cmake/Ceres \
    -DEigen3_DIR=/usr/share/eigen3/cmake -DCMAKE_INSTALL_RPATH=$HOME/.local/ceres/lib
cmake --build ~/open_vins/build -j$JOBS
cmake --install ~/open_vins/build

# RTAB-Map's core library only: no GUI, apps or tools; system PCL, OpenCV, SQLite
cmake -S ~/rtabmap -B ~/rtabmap/build_core -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/.local/rtabmap \
    -DOpenCV_DIR=$OPENCV -DWITH_QT=OFF -DBUILD_APP=OFF -DBUILD_TOOLS=OFF -DBUILD_EXAMPLES=OFF \
    -DWITH_PYTHON=OFF -DWITH_CERES=OFF -DWITH_TORCH=OFF -DWITH_OPENGV=OFF
cmake --build ~/rtabmap/build_core -j$JOBS
cmake --install ~/rtabmap/build_core
