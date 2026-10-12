# Python binding of ORB-SLAM3 stereo for main.py, built against an ORB-SLAM3 tree with
# patches/orbslam3-thor.patch applied:
#   cmake -S recon/orbslam3_py -B recon/orbslam3_py/build -DORB_SLAM3_ROOT=$HOME/ORB_SLAM3 \
#         -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
#   cmake --build recon/orbslam3_py/build -j
cmake_minimum_required(VERSION 3.16)
project(orbslam3_py CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(ORB_SLAM3_ROOT $ENV{HOME}/ORB_SLAM3 CACHE PATH "ORB-SLAM3 source tree (built)")
set(Pangolin_DIR $ENV{HOME}/.local/pangolin/lib/cmake/Pangolin CACHE PATH "")

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenCV 4.4 REQUIRED PATHS /usr/lib/aarch64-linux-gnu/cmake/opencv4 /usr/lib/x86_64-linux-gnu/cmake/opencv4 NO_DEFAULT_PATH)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

pybind11_add_module(orbslam3_py orbslam3_py.cc)
target_compile_definitions(orbslam3_py PRIVATE COMPILEDWITHC11)
target_compile_options(orbslam3_py PRIVATE -march=native)
target_include_directories(orbslam3_py PRIVATE
  ${ORB_SLAM3_ROOT} ${ORB_SLAM3_ROOT}/include ${ORB_SLAM3_ROOT}/include/CameraModels
  ${ORB_SLAM3_ROOT}/Thirdparty/Sophus ${EIGEN3_INCLUDE_DIR} ${Pangolin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(orbslam3_py PRIVATE
  ${ORB_SLAM3_ROOT}/lib/libORB_SLAM3.so
  ${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib/libDBoW2.so
  ${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib/libg2o.so
  ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
set_target_properties(orbslam3_py PROPERTIES
  BUILD_RPATH "${ORB_SLAM3_ROOT}/lib;${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib;${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib;$ENV{HOME}/.local/pangolin/lib"
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
# DT_RPATH (not RUNPATH) so that libORB_SLAM3.so also finds Pangolin, DBoW2 and g2o
target_link_options(orbslam3_py PRIVATE -Wl,--disable-new-dtags)
