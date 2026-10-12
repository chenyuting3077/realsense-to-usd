# Python binding of OpenVINS and RTAB-Map's core for the tracker process (recon/live_workers.py), built against
# OpenVINS (no ROS) and Ceres in ~/.local/{open_vins,ceres}, RTAB-Map in ~/.local/rtabmap (build_deps.sh):
#   cmake -S recon/vio_py -B recon/vio_py/build -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
#   cmake --build recon/vio_py/build -j
cmake_minimum_required(VERSION 3.16)
project(vio_py C CXX)  # C: MPI::MPI_C (below)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(OPENVINS_ROOT $ENV{HOME}/.local/open_vins CACHE PATH "OpenVINS install prefix")
set(RTABMap_DIR $ENV{HOME}/.local/rtabmap/lib/rtabmap-0.23 CACHE PATH "")
set(Ceres_DIR $ENV{HOME}/.local/ceres/lib/cmake/Ceres CACHE PATH "")
set(Eigen3_DIR /usr/share/eigen3/cmake CACHE PATH "")

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenCV 4.4 REQUIRED PATHS /usr/lib/aarch64-linux-gnu/cmake/opencv4 /usr/lib/x86_64-linux-gnu/cmake/opencv4 NO_DEFAULT_PATH)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time)
find_package(Ceres REQUIRED)
find_package(MPI COMPONENTS C)  # PCL's VTK links MPI::MPI_C
find_package(RTABMap REQUIRED COMPONENTS core utilite)

pybind11_add_module(vio_py vio_py.cc)
target_compile_definitions(vio_py PRIVATE ROS_AVAILABLE=0)
target_include_directories(vio_py PRIVATE ${OPENVINS_ROOT}/include/open_vins ${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vio_py PRIVATE ${OPENVINS_ROOT}/lib/libov_msckf_lib.so ${RTABMap_LIBRARIES} ${OpenCV_LIBS}
                      ${Boost_LIBRARIES} Ceres::ceres)
set_target_properties(vio_py PROPERTIES
  BUILD_RPATH "${OPENVINS_ROOT}/lib;$ENV{HOME}/.local/rtabmap/lib;$ENV{HOME}/.local/ceres/lib"
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
target_link_options(vio_py PRIVATE -Wl,--disable-new-dtags)
