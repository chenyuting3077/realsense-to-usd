# OpenVINS (ov_msckf, ROS 2) + RTAB-Map (rtabmap_ros) on ROS 2 Jazzy, arm64 (Jetson Thor) or x86_64.
FROM ros:jazzy
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-rtabmap-ros ros-jazzy-cv-bridge ros-jazzy-image-transport ros-jazzy-rosbag2-storage-mcap \
        libeigen3-dev libboost-all-dev libceres-dev libopencv-dev git python3-pip python3-numpy python3-scipy python3-opencv \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /ws
RUN git clone --depth 1 https://github.com/rpng/open_vins.git src/open_vins
# Jazzy has only the .hpp headers
RUN grep -rl --include=*.h --include=*.cpp -E "image_transport/image_transport.h|tf2_geometry_msgs/tf2_geometry_msgs.h|cv_bridge/cv_bridge.h" src \
    | xargs sed -i -E 's#(image_transport/image_transport|tf2_geometry_msgs/tf2_geometry_msgs|cv_bridge/cv_bridge)\.h>#\1.hpp>#'
RUN . /opt/ros/jazzy/setup.sh && colcon build --packages-up-to ov_msckf \
        --cmake-args -DCMAKE_BUILD_TYPE=Release -DENABLE_ROS=ON --parallel-workers 4
