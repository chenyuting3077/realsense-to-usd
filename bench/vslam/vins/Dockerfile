# VINS-Fusion (stereo + IMU VIO, loop_fusion for loop closures) on ROS 1 Noetic, arm64 (Jetson Thor) or x86_64.
FROM ros:noetic-ros-base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-noetic-cv-bridge ros-noetic-image-transport ros-noetic-tf ros-noetic-rosbag ros-noetic-message-filters \
        libceres-dev libeigen3-dev libopencv-dev git python3-numpy python3-opencv python3-scipy build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /ws
RUN git clone --depth 1 https://github.com/HKUST-Aerial-Robotics/VINS-Fusion.git src/VINS-Fusion
# OpenCV 4: the old C constants are gone
RUN cd src/VINS-Fusion && grep -rlE "CV_[A-Z_]+" --include=*.cpp --include=*.cc --include=*.h . | xargs sed -i -E \
        -e 's/CV_FONT_HERSHEY_SIMPLEX/cv::FONT_HERSHEY_SIMPLEX/g; s/CV_LOAD_IMAGE_GRAYSCALE/cv::IMREAD_GRAYSCALE/g' \
        -e 's/CV_BGR2GRAY/cv::COLOR_BGR2GRAY/g; s/CV_RGB2GRAY/cv::COLOR_RGB2GRAY/g; s/CV_GRAY2RGB/cv::COLOR_GRAY2RGB/g' \
        -e 's/CV_GRAY2BGR/cv::COLOR_GRAY2BGR/g; s/\bCV_AA\b/cv::LINE_AA/g; s/CV_MINMAX/cv::NORM_MINMAX/g' \
        -e 's/CV_CALIB_CB_([A-Z_]+)/cv::CALIB_CB_\1/g; s/CV_TERMCRIT_EPS/cv::TermCriteria::EPS/g' \
        -e 's/CV_TERMCRIT_ITER/cv::TermCriteria::MAX_ITER/g' \
    && grep -rl "imgproc/imgproc.hpp\|opencv2/opencv.hpp" --include=*.cc --include=*.h camera_models | xargs sed -i \
        '0,/#include/s//#include <opencv2\/imgproc\/types_c.h>\n#include <opencv2\/imgproc\/imgproc_c.h>\n#include <opencv2\/calib3d\/calib3d_c.h>\n#include/'
RUN . /opt/ros/noetic/setup.sh && catkin_make -DCMAKE_BUILD_TYPE=Release -j4
