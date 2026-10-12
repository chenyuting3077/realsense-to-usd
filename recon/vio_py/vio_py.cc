// Python binding of OpenVINS (stereo + IMU VIO) and RTAB-Map's core (loop closures and pose graph on the VIO's
// odometry) for the tracker process. Both release the GIL while they work, so RTAB-Map can run in a thread of
// its own next to the per-frame OpenVINS calls.
#include <memory>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <opencv2/core.hpp>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/State.h"
#include "track/TrackBase.h"
#include "types/IMU.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

#include <rtabmap/core/Parameters.h>
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/Statistics.h>
#include <rtabmap/core/StereoCameraModel.h>
#include <rtabmap/utilite/ULogger.h>

namespace py = pybind11;
using U8 = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
using F64 = py::array_t<double, py::array::c_style | py::array::forcecast>;

static cv::Mat gray(const U8 &a) {
  if (a.ndim() != 2) throw std::runtime_error("expected a HxW uint8 image");
  return cv::Mat(a.shape(0), a.shape(1), CV_8UC1, const_cast<uint8_t *>(a.data())).clone();
}

static py::array_t<double> mat4(const Eigen::Matrix4d &T) {
  py::array_t<double> out({4, 4});
  auto r = out.mutable_unchecked<2>();
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) r(i, j) = T(i, j);
  return out;
}

// --- OpenVINS -------------------------------------------------------------------------------------------------

// VioManager keeps its feature tracker protected: a subclass reads the last tracked points for the window.
class Vio : public ov_msckf::VioManager {
public:
  using VioManager::VioManager;
  std::vector<cv::KeyPoint> keypoints(size_t cam) {
    auto obs = trackFEATS->get_last_obs();
    return obs.count(cam) ? obs[cam] : std::vector<cv::KeyPoint>();
  }
};

class OpenVins {
public:
  explicit OpenVins(const std::string &config) {
    auto parser = std::make_shared<ov_core::YamlParser>(config);
    std::string verbosity = "WARNING";
    parser->parse_config("verbosity", verbosity, false);
    ov_core::Printer::setPrintLevel(verbosity);
    ov_msckf::VioManagerOptions params;
    params.print_and_load(parser);
    params.use_multi_threading_subs = false;  // tracking runs in the caller's thread, frame by frame
    if (!parser->successful()) throw std::runtime_error("OpenVINS: bad config " + config);
    vio_ = std::make_shared<Vio>(params);
  }

  // rows: t (s), accel xyz (m/s^2), gyro xyz (rad/s)
  void imu(const F64 &rows) {
    if (rows.ndim() != 2 || rows.shape(1) != 7) throw std::runtime_error("imu: expected Nx7");
    auto r = rows.unchecked<2>();
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < r.shape(0); i++) {
      ov_core::ImuData m;
      m.timestamp = r(i, 0);
      m.am << r(i, 1), r(i, 2), r(i, 3);
      m.wm << r(i, 4), r(i, 5), r(i, 6);
      vio_->feed_measurement_imu(m);
    }
  }

  // The stereo pair at t (s), after the IMU up to t. Returns (initialized, world_from_imu 4x4 or None).
  py::tuple track(double t, const U8 &left, const U8 &right) {
    ov_core::CameraData c;
    c.timestamp = t;
    c.sensor_ids = {0, 1};
    c.images = {gray(left), gray(right)};
    c.masks = {cv::Mat::zeros(c.images[0].size(), CV_8UC1), cv::Mat::zeros(c.images[1].size(), CV_8UC1)};
    bool ok;
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    {
      py::gil_scoped_release release;
      vio_->feed_measurement_camera(c);
      ok = vio_->initialized();
      if (ok) {
        auto imu = vio_->get_state()->_imu;
        T.block<3, 3>(0, 0) = imu->Rot().transpose();  // JPL R_GtoI -> R_ItoG
        T.block<3, 1>(0, 3) = imu->pos();
      }
    }
    if (!ok) return py::make_tuple(false, py::none());
    return py::make_tuple(true, mat4(T));
  }

  // The left camera's tracked points of the last frame, Nx2.
  py::array_t<float> keypoints() {
    auto kp = vio_->keypoints(0);
    py::array_t<float> out({(py::ssize_t)kp.size(), (py::ssize_t)2});
    auto r = out.mutable_unchecked<2>();
    for (size_t i = 0; i < kp.size(); i++) {
      r(i, 0) = kp[i].pt.x;
      r(i, 1) = kp[i].pt.y;
    }
    return out;
  }

private:
  std::shared_ptr<Vio> vio_;
};

// --- RTAB-Map -------------------------------------------------------------------------------------------------

static rtabmap::Transform transform(const F64 &T) {
  if (T.ndim() != 2 || T.shape(0) != 4 || T.shape(1) != 4) throw std::runtime_error("expected a 4x4 pose");
  auto r = T.unchecked<2>();
  return rtabmap::Transform(r(0, 0), r(0, 1), r(0, 2), r(0, 3), r(1, 0), r(1, 1), r(1, 2), r(1, 3), r(2, 0),
                            r(2, 1), r(2, 2), r(2, 3));
}

static py::array_t<double> mat4(const rtabmap::Transform &T) {
  Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
  M.block<3, 4>(0, 0) = T.toEigen4d().block<3, 4>(0, 0);
  return mat4(M);
}

class RtabMap {
public:
  // params: RTAB-Map parameters as strings ({"Vis/MinInliers": "20", ...}); the rectified stereo pair's
  // intrinsics; nodes are the left camera (its optical frame is RTAB-Map's base frame).
  RtabMap(const std::map<std::string, std::string> &params, double fx, double fy, double cx, double cy,
          double baseline, int width, int height, const std::string &database)
      : model_(fx, fy, cx, cy, baseline, rtabmap::Transform::getIdentity(), cv::Size(width, height)) {
    ULogger::setType(ULogger::kTypeNoLog);
    rtabmap::ParametersMap p;
    for (const auto &kv : params) p.insert(rtabmap::ParametersPair(kv.first, kv.second));
    rtab_.init(p, database);
  }
  ~RtabMap() { rtab_.close(false); }

  // A keyframe: the stereo pair at t (s) with the odometry pose (world_from_left camera) and its covariance
  // (6 variances: x y z roll pitch yaw). Returns (node id or 0 if not added, loop closure id, proximity id).
  py::tuple add(double t, const U8 &left, const U8 &right, const F64 &odom, const std::vector<double> &var) {
    rtabmap::SensorData data(gray(left), gray(right), model_, 0, t);
    rtabmap::Transform pose = transform(odom);
    cv::Mat cov = cv::Mat::eye(6, 6, CV_64FC1);
    for (int i = 0; i < 6 && i < (int)var.size(); i++) cov.at<double>(i, i) = var[i];
    int id = 0, loop = 0, prox = 0;
    {
      py::gil_scoped_release release;
      if (rtab_.process(data, pose, cov)) {
        id = rtab_.getLastLocationId();
        loop = rtab_.getStatistics().loopClosureId();
        prox = rtab_.getStatistics().proximityDetectionId();
      }
    }
    return py::make_tuple(id, loop, prox);
  }

  // Every node's pose on the current graph: {id: world_from_left camera 4x4}.
  std::map<int, py::array_t<double>> poses() {
    std::map<int, py::array_t<double>> out;
    for (const auto &kv : rtab_.getLocalOptimizedPoses()) out[kv.first] = mat4(kv.second);
    return out;
  }

private:
  rtabmap::StereoCameraModel model_;
  rtabmap::Rtabmap rtab_;
};

PYBIND11_MODULE(vio_py, m) {
  py::class_<OpenVins>(m, "OpenVins")
      .def(py::init<const std::string &>(), py::arg("config"))
      .def("imu", &OpenVins::imu, py::arg("rows"))
      .def("track", &OpenVins::track, py::arg("t"), py::arg("left"), py::arg("right"))
      .def("keypoints", &OpenVins::keypoints);
  py::class_<RtabMap>(m, "RtabMap")
      .def(py::init<const std::map<std::string, std::string> &, double, double, double, double, double, int, int,
                    const std::string &>(),
           py::arg("params"), py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"), py::arg("baseline"),
           py::arg("width"), py::arg("height"), py::arg("database") = "")
      .def("add", &RtabMap::add, py::arg("t"), py::arg("left"), py::arg("right"), py::arg("odom"), py::arg("var"))
      .def("poses", &RtabMap::poses);
}
