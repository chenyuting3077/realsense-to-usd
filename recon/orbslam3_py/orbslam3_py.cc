// Minimal Python binding of ORB-SLAM3 stereo for live use (recon/live.py).
//
//   slam = orbslam3_py.Stereo(vocabulary, settings_yaml, inertial=False)
//   ok, T_w_c, imu_only, imu_ready = slam.track(left, right, t_sec[, imu])
//                     # uint8 HxW rectified pair; T_w_c 4x4 or None; imu Nx7 (t_sec, ax, ay, az, gx, gy, gz):
//                     # the samples since the previous frame, in the IMU frame of the settings' IMU.T_b_c1;
//                     # imu_only: a pose carried on the IMU alone; imu_ready: gravity and biases estimated
//   slam.state()      # (big change count, maps, current map id): any change means keyframe poses
//                     # moved or a map started (loop closure / global BA, map merge, tracking lost)
//   slam.keypoints()  # Nx3 float32 (x, y, matched to a map point) of the last tracked left image
//   t, T_w_c, map_id = slam.trajectory()           # every tracked frame, from the current keyframe poses
//   slam.shutdown()
//
// track() and trajectory() must be called from the same thread: trajectory() walks the
// tracker's per-frame lists, which only track() appends to. Poses are in ORB-SLAM3's own
// world frame (the first keyframe at creation), which loop closure does not move, so live
// poses and corrected trajectories stay in one frame. Needs include/System.h from
// patches/orbslam3-thor.patch (GetTracker / GetAtlas).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <opencv2/core.hpp>

#include "Atlas.h"
#include "ImuTypes.h"
#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"
#include "System.h"
#include "Tracking.h"

namespace py = pybind11;

namespace {

py::array_t<double> to_numpy(const Sophus::SE3f& T) {
  py::array_t<double> out({4, 4});
  auto m = out.mutable_unchecked<2>();
  const Eigen::Matrix4f M = T.matrix();
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) m(r, c) = M(r, c);
  return out;
}

cv::Mat to_mat(const py::array_t<uint8_t, py::array::c_style>& a) {
  if (a.ndim() != 2) throw std::invalid_argument("expected a HxW uint8 image");
  // Copied: the tracker keeps references to the images it was given.
  return cv::Mat(a.shape(0), a.shape(1), CV_8UC1, const_cast<uint8_t*>(a.data())).clone();
}

class Stereo {
 public:
  Stereo(const std::string& vocabulary, const std::string& settings, bool inertial)
      : inertial_(inertial),
        slam_(vocabulary, settings, inertial ? ORB_SLAM3::System::IMU_STEREO : ORB_SLAM3::System::STEREO, false) {}

  py::tuple track(const py::array_t<uint8_t, py::array::c_style>& left,
                  const py::array_t<uint8_t, py::array::c_style>& right, double t,
                  const py::object& imu) {
    cv::Mat l = to_mat(left), r = to_mat(right);
    std::vector<ORB_SLAM3::IMU::Point> meas;
    if (!imu.is_none()) {
      auto a = imu.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
      if (a.ndim() != 2 || a.shape(1) != 7) throw std::invalid_argument("imu: expected Nx7 (t, ax, ay, az, gx, gy, gz)");
      auto m = a.unchecked<2>();
      for (py::ssize_t i = 0; i < a.shape(0); ++i)
        meas.emplace_back(m(i, 1), m(i, 2), m(i, 3), m(i, 4), m(i, 5), m(i, 6), m(i, 0));
    }
    Sophus::SE3f Tcw;
    int state;
    bool imu_ready;
    {
      py::gil_scoped_release release;
      Tcw = slam_.TrackStereo(l, r, t, meas);
      state = slam_.GetTrackingState();
      imu_ready = inertial_ && slam_.GetAtlas()->isImuInitialized();
    }
    // With an initialized IMU, RECENTLY_LOST still has a pose (predicted from the IMU, for a few
    // seconds): what bridges a blurred or textureless stretch.
    const bool imu_only = imu_ready && state == ORB_SLAM3::Tracking::RECENTLY_LOST;
    const bool ok = state == ORB_SLAM3::Tracking::OK || state == ORB_SLAM3::Tracking::OK_KLT || imu_only;
    return py::make_tuple(ok, ok ? py::object(to_numpy(Tcw.inverse())) : py::none(), imu_only, imu_ready);
  }

  // Same walk as System::SaveTrajectoryEuRoC: each frame is stored relative to its
  // reference keyframe; culled keyframes are replaced by their parent in the spanning tree.
  py::tuple trajectory() {
    ORB_SLAM3::Tracking* tr = slam_.GetTracker();
    std::vector<double> times;
    std::vector<Sophus::SE3f> poses;
    std::vector<long> maps;
    {
      py::gil_scoped_release release;
      auto lRit = tr->mlpReferences.begin();
      auto lT = tr->mlFrameTimes.begin();
      auto lbL = tr->mlbLost.begin();
      for (auto lit = tr->mlRelativeFramePoses.begin(); lit != tr->mlRelativeFramePoses.end();
           ++lit, ++lRit, ++lT, ++lbL) {
        if (*lbL) continue;
        ORB_SLAM3::KeyFrame* pKF = *lRit;
        if (!pKF) continue;
        Sophus::SE3f Trw;
        while (pKF && pKF->isBad()) {
          Trw = Trw * pKF->mTcp;
          pKF = pKF->GetParent();
        }
        if (!pKF) continue;
        Trw = Trw * pKF->GetPose();
        const Sophus::SE3f Tcw = (*lit) * Trw;
        times.push_back(*lT);
        poses.push_back(Tcw.inverse());
        maps.push_back(pKF->GetMap()->GetId());
      }
    }
    const py::ssize_t n = static_cast<py::ssize_t>(times.size());
    py::array_t<double> t_out(n), T_out({n, py::ssize_t(4), py::ssize_t(4)});
    py::array_t<long> m_out(n);
    auto t_m = t_out.mutable_unchecked<1>();
    auto T_m = T_out.mutable_unchecked<3>();
    auto m_m = m_out.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < n; ++i) {
      t_m(i) = times[i];
      m_m(i) = maps[i];
      const Eigen::Matrix4f M = poses[i].matrix();
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) T_m(i, r, c) = M(r, c);
    }
    return py::make_tuple(t_out, T_out, m_out);
  }

  py::array_t<float> keypoints() {
    const std::vector<cv::KeyPoint> kps = slam_.GetTrackedKeyPointsUn();
    const std::vector<ORB_SLAM3::MapPoint*> mps = slam_.GetTrackedMapPoints();
    const py::ssize_t n = static_cast<py::ssize_t>(kps.size());
    py::array_t<float> out({n, py::ssize_t(3)});
    auto o = out.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < n; ++i) {
      o(i, 0) = kps[i].pt.x;
      o(i, 1) = kps[i].pt.y;
      o(i, 2) = i < static_cast<py::ssize_t>(mps.size()) && mps[i] && !mps[i]->isBad();
    }
    return out;
  }

  py::tuple state() {
    ORB_SLAM3::Atlas* atlas = slam_.GetAtlas();
    ORB_SLAM3::Map* map = atlas->GetCurrentMap();
    return py::make_tuple(atlas->GetLastBigChangeIdx(), atlas->CountMaps(), map->GetId());
  }

  void shutdown() {
    py::gil_scoped_release release;
    slam_.Shutdown();
  }

 private:
  bool inertial_;
  ORB_SLAM3::System slam_;
};

}  // namespace

PYBIND11_MODULE(orbslam3_py, m) {
  py::class_<Stereo>(m, "Stereo")
      .def(py::init<const std::string&, const std::string&, bool>(), py::arg("vocabulary"), py::arg("settings"),
           py::arg("inertial") = false)
      .def("track", &Stereo::track, py::arg("left"), py::arg("right"), py::arg("t"), py::arg("imu") = py::none())
      .def("state", &Stereo::state)
      .def("trajectory", &Stereo::trajectory)
      .def("keypoints", &Stereo::keypoints)
      .def("shutdown", &Stereo::shutdown);
}
