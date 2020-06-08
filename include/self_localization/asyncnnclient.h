#ifndef ASYNC_NN_CLIENT
#define ASYNC_NN_CLIENT
#include <math.h>
#include <stdio.h>
#include <vector>
#include <cstdio>
#include <stdlib.h>
#include <cstring>
#include "Eigen/Dense"
#include <iostream>
#include <mutex>
#include <ndt_generic/pcl_utils.h>
// PCL specific includes
#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <ndt_map/ndt_map.h>
#include <ndt_map/ndt_cell.h>
#include <ndt_map/pointcloud_utils.h>
#include <tf_conversions/tf_eigen.h>
#include <cstdio>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <algorithm>

#include "ros/ros.h"
#include "nav_msgs/Odometry.h"
#include "ros/publisher.h"
#include "tf/transform_broadcaster.h"
#include "ndt_generic/eigen_utils.h"

#include "ndt_generic/io.h"

#include "ndt_generic/io.h"
#include "ndt_generic/sensors_utils.h"

#include "ndt_generic/pcl_utils.h"
#include "ndt_generic/pcl_utils_impl.h"
#include "self_localization/NNPose.h"
#include "eigen_conversions/eigen_msg.h"
#include <thread>
#include <tuple>
namespace AsyncNNClient{
typedef Eigen::Matrix<double,6,6> Matrix6d;
typedef std::pair<Eigen::Affine3d,Matrix6d> poseWithCovariance;


using std::cout;
using std::cerr;
using std::endl;
class AsyncNNClient
{
public:
  AsyncNNClient();

  bool GetLatest( Eigen::Affine3d &Tnn_est, Eigen::Matrix<double,6,6> &cov);

  bool AddLatest( pcl::PointCloud<pcl::PointXYZ> &cloud, const Eigen::Affine3d &Test);

  void Clear();

  void Activate(){gt_nn_estimats_ = true;}

  void Deactivate(){gt_nn_estimats_ = false;}


protected:

  void ClientThread( );

  void push_front(const poseWithCovariance &est);

  bool pop_front( poseWithCovariance &est);

  bool pop_back( poseWithCovariance &est);

  bool Last( poseWithCovariance &est);

  void InitPublishers();



private:


  std::mutex m;
  std::mutex m_clear_;
  bool gt_nn_estimats_ = false;
  ndt_generic::PointCloudQueue<pcl::PointXYZ> queue_;
  bool cloud_queue_updated_;
  Eigen::Affine3d Tlatest_scan_; //assiciated with last element of queue
  self_localization::NNPose srv_;
  std::list<poseWithCovariance> est_;
  std::thread *input_th_;
  ros::NodeHandle nh_;
  ros::ServiceClient client_;
  std::string topic_ = "/nn_localization";
  int max_buffer_size = 10;

  ros::Publisher pub;
  ros::Rate loop_rate_;

};
}


#endif
