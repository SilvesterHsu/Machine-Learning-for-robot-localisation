//perception_oru
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "graph_map/graph_map_navigator.h"
#include "graph_map/graphfactory.h"

#include "ndt_localization/particle_filter.hpp"
#include "ndt_map/ndt_map.h"
#include "ndt_map/ndt_conversions.h"
//pcl
#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
//ros
#include "geometry_msgs/PoseArray.h"
#include "laser_geometry/laser_geometry.h"
#include "sensor_msgs/LaserScan.h"
#include "sensor_msgs/PointCloud2.h"
#include "nav_msgs/Odometry.h"
#include "ros/ros.h"
#include "ros/rate.h"
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <velodyne_pointcloud/rawdata.h>
#include <velodyne_pointcloud/point_types.h>
#include "eigen_conversions/eigen_msg.h"
//std
#include <Eigen/Dense>
#include <string>
#include <map>
#include <fstream>
#include <iostream>
#include <chrono>
#include "graph_localization/mcl_ndt/mcl_ndt.h"
#include "graph_localization/localization_factory.h"
#include "ndt_generic/motionmodels.h"
#include "ndt_generic/motion_model_2d.h"
#include "graph_map/visualization/graph_visualization.h"
#include "ndt_generic/pcl_utils.h"
#include "std_srvs/TriggerRequest.h"
#include "std_srvs/Empty.h"
#include "self_localization/get_pose_at_time.h"
using namespace perception_oru;
using namespace graph_map;
using namespace graph_localization;
class LidarOdometryQueue {
  ros::NodeHandle nh;

  bool visualize;
  graphVisualization *vis_;

  //laser input
  std::string points_topic;
  ros::Publisher cloud_pub;
  ros::Subscriber PCSub;
  double min_range;
  double max_range;
  double keyframe_min_distance;

  Eigen::Affine3d Tsens;
  std::string rootTF, baseTF, mclTF, gtTF, laser_link, pose_est_link;
  std::string dataset;
  Eigen::Affine3d Tcurr;
  bool initialized, firstLoad;

  Eigen::Affine3d cloud_offset;
  ndt_generic::PointCloudQueue<pcl::PointXYZ> cloud_queue;


  void VisualizeAll(pcl::PointCloud<pcl::PointXYZ> &cloud_ts, ros::Time ts){

    cloud_ts.header.frame_id = laser_link;
    pcl_conversions::toPCL(ts, cloud_ts.header.stamp);
    cloud_pub.publish(cloud_ts);
  }

  void PCCallback(const sensor_msgs::PointCloud2::ConstPtr& msg){
    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    pcl::fromROSMsg (*msg, pcl_cloud);
    this->processFrame(pcl_cloud,msg->header.stamp);
  }
  bool LookUpTransform(const std::string &robot_tf,ros::Time ts, Eigen::Affine3d &T){
    static tf::TransformListener tf_listener;
    tf::StampedTransform transform;
    tf_listener.waitForTransform(rootTF, robot_tf, ts, ros::Duration(0.1));
    try{
      tf_listener.lookupTransform(rootTF, robot_tf, ts, transform);
    }
    catch(tf::TransformException ex){
      ROS_ERROR("%s", ex.what());
      return false;
    }
    tf::poseTFToEigen(transform, T);
    return true;
  }

  void processFrame(pcl::PointCloud<pcl::PointXYZ> &cloud, ros::Time ts){

    Eigen::Affine3d T;
    pcl::PointCloud<pcl::PointXYZ> cloud_sensorframe = cloud;

    if(!LookUpTransform(baseTF, ts, T))
      return;
    static Eigen::Affine3d Tprev=T;

    if( (T.translation()-Tprev.translation()).norm()<keyframe_min_distance )
      return;

    VisualizeAll(cloud_sensorframe, ts);
    cloud.clear();
    cloud_sensorframe.clear();
  }

public:
  LidarOdometryQueue(ros::NodeHandle param):initialized{false}, cloud_queue(10)
  {
    param.param<bool>("visualize", visualize, true);

    param.param<std::string>("points_topic", points_topic, "/velodyne_points");

    param.param<std::string>("root_tf", rootTF, "/world");
    param.param<std::string>("base_tf", baseTF, "/robot_odom_link");
    param.param<std::string>("gt_tf", gtTF, "/state_base_link");
    param.param("keyframe_min_distance", keyframe_min_distance, 1.0);

    Eigen::Vector3d sensor_offset_pos, sensor_offset_euler;
    param.param("sensor_pose_x",sensor_offset_pos(0),0.);
    param.param("sensor_pose_y",sensor_offset_pos(1),0.);
    param.param("sensor_pose_z",sensor_offset_pos(2),0.);
    param.param("sensor_pose_r",sensor_offset_euler(0),0.);
    param.param("sensor_pose_p",sensor_offset_euler(1),0.);
    param.param("sensor_pose_t",sensor_offset_euler(2),0.);
    Tsens = ndt_generic::vectorsToAffine3d(sensor_offset_pos, sensor_offset_euler);

    Eigen::Matrix<double, 3, 1> init_vec;
    param.param("init_x",init_vec(0),0.);
    param.param("init_y",init_vec(1),0.);
    param.param("init_z",init_vec(2),0.);
    param.param("init_ex",init_vec(0),0.);
    param.param("init_ey",init_vec(1),0.);
    param.param("init_ez",init_vec(2),0.);

    cloud_offset = ndt_generic::xyzrpyToAffine3d(init_vec(0),init_vec(1),init_vec(2),init_vec(3),init_vec(4),init_vec(5) );

    param.param("min_range", min_range, 1.5);
    param.param("max_range", max_range, 130.0);
    param.param<std::string>("dataset", dataset, "michigan");

    PCSub = nh.subscribe(points_topic, 1, &LidarOdometryQueue::PCCallback, this);
    cout<<"Listen to sensor_msgs/PointCloud2 at topic \""<<points_topic<<"\""<<endl;

    cloud_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZ>>("cloud_localized", 10);
    // *client = new ros::ServiceClient<std_srvs::TriggerRequest>("GetPose");
    ros::spin();
  }
};

int main(int argc, char **argv){
  ros::init(argc, argv, "NDTLidarOdometry");
  ros::NodeHandle parameters("~");
  LidarOdometryQueue pf(parameters);
}

