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
class localisation_node {
  ros::NodeHandle nh;
  //Map parameters
  std::string map_path;
  GraphMapNavigatorPtr graph_map_;
  LocalisationTypePtr localisation_type_ptr_;
  //MCL
  bool visualize;
  graphVisualization *vis_;


  //laser input
  std::string points_topic;   //std::string laserTopicName;

  ros::Publisher cloud_pub;
  ros::Publisher estPosePub;
  ros::Subscriber initPoseSub;
  ros::Subscriber PCSub;
  tf::Transform robot_tf,sensor_tf;

  Eigen::Affine3d Tsens;
  std::string rootTF, baseTF, mclTF, gtTF, laser_link, pose_est_link;

  string localisation_type_name="";
  string dataset="";

  Eigen::Affine3d Tprev;
  bool firstLoad;

  bool initialized, gt_initialize, map_localizaiton;

  double min_range;
  double max_range;
  Eigen::Affine3d pose_;
  Eigen::MatrixXd C_;
  tf::TransformBroadcaster trans_pub;
  Eigen::Affine3d pose_init_offset_;
  ndt_generic::PointCloudQueue<pcl::PointXYZ> cloud_queue;
  ros::ServiceClient *client;
  /*void Pose2DToTF(Eigen::Vector3d mean, ros::Time ts, Eigen::Affine3d Todometry){
    static tf::TransformBroadcaster br, br_mapOdom;
    tf::Transform transform;
    tf::Quaternion q;
    q.setRPY(0, 0, mean[2]);
    transform.setOrigin( tf::Vector3(mean[0], mean[1], 0.0) );
    transform.setRotation( q );
    br.sendTransform(tf::StampedTransform(transform, ts, rootTF, mclTF));
  }*/

  int LoadMap(){
    LoadGraphMap(map_path,graph_map_);
    if(graph_map_==NULL){
      std::cerr<<"ERROR LOADING NDT MAP FROM FILE"<<std::endl;
      exit(0);
    }
  }
  void Initialize(const Eigen::Affine3d &pose_init){
    geometry_msgs::Pose pose_init_geom;
    tf::poseEigenToMsg(pose_init, pose_init_geom);
    Initialize(pose_init_geom, ros::Time::now());
  }
  void initialposeCallback(geometry_msgs::PoseWithCovarianceStamped input_init){
    Initialize(input_init.pose.pose,input_init.header.stamp);
  }

  void Initialize(const geometry_msgs::Pose &pose_init,const ros::Time &t_init){

    Eigen::Affine3d pose_init_eig;
    tf::poseMsgToEigen(pose_init,pose_init_eig);
    cout<<"Initial position set to"<<pose_init_eig.translation().transpose()<<endl;
    Vector6d var;
    var<<0.5, 0.5, 0.0, 0.0, 0.0, 0.2;
    localisation_type_ptr_->InitializeLocalization(pose_init_eig, var);
    usleep(100);
    initialized=true;
  }
  void VisualizeAll(pcl::PointCloud<pcl::PointXYZ> &cloud_ts, ros::Time ts){

    nav_msgs::Odometry odom_pose_est;
    for(int i = 0 ; i<36 ; i++)
      odom_pose_est.pose.covariance[i] = C_.data()[i];

    odom_pose_est.header.frame_id="world";
    tf::poseEigenToMsg(pose_, odom_pose_est.pose.pose);
    odom_pose_est.header.stamp = ts;
    estPosePub.publish(odom_pose_est);
    if(visualize){
      tf::poseEigenToTF(pose_, robot_tf);
      trans_pub.sendTransform(tf::StampedTransform( robot_tf, ts, rootTF, pose_est_link) );
      tf::poseEigenToTF(Tsens, sensor_tf);
      trans_pub.sendTransform( tf::StampedTransform( sensor_tf, ts, pose_est_link, laser_link) );
    }
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
  void MapLocalization(pcl::PointCloud<pcl::PointXYZ> &cloud, Eigen::Affine3d &Todom_new, ros::Time ts){
    static unsigned int frames=0;
    static ros::Duration total = ros::Duration(0);

    static Eigen::Affine3d Tprev = Todom_new;
    Eigen::Affine3d Tmotion = Tprev.inverse() * Todom_new;
    Tprev = Todom_new;

    transformPointCloudInPlace(Tsens, cloud);
    localisation_type_ptr_->UpdateAndPredict(cloud, Tmotion, Tsens);
    frames++;
    pose_=localisation_type_ptr_->GetPose();
  }
  void DLLocalization(pcl::PointCloud<pcl::PointXYZ> &cloud, Eigen::Affine3d &Todom_new, ros::Time ts){
    static Eigen::Affine3d Tprev = Todom_new;
    Eigen::Affine3d Tmotion = Tprev.inverse() * Todom_new;
    Tprev = Todom_new;
    pose_=pose_*Tmotion;
    self_localization::get_pose_at_time srv;
    srv.request.stamp= ts;

    if(client->call(srv)){
      Eigen::Affine3d T_dl;
      tf::poseMsgToEigen(srv.response.pose.pose, T_dl);
      Eigen::MatrixXd C_dl(1, srv.response.pose.covariance.size());

      for( int i = 0 ; i<srv.response.pose.covariance.size() ; i++ )
        C_dl(1,i) = srv.response.pose.covariance[i];

      Eigen::Map<Eigen::MatrixXd> M2(C_dl.data(), 6, 6);
      /*C_dl.resize(srv.response.pose.covariance.size()/2,srv.response.pose.covariance.size()/2);*/
    }
  }
  void processFrame(pcl::PointCloud<pcl::PointXYZ> &cloud, ros::Time ts){

    Eigen::Affine3d T;
    pcl::PointCloud<pcl::PointXYZ> cloud_sensorframe = cloud;

    if(!LookUpTransform(baseTF,ts,T))
      return;

    Eigen::Affine3d Tinit;
    if( !initialized ){ //Initialize map or dl localization
      if(gt_initialize){
        if(!LookUpTransform(gtTF, ts, Tinit))
          return;
      }
      else
        Tinit=Eigen::Affine3d::Identity();
      Eigen::Affine3d pose_init_ = Tinit*pose_init_offset_;
      if(map_localizaiton){
        geometry_msgs::Pose pose_init_geom;
        tf::poseEigenToMsg(pose_init_, pose_init_geom);
        Initialize(pose_init_geom, ts);
      }
      initialized=true;
    }

    if(map_localizaiton)
      MapLocalization(cloud, T, ts);
    else
      DLLocalization(cloud, T, ts);

    VisualizeAll(cloud_sensorframe, ts);
    cloud.clear();
    cloud_sensorframe.clear();
  }

public:
  localisation_node(ros::NodeHandle param):initialized{false}, cloud_queue(10)
  {
    param.param<std::string>("map_file", map_path, "");
    param.param<bool>("visualize", visualize, true);
    param.param<bool>("gt_initialize", gt_initialize, true);
    param.param<bool>("map_localizaiton", map_localizaiton, true);



    param.param<std::string>("points_topic", points_topic, "/velodyne_points");

    param.param<std::string>("root_tf", rootTF, "/world");
    param.param<std::string>("base_tf", baseTF, "/robot_odom_link");
    param.param<std::string>("gt_tf", gtTF, "/state_base_link");
    param.param<std::string>("laser_tf", laser_link, "/velodyne");
    param.param<std::string>("pose_tf", pose_est_link, "/pose_est");

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

    pose_init_offset_ = ndt_generic::xyzrpyToAffine3d(init_vec(0),init_vec(1),init_vec(2),init_vec(3),init_vec(4),init_vec(5) );
    C_ = 1000*Eigen::MatrixXd::Identity(6,6);




    param.param("min_range", min_range, 1.5);
    param.param("max_range", max_range, 130.0);

    param.param<std::string>("dataset", dataset, "michigan");
    param.param<std::string>("localisation_type_name", localisation_type_name, "mcl_ndt");
    LoadMap();
    if(visualize)
      vis_=new graphVisualization(graph_map_, true, true, true);

    LocalisationParamPtr  loc_ptr=LocalisationFactory::CreateLocalisationParam(localisation_type_name);

    loc_ptr->GetParamFromRos();
    loc_ptr->sensor_pose=Tsens;
    loc_ptr->graph_map_=graph_map_;

    if(MCLNDTParamPtr parPtr=boost::dynamic_pointer_cast<MCLNDTParam>(loc_ptr )){
      cout<<"Read motion model for MCL"<<endl;
      GetMotionModel(dataset, parPtr->motion_model);
    }

    cout<<"----------------Localisation parameters------------------\n"<<loc_ptr->ToString()<<endl;
    localisation_type_ptr_=LocalisationFactory::CreateLocalisationType(loc_ptr);
    cout<<"---------------------------------------------------------"<<endl;

    PCSub = nh.subscribe(points_topic, 1, &localisation_node::PCCallback, this);
    cout<<"Listen to sensor_msgs/PointCloud2 at topic \""<<points_topic<<"\""<<endl;

    initPoseSub = nh.subscribe("/initialpose", 1000, &localisation_node::initialposeCallback, this);
    cout<<"Listan to initialization at \"/initialpose\""<<endl;

    estPosePub = nh.advertise<nav_msgs::Odometry>("pose_est", 20);
    cloud_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZ>>("cloud_localized", 10);
    // *client = new ros::ServiceClient<std_srvs::TriggerRequest>("GetPose");
    ros::spin();
  }
};

int main(int argc, char **argv){
  ros::init(argc, argv, "ndt_mcl");
  ros::NodeHandle parameters("~");
  localisation_node pf(parameters);
}

