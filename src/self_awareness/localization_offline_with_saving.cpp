#include <ndt_generic/pcl_utils.h>
// PCL specific includes
#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <tf_conversions/tf_eigen.h>
#include <cstdio>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <algorithm>
#include <boost/program_options.hpp>
#include "ros/ros.h"
#include "nav_msgs/Odometry.h"
#include "ros/publisher.h"
#include "tf/transform_broadcaster.h"
#include "ndt_generic/eigen_utils.h"
//#include "ndt_mcl/3d_ndt_mcl.h"
#include "ndt_localization/3d_ndt_mcl.hpp"
#include "ndt_generic/io.h"

#include "ndt_offline/readbagfilegeneric.h"
#include <ndt_offline/VelodyneBagReader.h>
#include "ndt_generic/motionmodels.h"
#include "ndt_generic/io.h"
#include "ndt_generic/sensors_utils.h"
#include "ndt_offline/readpointcloud.h"
#include "ndt_generic/pcl_utils.h"
#include "ndt_generic/pcl_utils_impl.h"
#include "self_localization/NNPose.h"
#include "graph_map/NNDataSaver.h"

using namespace perception_oru;
namespace po = boost::program_options;
using namespace std;



std::string output_dir_name="";
std::string output_file_name="";
std::string base_name="";
std::string dataset="";

//map parameters
bool gt_localisation=false;
int itrs=0;
int attempts=1;
int nb_neighbours=0;
int nb_scan_msgs=0;
bool use_odometry_source=true;
bool use_gt_data=false;
bool visualize=true;
bool filter_fov=false;
bool filter_ring_nb=false;
bool step_control=false;
bool alive=false;
bool disable_reg=false, do_soft_constraints=false;
bool save_eval_results=false;
bool keyframe_update=false;
bool hold_start=false;
bool extrapolate_odometry=false;
string registration_type_name="ndt_d2d_reg";
perception_oru::MotionModel2d::Params motion_params;
std::string base_link_id="", gt_base_link_id="", tf_world_frame="", tf_fuser_frame="fuser";
std::string velodyne_config_file="";
std::string lidar_topic="";
std::string velodyne_frame_id="";
std::string tf_topic="";
std::string localisation_type="";
tf::Transform Tsensor_offset_tf;
Eigen::Affine3d Tsensor_offset,fuser_pose;//Mapping from base frame to sensor frame
ros::NodeHandle *n_=NULL;
ndt_generic::Vector6d init;
double sensor_time_offset=0;
double resolution_local_factor=0;
unsigned int n_particles=0;
int max_nb_iters = 0;
double resolution = 0;
double SIR_varP_threshold=0;
double range_filter_max_dist=0;
double max_range=0, min_range=0;
double maxRotationNorm_=0;
double interchange_radius_=0;
double maxTranslationNorm_=0;
double rotationRegistrationDelta_=0;
double sensorRange_=100;
double translationRegistrationDelta_=0;
double hori_min=0, hori_max=0;
double min_keyframe_dist=0, min_keyframe_rot_deg=0;
double z_filter_min_height=0;
double score_cell_weight=0;
double alpha_distance=0;
unsigned int skip_frame=20;
int attempt =1;
double ms_delay;
unsigned int nodeid=-1;
unsigned int n_obs_search=5;
double consistency_max_dist, consistency_max_rot;
double th_segment;
ros::Publisher *gt_pub,*fuser_pub,*cloud_pub,*odom_pub, *cloud_segmented_pub, *nn_est_pub;
ros::ServiceClient *client;
ros::ServiceClient *client2;
nav_msgs::Odometry gt_pose_msg,fuser_pose_msg,odom_pose_msg;
pcl::PointCloud<pcl::PointXYZ>::Ptr msg_cloud;
ReadBagFileGeneric<pcl::PointXYZ> *reader;
ros::Subscriber *manual_nn_sub, *kidnapped_robot_sub;
/// Set up the sensor link
tf::Transform sensor_link; ///Link from /odom_base_link -> velodyne
std::string bagfilename;
std::string reader_type="velodyne_reader";
bool use_pointtype_xyzir;
int min_nb_points_for_gaussian;
bool forceSIR;
bool keep_min_nb_points;
bool min_nb_points_set_uniform;
bool random_init_pose;
unsigned int bag_start_index =0;
ndt_generic::CreateEvalFiles *eval_files;
ndt_offline::OdometryType odom_type;
std::vector<double> initial_noise_vec;
bool use_nn_estimates;

ndt_generic::StepControl *step_controller;

Eigen::Affine3d Todom_base , Todom_base_prev, Todom_init; //Todom_base =current odometry pose, odom_pose=current aligned with gt, Todom_base_prev=previous pose, Todom_init= first odometry pose in dataset.
Eigen::Affine3d Tgt_base, Tgt_base_prev, Tinit,Tgt_t0;//Tgt_base=current GT pose,Tgt_base_prev=previous GT pose, Tgt_init=first gt pose in dataset;

template<class T> std::string toString (const T& x)
{
  std::ostringstream o;

  if (!(o << x))
    throw std::runtime_error ("::toString()");

  return o.str ();
}

void filter_ring_nb_fun(pcl::PointCloud<pcl::PointXYZ> &cloud, pcl::PointCloud<pcl::PointXYZ> &cloud_nofilter, const std::set<int>& rings) {
  std::cerr << "Can only filter_ring_nb if they are of type velodyne_pointcloud::PointXYZIR" << std::endl;
  cloud = cloud_nofilter;
}

void filter_ring_nb_fun(pcl::PointCloud<velodyne_pointcloud::PointXYZIR> &cloud,
                        pcl::PointCloud<velodyne_pointcloud::PointXYZIR> &cloud_nofilter,
                        const std::set<int>& rings) {
  for(int i=0; i<cloud_nofilter.points.size(); ++i) {
    if (rings.find((int)cloud_nofilter[i].ring) != rings.end()) {
      cloud.points.push_back(cloud_nofilter.points[i]);
    }
  }
  cloud.width = cloud.points.size();
  cloud.height = 1;
}


std::string transformToEvalString(const Eigen::Transform<double,3,Eigen::Affine,Eigen::ColMajor> &T) {
  std::ostringstream stream;
  stream << std::setprecision(std::numeric_limits<double>::digits10);
  Eigen::Quaternion<double> tmp(T.rotation());
  stream << T.translation().transpose() << " " << tmp.x() << " " << tmp.y() << " " << tmp.z() << " " << tmp.w() << std::endl;
  return stream.str();
}


void ReadAllParameters(po::options_description &desc,int &argc, char ***argv){

  Eigen::Vector3d transl;
  Eigen::Vector3d euler;
  // First of all, make sure to advertise all program options
  desc.add_options()
      ("help", "produce help message")
      ("reader-type", po::value<std::string>(&reader_type)->default_value(std::string("velodyne_reader")), "Type of reader to use when open rosbag e.g. velodyne_reader (config file needed) or pcl_reader when opening pcl2 messages")
      ("bag-file-path", po::value<string>(&bagfilename)->default_value(""), "File path to rosbag to play with maps")
      ("visualize", "visualize the rosbag and fuser estimate/gt")
      ("save-results", "save trajectory for gt, estimation, sensor and odometry")
      ("base-name", po::value<string>(&base_name)->default_value(std::string("generated_")), "prefix for all generated files")
      ("output-dir-name", po::value<string>(&output_dir_name)->default_value(""), "where to save the pieces of the map (default it ./map)")
      ("data-set", po::value<string>(&dataset)->default_value(""), "where to save the pieces of the map (default it ./map)")
      ("filter-fov", "cutoff part of the field of view")
      ("hori-max", po::value<double>(&hori_max)->default_value(2*M_PI), "the maximum field of view angle horizontal")
      ("hori-min", po::value<double>(&hori_min)->default_value(-hori_max), "the minimum field of view angle horizontal")
      ("min-range", po::value<double>(&min_range)->default_value(0.6), "minimum range used from scanner")
      ("max-range", po::value<double>(&max_range)->default_value(130), "minimum range used from scanner")
      ("filter-ring-nb", "if the number of rings should be reduced")
      ("keyframe-min-distance", po::value<double>(&min_keyframe_dist)->default_value(0.1), "minimum distance traveled before adding cloud")
      ("keyframe-min-rot-deg", po::value<double>(&min_keyframe_rot_deg)->default_value(1), "minimum rotation before adding cloud")
      ("tf-base-link", po::value<std::string>(&base_link_id)->default_value(std::string("")), "tf_base_link")
      ("tf-gt-link", po::value<std::string>(&gt_base_link_id)->default_value(std::string("")), "tf ground truth link")
      ("velodyne-config-file", po::value<std::string>(&velodyne_config_file)->default_value(std::string("../config/velo32.yaml")), "configuration file for the scanner")
      ("tf_world_frame", po::value<std::string>(&tf_world_frame)->default_value(std::string("/world")), "tf world frame")
      ("lidar-topic", po::value<std::string>(&lidar_topic)->default_value(std::string("/velodyne_packets")), "velodyne packets topic used")
      ("velodyne-frame-id", po::value<std::string>(&velodyne_frame_id)->default_value(std::string("/velodyne")), "frame_id of the velodyne")
      ("bag-start-index", po::value<unsigned int>(&bag_start_index)->default_value(0), "start with an offset in the bag file")
      ("tf-topic", po::value<std::string>(&tf_topic)->default_value(std::string("/tf")), "tf topic to listen to")
      ("x", po::value<double>(&transl[0])->default_value(0.), "sensor pose - translation vector x")
      ("y", po::value<double>(&transl[1])->default_value(0.), "sensor pose - translation vector y")
      ("z", po::value<double>(&transl[2])->default_value(0.), "sensor pose - translation vector z")
      ("ex", po::value<double>(&euler[0])->default_value(0.), "sensor pose - euler angle vector x")
      ("ey", po::value<double>(&euler[1])->default_value(0.), "sensor pose - euler angle vector y")
      ("ez", po::value<double>(&euler[2])->default_value(0.), "sensor pose - euler angle vector z")
      ("init-x", po::value<double>(&init[0])->default_value(0.0), "init-x")
      ("init-y", po::value<double>(&init[1])->default_value(0.0), "init-y")
      ("init-z", po::value<double>(&init[2])->default_value(0.0), "init-z")
      ("init-ex", po::value<double>(&init[3])->default_value(0.0), "init-ex")
      ("init-ey", po::value<double>(&init[4])->default_value(0.0), "init-ey")
      ("init-ez", po::value<double>(&init[5])->default_value(0.0), "init-ez")
      ("th-segment", po::value<double>(&th_segment)->default_value(2.5), "offset with respect to the fixed Lidar odometry Frame which will be used to segment ground")
      ("sensor_time_offset", po::value<double>(&sensor_time_offset)->default_value(0.), "timeoffset of the scanner data")
      ("step-control", "Step thorugh frames using keyboard input")
      ("gt-localization", "Localisation estimate is set to previous GT each frame, Tmotion is set to TgtMotion")
      ("no-odometry", "Do not make any odometry based predictions")
      ("disable-unwarp","do not unwarp the pointcloud using odometry");


  po::variables_map vm;
  po::store(po::parse_command_line(argc, *argv, desc), vm);
  po::notify(vm);
  use_gt_data=gt_base_link_id!="";

  srand(time(NULL));

  gt_localisation = vm.count("gt-localization");
  use_odometry_source=!vm.count("no-odometry");
  if(vm.count("disable-unwarp") )
    odom_type = ndt_offline::NO_ODOM;
  else
    odom_type = ndt_offline::WHEEL_ODOM;

  save_eval_results=vm.count("save-results");
  visualize = vm.count("visualize");
  step_control = vm.count("step-control");


  if (vm.count("help")){
    cout << desc << "\n";
    exit(0);
  }

  if( !ndt_generic::GetSensorPose(dataset,transl,euler,Tsensor_offset_tf,Tsensor_offset))
    cout << "no valid dataset specified, will use the provided sensor pose params" << endl;
  else
    cout<<"Sensor Pose: "<<Tsensor_offset.translation().transpose()<<endl;

  gt_pose_msg.header.frame_id = tf_world_frame;
  fuser_pose_msg.header.frame_id = tf_world_frame;
  odom_pose_msg.header.frame_id = tf_world_frame;
  step_controller = new ndt_generic::StepControl();

  return;
}
void initializeRosPublishers(){
  gt_pub = new ros::Publisher();
  odom_pub = new ros::Publisher();
  nn_est_pub= new ros::Publisher();
  fuser_pub = new ros::Publisher();
  cloud_pub = new ros::Publisher();
  client = new ros::ServiceClient();
  client2 = new ros::ServiceClient();
  cloud_segmented_pub = new ros::Publisher();
  manual_nn_sub = new ros::Subscriber();
  kidnapped_robot_sub = new ros::Subscriber();
  *gt_pub    = n_->advertise<nav_msgs::Odometry>("/GT", 50);
  *fuser_pub = n_->advertise<nav_msgs::Odometry>("/fuser", 50);
  *odom_pub = n_->advertise<nav_msgs::Odometry>("/odom", 50);
  *cloud_pub = n_->advertise<pcl::PointCloud<pcl::PointXYZ>>("/points_original", 1);
  *cloud_segmented_pub = n_->advertise<sensor_msgs::PointCloud2>("/points2", 1);

  *client2 = n_->serviceClient<graph_map::NNDataSaver>("/nn_data_saver");


  cout<<"initialized publishers"<<endl;
}
void printParameters(){
  cout<<"Output directory: "<<output_dir_name<<endl;


  if(reader_type.compare("velodyne_reader"));
  cout<<"Velodyne config path:"<<velodyne_config_file<<endl;

  cout<<"Bagfile: "<<bagfilename<<endl;
  cout<<"Lidar topic: "<<lidar_topic<<", lidar frame id: "<<velodyne_frame_id<<endl;
  cout<<"World frame: "<<tf_world_frame<<", tf topic"<<tf_topic<<endl;
}




void PlotAll(pcl::PointCloud<pcl::PointXYZ> &cloud, const ros::Duration &tupd, bool new_update, ros::Time tcloud, int counter){
  static tf::TransformBroadcaster br;
  ros::Time tplot =ros::Time::now();

  if (visualize ){
    tf::Transform tf_fuser;
    tf::transformEigenToTF(fuser_pose, tf_fuser);
    br.sendTransform(tf::StampedTransform(tf_fuser,tplot, tf_world_frame,  tf_fuser_frame));
    if (tf_world_frame != "/world") {
      tf::Transform tf_none;
      tf_none.setIdentity();
      br.sendTransform(tf::StampedTransform(tf_none, tplot, "/world", tf_world_frame));
    }
    br.sendTransform(tf::StampedTransform(Tsensor_offset_tf, tplot , tf_fuser_frame, velodyne_frame_id));
  }
  Eigen::Affine3d gt_pose = Tinit*Tgt_t0.inverse()*Tgt_base;
  if(visualize ){
    cloud.header.frame_id = tf_fuser_frame;
    pcl_conversions::toPCL(tplot, cloud.header.stamp);
    cloud_pub->publish(cloud);
    gt_pose_msg.header.stamp = tplot;
    fuser_pose_msg.header.stamp=gt_pose_msg.header.stamp;
    tf::poseEigenToMsg(gt_pose, gt_pose_msg.pose.pose);
    tf::poseEigenToMsg(fuser_pose, fuser_pose_msg.pose.pose);
    gt_pub->publish(gt_pose_msg);
    fuser_pub->publish(fuser_pose_msg);
  }
  if(new_update){
    eval_files->Write( tcloud,Tgt_base, fuser_pose, fuser_pose, fuser_pose*Tsensor_offset);
    if(step_control)
      step_controller->Step(counter);

    double  error = (gt_pose.translation().head(3)-fuser_pose.translation().head(3)).norm();
    cout<<"frame: "<<counter<<", t upd: "<<tupd<<", error="<<error<<", tplot: "<<ros::Time::now()-tplot<<endl;
  }
}
void SaveAll(){
  eval_files->Close();
}


void SegmentGroundAndPublishCloud(const pcl::PointCloud<pcl::PointXYZ> &cloud, const Eigen::Affine3d &pose_est, pcl::PointCloud<pcl::PointXYZ> &output) { //pose est is given in a fixed frame //cloud is given in sensor frame
  pcl::PointCloud<pcl::PointXYZ> cloud_transformed = cloud;
  Eigen::Affine3d tmp = pose_est;
  perception_oru::transformPointCloudInPlace(tmp, cloud_transformed);
  //static ndt_generic::PointCloudQueue<PointT> points_filtered(10);

  //th_segment //with respect to the robot position. 0 hight of odometry frame
  double wheel_radius=0.12;
  output.clear();
  for(int i=0;i<cloud.size();i++){
    if(cloud_transformed[i].z>pose_est.translation()(2)-wheel_radius+th_segment )
      output.push_back(cloud_transformed[i]);
  }
  output.header.stamp = cloud.header.stamp;
  output.header.frame_id = "/world";
  //cloud_segmented_pub->publish(output);
}



bool SaveData(ndt_generic::PointCloudQueue<pcl::PointXYZ> &queue, nav_msgs::Odometry &pose, Eigen::Affine3d &Tlatest_scan, Eigen::Affine3d &Tsensor_offset){

  pcl::PointCloud<pcl::PointXYZ> aggregated_scans;

  queue.GetCloud(aggregated_scans);

  Tlatest_scan = Tlatest_scan*Tsensor_offset.inverse();
  Eigen::Affine3d Tlocal_frame = Eigen::Affine3d::Identity();
  Tlocal_frame = Tlatest_scan.inverse(); // ndt_generic::xyzrpyToAffine3d(init[0],init[1],init[2],init[3],init[4],init[5]).inverse()*(Tlatest_scan.inverse());
  transformPointCloudInPlace(Tlocal_frame, aggregated_scans);
  toPCL(ros::Time::now(), aggregated_scans.header.stamp);
  aggregated_scans.header.frame_id = "/world";
  toPCL(ros::Time::now(), aggregated_scans.header.stamp);


  graph_map::NNDataSaver srv;
  cout<<"Request with cloud of size: "<<aggregated_scans.size()<<endl;
  pcl::toROSMsg(aggregated_scans, srv.request.vels);
  cloud_segmented_pub->publish(srv.request.vels);
  srv.request.pose = pose;

  if (client2->call(srv)){
    std::cout << srv.response.status << std::endl;;
    return true;
  }
  else{
    ROS_ERROR("Could not get a response");
    return false;
  }
}

void processData() {


 output_file_name = "OUTPUT.TXT";


  eval_files = new ndt_generic::CreateEvalFiles(output_dir_name, output_file_name, save_eval_results);
  int counter = 0;

  ndt_offline::readPointCloud reader(bagfilename, Tsensor_offset, odom_type , lidar_topic, min_range, max_range, velodyne_config_file, 0, tf_topic, tf_world_frame, gt_base_link_id);
  pcl::PointCloud<pcl::PointXYZ> cloud, filtered;

  bool initialized = false;
  ndt_generic::PointCloudQueue<pcl::PointXYZ> queue(10);

  while(reader.readNextMeasurement(cloud) && ros::ok()){
    ros::Time t1 = ros::Time::now();
    counter ++;
    if (cloud.size() == 0) continue; // Check that we have something to work with depending on the FOV filter here...
    Tgt_base = Todom_base = Eigen::Affine3d::Identity();
    bool odom_valid=false, gt_valid=false;

    odom_valid = reader.GetOdomPose(reader.GetTimeOfLastCloud(), base_link_id, Todom_base);
    gt_valid = reader.GetOdomPose(reader.GetTimeOfLastCloud(), gt_base_link_id, Tgt_base);

    if( !(!(use_gt_data && !gt_valid  ||  use_odometry_source && !odom_valid)) ){//everything is valid
      cout<<"skipping this frame"<<endl;
      continue;
    }

    if((!initialized)){
      Todom_init = Todom_base;
      Todom_base_prev = Todom_base;
      Tgt_base_prev = Tgt_base;
      Tinit = ndt_generic::xyzrpyToAffine3d(init[0],init[1],init[2],init[3],init[4],init[5])*Tgt_base;
      Tgt_t0 = Tgt_base;
      initialized = true;
      cout<<"initialized"<<endl;
      counter ++;
      continue;
    }
    Eigen::Affine3d Tgt_world = Tinit*Tgt_t0.inverse()*Tgt_base*Tsensor_offset;
    SegmentGroundAndPublishCloud(cloud, Tgt_world, filtered );
    queue.Push(filtered);

    bool status = SaveData(queue, gt_pose_msg, Tgt_world, Tsensor_offset);


  }

}

  /////////////////////////////////////////////////////////////////////////////////7
  /////////////////////////////////////////////////////////////////////////////////7
  /// *!!MAIN!!*
  /////////////////////////////////////////////////////////////////////////////////7
  /////////////////////////////////////////////////////////////////////////////////7
  ///


  int main(int argc, char **argv){



    ros::init(argc, argv, "generate_datga");
    po::options_description desc("Allowed options");
    n_=new ros::NodeHandle("~");
    ros::Time::init();
    initializeRosPublishers();
    ReadAllParameters(desc,argc,&argv);

    processData();
    cout<<"end of program"<<endl;

  }


