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
#include <boost/program_options.hpp>
#include "graph_map/graph_map_fuser.h"
#include "graph_map/ndt/ndt_map_param.h"
#include "graph_map/ndt/ndtd2d_reg_type.h"
#include "graph_map/ndt/ndt_map_type.h"
#include "graph_map/ndt_dl/ndtdl_map_param.h"
#include "graph_map/ndt_dl/ndtdl_reg_type.h"
#include "graph_map/ndt_dl/ndtdl_map_type.h"
#include "graph_map/ndt_dl/point_curv3.h"
#include "ros/ros.h"
#include "nav_msgs/Odometry.h"
#include "ros/publisher.h"
#include "tf/transform_broadcaster.h"
#include "ndt_generic/eigen_utils.h"
//#include "ndt_mcl/3d_ndt_mcl.h"
#include "ndt_localization/3d_ndt_mcl.hpp"
#include "ndt_generic/io.h"
#include "graph_localization/localization_factory.h"
#include "graph_localization/localization_type.h"
#include "graph_localization/reg_localization_type/reg_localization_type.h"
#include "graph_localization/mcl_ndt/mcl_ndt.h"
#include "graph_localization/mcl_ndt/mcl_ndtdl.h"
#include "graph_localization/ukf_ndt/ukf_ndt.h"
#include "graph_localization/ukf_ndt/ukf_reg.h"
#include "graph_map/graph_map_navigator.h"
#include "graph_map/graphfactory.h"
#include "graph_map/reg_type.h"
#include "ndt_offline/readbagfilegeneric.h"
#include <ndt_offline/VelodyneBagReader.h>
#include "ndt_generic/motionmodels.h"
#include "ndt_generic/io.h"
#include "ndt_generic/sensors_utils.h"
#include "ndt_offline/readpointcloud.h"
#include "graph_map/visualization/graph_visualization.h"
#include "ndt_generic/pcl_utils.h"
#include "ndt_generic/pcl_utils_impl.h"
#include "self_localization/NNPose.h"
#include "graph_localization/mcl_ndt/submap_mcl.h"
#include "graph_localization/localization_factory.h"
#include "self_localization/asyncnnclient.h"
using namespace perception_oru;
using namespace graph_localization;
using namespace graph_map;
namespace po = boost::program_options;
using namespace std;


std::string map_dir_name="";
std::string output_dir_name="";
std::string output_file_name="";
std::string sequence_name = "";
std::string bag_file = "";
std::string map_file_directory="";
std::string base_name="";
std::string dataset="";
std::string map_switching_method="";
graphVisualization *vis;
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
bool uniform_initialization = false;
unsigned int max_frames = 50;
float convergance_rate =0.5;
string registration_type_name="ndt_d2d_reg";
perception_oru::MotionModel2d::Params motion_params;
std::string base_link_id="", gt_base_link_id="", tf_world_frame="", tf_fuser_frame="fuser";
std::string velodyne_config_file="";
std::string lidar_topic="";
std::string velodyne_frame_id="";
std::string map_file_path="";
std::string heatmap_path;
std::string tf_topic="";
std::string localisation_type="";
tf::Transform Tsensor_offset_tf;
Eigen::Affine3d Tsensor_offset,fuser_pose;//Mapping from base frame to sensor frame
ros::NodeHandle *n_=NULL;
MapParamPtr mapParPtr=NULL;
GraphMapParamPtr graphParPtr=NULL;
std::ofstream metadata_ofs;
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
double convergence_th = 0;
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
unsigned int bag_index_spacing = 1;
double ms_delay;
unsigned int nodeid=-1;
bool stop_at_convergence = false;
unsigned int n_obs_search=5;
std::vector<unsigned int> test_indecies;
double consistency_max_dist, consistency_max_rot;
double th_segment;
ros::Publisher *gt_pub,*fuser_pub,*cloud_pub,*odom_pub, *cloud_segmented_pub, *nn_est_pub;
ros::ServiceClient *client;
nav_msgs::Odometry gt_pose_msg,fuser_pose_msg,odom_pose_msg;
pcl::PointCloud<pcl::PointXYZ>::Ptr msg_cloud;
LocalisationTypePtr localisation_type_ptr;
LocalisationParamPtr localisation_param_ptr;
GraphMapNavigatorPtr graph_map_;
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
bool visualize_map;
bool random_init_pose;
unsigned int bag_start_index =0;
plotmarker marker_=plotmarker::point;
ndt_generic::CreateEvalFiles *eval_files;
ndt_offline::OdometryType odom_type;
Vector6d initial_noise;
bool use_ref_frame;
std::vector<double> initial_noise_vec;
bool use_nn_estimates;
bool fixed_covariance = false;
ndt_generic::StepControl *step_controller;
Eigen::Affine3d gt_pose;
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

bool LocateMapFilePath(const std::string &folder_name,std::vector<std::string> &scanfiles){
  DIR *dir;
  struct dirent *ent;
  if ((dir = opendir (folder_name.c_str())) != NULL) {
    while ((ent = readdir (dir)) != NULL) {
      if(ent->d_name[0] == '.') continue;
      char tmpcname[400];
      snprintf(tmpcname,399,"%s/%s",folder_name.c_str(),ent->d_name);
      std::string tmpfname = tmpcname;
      if(tmpfname.substr(tmpfname.find_last_of(".") + 1) == "MAP"|| tmpfname.substr(tmpfname.find_last_of(".") + 1) == "map") {
        scanfiles.push_back(tmpfname);
      }
    }
    closedir (dir);
  } else {
    std::cerr<<"Could not parse dir name\n";
    return false;
  }
  sort(scanfiles.begin(),scanfiles.end());
  {
    std::cout << "files to be loaded : " << std::endl;
    for (size_t i = 0; i < scanfiles.size(); i++) {
      std::cout << " " << scanfiles[i] << std::flush;
    }
    std::cout << std::endl;
  }
  return true;
}

void Initialize(const geometry_msgs::PoseWithCovarianceStamped& target_pose, bool keep_current_particles=true){
  Eigen::Affine3d Tpose;
  tf::poseMsgToEigen(target_pose.pose.pose, Tpose);
  Eigen::Affine3d Tclosest = (*graph_map_->begin())->GetObservationVector()[0];
  double min_distance = ndt_generic::GetDistance(Tpose, Tclosest);
  for(std::vector<MapNodePtr>::iterator itr_node = graph_map_->begin(); itr_node != graph_map_->end(); ++itr_node) { //loop thorugh all close nodes and find the observation closest to Tnow among all map nodes
    ndt_generic::Affine3dSTLVek obs_vec= (*itr_node)->GetObservationVector();
    Eigen::Affine3d tmp_closest = Tclosest;
    bool found = false;
    double tmp_distance = ndt_generic::SearchForClosestElement(Tpose, obs_vec,tmp_closest, found);
    if(tmp_distance < min_distance && found){
      min_distance = tmp_distance;
      Tclosest = tmp_closest;
    }

  }
  Tclosest = Tclosest*Tsensor_offset.inverse(); //remove sensor offset
  Tpose.translation()(2) = Tclosest.translation()(2); //pass z value of closest mapped pose

  if( SubmapMCLTypePtr mcl_ptr = boost::dynamic_pointer_cast<SubmapMCLType>(localisation_type_ptr) ){
    if(graph_map_!=NULL){
      graph_map_->m_graph.lock();
      if(keep_current_particles)
        mcl_ptr->DelayedInitialization(Tpose, initial_noise); //Add particles in a sound way to existing particles
      else
        mcl_ptr->InitializeLocalization(Tpose, initial_noise, false);

      graph_map_->m_graph.unlock();
    }
    else
      cerr<<"Cannot initialize pose before program has started properly"<<endl;
  }
}
//Acts like nn callback except that the initial_noise determines the distribution rather than the NN covariance
void ManualNNEstimateCallback (const geometry_msgs::PoseWithCovarianceStamped& target_pose){
  cout<<"Estimate Manually"<<endl;
  Initialize(target_pose, true);
}
//Callback similar to ManualNNEstimateCallback except that all precious particles are cleared upon callback
void KidnappedCallback (const geometry_msgs::PoseWithCovarianceStamped& target_pose){
  cout<<"Kidnapped robot"<<endl;
  Initialize(target_pose, false);
}

void ReadAllParameters(po::options_description &desc,int &argc, char ***argv){

  Eigen::Vector3d transl;
  Eigen::Vector3d euler;
  // First of all, make sure to advertise all program options
  desc.add_options()
      ("help", "produce help message")
      ("map-file-path", po::value<std::string>(&map_file_path)->default_value(std::string("no-path")), "file path to .MAP file containing graphMapNavigator")
      ("map-file-directory", po::value<std::string>(&map_file_directory)->default_value(std::string("")), "file path to a directory which contain a .map file")
      ("reader-type", po::value<std::string>(&reader_type)->default_value(std::string("velodyne_reader")), "Type of reader to use when open rosbag e.g. velodyne_reader (config file needed) or pcl_reader when opening pcl2 messages")
      ("bag-file-path", po::value<string>(&bagfilename)->default_value(""), "File path to rosbag to play with maps")
      ("attempts", po::value<int>(&attempts)->default_value(1), "Total retries of localisation, can be used to generate multiple files")
      ("visualize-map", "visualize  the map")
      ("visualize", "visualize the rosbag and fuser estimate/gt")
      ("save-results", "save trajectory for gt, estimation, sensor and odometry")
      ("base-name", po::value<string>(&base_name)->default_value(std::string("mcl")), "prefix for all generated files")
      ("output-dir-name", po::value<string>(&output_dir_name)->default_value(""), "where to save the pieces of the map (default it ./map)")
      ("data-set", po::value<string>(&dataset)->default_value(""), "where to save the pieces of the map (default it ./map)")
      ("localisation-algorithm-name", po::value<string>(&localisation_type)->default_value("reg_localisation_type"), "name of localisation algorihm e.g. mcl_ndt")
      ("registration-type-name", po::value<string>(&registration_type_name)->default_value("ndt_d2d_reg"), "name of registration method (only if registration is of type 'reg_localisation_type'")
      ("map-switching-method", po::value<std::string>(&map_switching_method)->default_value(std::string("")), "Type of reader to use when open rosbag e.g. velodyne_reader (config file needed) or pcl_reader when opening pcl2 messages")
      ("filter-fov", "cutoff part of the field of view")
      ("hori-max", po::value<double>(&hori_max)->default_value(2*M_PI), "the maximum field of view angle horizontal")
      ("hori-min", po::value<double>(&hori_min)->default_value(-hori_max), "the minimum field of view angle horizontal")
      ("filter-ring-nb", "if the number of rings should be reduced")
      ("z-filter-height", po::value<double>(&z_filter_min_height)->default_value(-100000000), "The minimum height of which ndtcells are used for localisation")
      ("score-cell-weight", po::value<double>(&score_cell_weight)->default_value(0.1), "The constant score added to the likelihood by hitting a cell with a gaussian.")
      ("Dd", po::value<double>(&motion_params.Dd)->default_value(1.), "forward uncertainty on distance traveled")
      ("Dt", po::value<double>(&motion_params.Dt)->default_value(1.), "forward uncertainty on rotation")
      ("Cd", po::value<double>(&motion_params.Cd)->default_value(1.), "side uncertainty on distance traveled")
      ("Ct", po::value<double>(&motion_params.Ct)->default_value(1.), "side uncertainty on rotation")
      ("Td", po::value<double>(&motion_params.Td)->default_value(1.), "rotation uncertainty on distance traveled")
      ("Tt", po::value<double>(&motion_params.Tt)->default_value(1.), "rotation uncertainty on rotation")
      ("keyframe-min-distance", po::value<double>(&min_keyframe_dist)->default_value(0.1), "minimum distance traveled before adding cloud")
      ("keyframe-min-rot-deg", po::value<double>(&min_keyframe_rot_deg)->default_value(1), "minimum rotation before adding cloud")
      ("consistency-max-dist", po::value<double>(&consistency_max_dist)->default_value(0.6), "minimum distance traveled before adding cloud")
      ("consistency-max-rot", po::value<double>(&consistency_max_rot)->default_value(M_PI/2.0), "minimum rotation before adding cloud")
      ("tf-base-link", po::value<std::string>(&base_link_id)->default_value(std::string("")), "tf_base_link")
      ("tf-gt-link", po::value<std::string>(&gt_base_link_id)->default_value(std::string("")), "tf ground truth link")
      ("velodyne-config-file", po::value<std::string>(&velodyne_config_file)->default_value(std::string("../config/velo32.yaml")), "configuration file for the scanner")
      ("sequence-name", po::value<std::string>(&sequence_name)->default_value(std::string("seq")), "sequence name, used to save output_file")
      ("bag-file", po::value<std::string>(&bag_file)->default_value(std::string("")), "bag name, used to save output_file only")
      ("tf_world_frame", po::value<std::string>(&tf_world_frame)->default_value(std::string("/world")), "tf world frame")
      ("lidar-topic", po::value<std::string>(&lidar_topic)->default_value(std::string("/velodyne_packets")), "velodyne packets topic used")
      ("velodyne-frame-id", po::value<std::string>(&velodyne_frame_id)->default_value(std::string("/velodyne")), "frame_id of the velodyne")
      ("min-range", po::value<double>(&min_range)->default_value(0.6), "minimum range used from scanner")
      ("max-range", po::value<double>(&max_range)->default_value(130), "minimum range used from scanner")
      ("skip-frame", po::value<unsigned int>(&skip_frame)->default_value(20), "sframes to skip before plot map etc.")
      ("max-frames", po::value<unsigned int>(&max_frames)->default_value(140), "Max frames per init attempt.")
      ("bag-start-index", po::value<unsigned int>(&bag_start_index)->default_value(0), "start with an offset in the bag file")
      ("test-index-spacing", po::value<unsigned int>(&bag_index_spacing)->default_value(1), "Amount of scans between each testing point")
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
      ("ms-delay", po::value<double>(&ms_delay)->default_value(0.0), "define the looprate, not acterting for")
      ("th-segment", po::value<double>(&th_segment)->default_value(2.5), "offset with respect to the fixed Lidar odometry Frame which will be used to segment ground")
      ("alpha-distance", po::value<double>(&alpha_distance)->default_value(0.), "sensor pose - euler angle vector z")
      ("sensor_time_offset", po::value<double>(&sensor_time_offset)->default_value(0.), "timeoffset of the scanner data")
      ("resolution-local-factors", po::value<std::vector<double> >()->multitoken(), "resolution factor of the local map used in the match and fusing step for multiple map representations")
      ("resolution-local-factor", po::value<double>(&resolution_local_factor)->default_value(1.), "resolution factor of the local map used in the match and fusing step")
      ("n-particles", po::value<unsigned int>(&n_particles)->default_value(270), "Total number of particles to use")
      ("convergance-rate", po::value<float>(&convergance_rate)->default_value(0.5), "speed of convergance during uniform initialization")
      ("uniform-particle-initialization", "uniform_initialization")
      ("SIR_varP_threshold", po::value<double>(&SIR_varP_threshold)->default_value(0.6), "resampling threshold")
      ("forceSIR", "force Sample Importance Reasampling")
      ("range_filter_max_dist", po::value<double>(&range_filter_max_dist)->default_value(1.), "max allowed range difference in UKF")
      ("min-observation-variance", po::value<std::vector<double> >()->multitoken(), "minimum observation variance in UKF reg")
      ("use_pointtype_xyzir", "If the points to be processed should contain ring and intensity information (velodyne_pointcloud::PointXYZIR)")
      ("min_nb_points_for_gaussian", po::value<int>(&min_nb_points_for_gaussian)->default_value(6), "minimum number of points per cell to compute a gaussian")
      ("n-obs-search", po::value<unsigned int>(&n_obs_search)->default_value(5), "minimum number of points per cell to compute a gaussian")
      ("max-nb-iters", po::value<int>(&max_nb_iters)->default_value(30), "max number of iteration in the registration")
      ("keep_min_nb_points", "If the number of points stored in a NDTCell should be cleared if the number is less than min_nb_points_for_gaussian")
      ("load-heatmap", "Load previous heatmap")
      ("save-heatmap", "Save current heatmap")
      ("heatmap-path", po::value<std::string>(&heatmap_path)->default_value(std::string("")), "path to heatmap, this reduces loading times")
      ("convergence-th", po::value<double>(&convergence_th)->default_value(0.3), "init-ey")
      ("exit-at-convergence", "Stop at convergence")
      ("min_nb_points_set_uniform", "If the number of points of one cell is less than min_nb_points_for_gaussian, set the distribution to a uniform one (cov = Identity)")
      ("step-control", "Step thorugh frames using keyboard input")
      ("gt-localization", "Localisation estimate is set to previous GT each frame, Tmotion is set to TgtMotion")
      ("key-frame-update", "localisation update based on keyframes, update triggeded upon moving a distance from previoussly updated pose")
      ("fixed-covariance", "use fixed covariance")
      ("hold-start", "localisation update based on keyframes, update triggeded upon moving a distance from previoussly updated pose")
      ("disable-localization", "No registration, resampling etc")
      ("no-odometry", "Do not make any odometry based predictions")
      ("localization3d", "use 3d localization type (not 2d assumption)")
      ("extrapolate-odometry", "Linear extrapolation of motion")
      ("multi-res","multiresaolution in registration")
      ("random-initial-pose","randomize initial pose - global localizaiton required")
      ("disable-unwarp","do not unwarp the pointcloud using odometry")
      ("disable-registration","Disable registration")
      ("check-consistency","concistency check of registration result")
      ("initial_noise", po::value<std::vector<double> >()->multitoken(), "initial noise in x,y,z, roll, pitch yaw")
      ("nn-estimates","use estimates obtained from a service")
      ("use-ref-frame", "if the submaps should be re-aligned with a reference frame (using frame data if available in the map node)");




  po::variables_map vm;
  po::store(po::parse_command_line(argc, *argv, desc), vm);
  po::notify(vm);
  use_gt_data=gt_base_link_id!="";
  keep_min_nb_points = vm.count("clear_min_nb_points");
  min_nb_points_set_uniform = vm.count("min_nb_points_set_uniform");
  NDTCell::setParameters(0.1, 8*M_PI/18., 1000, min_nb_points_for_gaussian, !keep_min_nb_points, min_nb_points_set_uniform);
  srand(time(NULL));
  stop_at_convergence = vm.count("exit-at-convergence");
  gt_localisation = vm.count("gt-localization");
  keyframe_update = vm.count("key-frame-update");
  disable_reg = vm.count("disable-registration");

  use_nn_estimates = vm.count("nn-estimates");
  uniform_initialization = vm.count("uniform-particle-initialization");
  use_odometry_source=!vm.count("no-odometry");
  fixed_covariance = vm.count("fixed-covariance");
  if(vm.count("disable-unwarp") )
    odom_type = ndt_offline::NO_ODOM;
  else
    odom_type = ndt_offline::WHEEL_ODOM;

  random_init_pose = vm.count("random-initial-pose");
  save_eval_results=vm.count("save-results");
  visualize = vm.count("visualize");
  visualize_map = vm.count("visualize-map");
  step_control = vm.count("step-control");

  //Check if all iputs are assigned
  if (!vm.count("map-dir-path") && !vm.count("map-file-path")){
    cout << "No .map file specified. Missing map-dir-path and map-file-path.\n";
    cout << desc << "\n";
    exit(0);
  }
  if (vm.count("help")){
    cout << desc << "\n";
    exit(0);
  }

  vector<double> resolution_local_factors;
  if (vm.count("resolution-local-factors"))
    resolution_local_factors = vm["resolution-local-factors"].as<vector<double> >();

  if (resolution_local_factors.size() != 3) {
    resolution_local_factors.clear();
    for (int i = 0; i < 3; i++) {
      resolution_local_factors.push_back(resolution_local_factor);
    }
  }
  vector<double> min_observation_variance;
  if (vm.count("min-observation-variance"))
  {
    min_observation_variance = vm["min-observation-variance"].as<vector<double> >();
    if (min_observation_variance.size() != 6) {
      min_observation_variance.clear();
      for (int i = 0; i < 6; i++) {
        min_observation_variance[i] = 0.1;
      }
    }
  }

  if (vm.count("initial_noise"))
  {
    initial_noise_vec = vm["initial_noise"].as<vector<double> >();
    if (initial_noise_vec.size() != 6) {
      initial_noise <<0.1,0.1,0.0,0.000,0.000,0.001;
    }
    else {
      initial_noise << initial_noise_vec[0], initial_noise_vec[1],initial_noise_vec[2],initial_noise_vec[3],initial_noise_vec[4],initial_noise_vec[5];
    }
  }

  if( !ndt_generic::GetSensorPose(dataset,transl,euler,Tsensor_offset_tf,Tsensor_offset))
    cout << "no valid dataset specified, will use the provided sensor pose params" << endl;
  else
    cout<<"Sensor Pose: "<<Tsensor_offset.translation().transpose()<<endl;

  localisation_param_ptr=LocalisationFactory::CreateLocalisationParam(localisation_type);
  localisation_param_ptr->visualize=visualize;
  localisation_param_ptr->sensor_pose=Tsensor_offset;
  localisation_param_ptr->switch_map_method=GraphMapNavigatorParam::String2SwitchMethod(map_switching_method);
  localisation_param_ptr->n_obs_search=n_obs_search;
  localisation_param_ptr->enable_localisation=!vm.count("disable-localization");
  cout<<"will use method: "<<map_switching_method<<endl;
  localisation_param_ptr->min_keyframe_dist = min_keyframe_dist;
  localisation_param_ptr->min_keyframe_dist_rot_deg = min_keyframe_rot_deg;
  if(MCLNDTParamPtr parPtr=boost::dynamic_pointer_cast<MCLNDTParam>(localisation_param_ptr )){
    parPtr->n_particles=n_particles;
    parPtr->z_filter_min=z_filter_min_height;
    parPtr->score_cell_weight=score_cell_weight;
    parPtr->SIR_varP_threshold=SIR_varP_threshold;
    parPtr->forceSIR=vm.count("forceSIR");
    GetMotionModel(dataset,parPtr->motion_model);
  }
  else if(SubmapMCLParamPtr parPtr = boost::dynamic_pointer_cast<SubmapMCLParam>(localisation_param_ptr )){
    parPtr->n_particles=n_particles;
    parPtr->z_filter_min=z_filter_min_height;
    parPtr->score_cell_weight=score_cell_weight;
    parPtr->SIR_varP_threshold=SIR_varP_threshold;
    parPtr->forceSIR=vm.count("forceSIR");

    parPtr->load_previous_heatmap = vm.count("load-heatmap");
    parPtr->save_heatmap = vm.count("save-heatmap");
    parPtr->heatmap_file_path = heatmap_path;
    parPtr->uniform_initialization = uniform_initialization;
    parPtr->convergance_rate = convergance_rate ;

    GetMotionModel(dataset,parPtr->motion_model);
  }
  else if (MCLNDTDLParamPtr parPtr=boost::dynamic_pointer_cast<MCLNDTDLParam>(localisation_param_ptr )){
    parPtr->n_particles=n_particles;
    parPtr->z_filter_min=z_filter_min_height;
    parPtr->score_cell_weight=score_cell_weight;
    parPtr->SIR_varP_threshold=SIR_varP_threshold;
    parPtr->forceSIR=vm.count("forceSIR");
    GetMotionModel(dataset,parPtr->motion_model);
  }
  else if (UKFNDTParamPtr parPtr=boost::dynamic_pointer_cast<UKFNDTParam>(localisation_param_ptr )) {
    parPtr->range_filter_max_dist=range_filter_max_dist;
    GetMotionModel(dataset,parPtr->motion_model);
    /* parPtr->motion_model.params.set3DParams(1, 1, 1,
                                            1, 1, 1,
                                            1, 1, 1,
                                            1, 1, 1);*/
  }
  else if(UKFRegParamPtr parPtr=boost::dynamic_pointer_cast<UKFRegParam>(localisation_param_ptr)){

    //    parPtr->keyframe_update=keyframe_update;
    //    parPtr->min_keyframe_dist=min_keyframe_dist;
    //    parPtr->min_keyframe_dist_rot_deg=min_keyframe_rot_deg;
    GetMotionModel(dataset,parPtr->motion_model);
    parPtr->min_obs_variance = min_observation_variance;


    if(graph_map::RegParamPtr reg_par_ptr=GraphFactory::CreateRegParam(registration_type_name)){
      // if(NDTD2DRegParamPtr ndt_reg_par=boost::dynamic_pointer_cast<NDTD2DRegParam>(reg_par_ptr)){
      reg_par_ptr->sensor_pose=Tsensor_offset;
      reg_par_ptr->sensor_range=max_range;
      reg_par_ptr->enable_registration=!disable_reg;
      reg_par_ptr->check_consistency=vm.count("check-consistency");
      reg_par_ptr->max_translation_norm=consistency_max_dist;
      reg_par_ptr->max_rotation_norm=consistency_max_rot;
      reg_par_ptr->registration2d=!vm.count("localization3d");
      parPtr->registration_parameters=reg_par_ptr;

      if(NDTD2DRegParamPtr ndt_par_ptr=boost::dynamic_pointer_cast<NDTD2DRegParam>(reg_par_ptr)){
        ndt_par_ptr->resolution_local_factor=resolution_local_factor;
        ndt_par_ptr->multires=vm.count("multi-res");
        ndt_par_ptr->matcher2D_ITR_MAX = max_nb_iters;
      }
      if(NDTDLRegParamPtr ndt_par_ptr=boost::dynamic_pointer_cast<NDTDLRegParam>(reg_par_ptr)){
        ndt_par_ptr->resolution_local_factors=resolution_local_factors;
        ndt_par_ptr->multires=vm.count("multi-res");
        ndt_par_ptr->matcher2D_ITR_MAX = max_nb_iters;
      }
      parPtr->registration_parameters=reg_par_ptr;
    }
  }
  else if(RegLocalisationParamPtr parPtr=boost::dynamic_pointer_cast<RegLocalisationParam>(localisation_param_ptr)){

    parPtr->keyframe_update=keyframe_update;
    parPtr->min_keyframe_dist=min_keyframe_dist;
    parPtr->min_keyframe_dist_rot_deg=min_keyframe_rot_deg;

    if(graph_map::RegParamPtr reg_par_ptr=GraphFactory::CreateRegParam(registration_type_name)){
      // if(NDTD2DRegParamPtr ndt_reg_par=boost::dynamic_pointer_cast<NDTD2DRegParam>(reg_par_ptr)){
      reg_par_ptr->sensor_pose=Tsensor_offset;
      reg_par_ptr->sensor_range=max_range;
      reg_par_ptr->enable_registration=!disable_reg;
      reg_par_ptr->check_consistency=vm.count("check-consistency");
      reg_par_ptr->max_translation_norm=consistency_max_dist;
      reg_par_ptr->max_rotation_norm=consistency_max_rot;
      reg_par_ptr->registration2d=!vm.count("localization3d");
      parPtr->registration_parameters=reg_par_ptr;

      if(NDTD2DRegParamPtr ndt_par_ptr=boost::dynamic_pointer_cast<NDTD2DRegParam>(reg_par_ptr)){
        ndt_par_ptr->resolution_local_factor=resolution_local_factor;
        ndt_par_ptr->multires=vm.count("multi-res");
      }
      if(NDTDLRegParamPtr ndt_par_ptr=boost::dynamic_pointer_cast<NDTDLRegParam>(reg_par_ptr)){
        ndt_par_ptr->resolution_local_factors=resolution_local_factors;
        ndt_par_ptr->multires=vm.count("multi-res");
        ndt_par_ptr->matcher2D_ITR_MAX = max_nb_iters;
      }
    }
  }
  else {
    cerr << "Unknown localization type : " << localisation_type << std::endl;
    exit(0);
  }
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
  cloud_segmented_pub = new ros::Publisher();
  manual_nn_sub = new ros::Subscriber();
  kidnapped_robot_sub = new ros::Subscriber();
  *gt_pub    = n_->advertise<nav_msgs::Odometry>("/GT", 50);
  *fuser_pub = n_->advertise<nav_msgs::Odometry>("/fuser", 50);
  *odom_pub = n_->advertise<nav_msgs::Odometry>("/odom", 50);
  *cloud_pub = n_->advertise<pcl::PointCloud<pcl::PointXYZ>>("/points_original", 1);
  *cloud_segmented_pub = n_->advertise<sensor_msgs::PointCloud2>("/points2", 1);
  *nn_est_pub = n_->advertise<nav_msgs::Odometry>("/nn_estimate", 50);
  *client = n_->serviceClient<self_localization::NNPose>("/nn_localization");
  *kidnapped_robot_sub = n_->subscribe("/manual_estimate", 10, &ManualNNEstimateCallback );
  *manual_nn_sub = n_->subscribe("/kidnapped_robot", 10,&KidnappedCallback );

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


bool UpdateAndPredict(pcl::PointCloud<pcl::PointXYZ> &cloud, const Eigen::Affine3d &Tmotion, const Eigen::Affine3d &Tsensor)
{
  return localisation_type_ptr->UpdateAndPredict(cloud, Tmotion, Tsensor);
}


bool UpdateAndPredict(pcl::PointCloud<velodyne_pointcloud::PointXYZIR> &cloud, const Eigen::Affine3d &Tmotion, const Eigen::Affine3d &Tsensor)
{
  std::vector<pcl::PointCloud<pcl::PointXYZ> > clouds;
  segmentPointCurvature3(Tsensor, cloud, clouds);
  return localisation_type_ptr->UpdateAndPredict(clouds, Tmotion, Tsensor);
}


void PlotAll(pcl::PointCloud<pcl::PointXYZ> &cloud, const ros::Duration &tupd, bool new_update, ros::Time tcloud, int counter){
  static tf::TransformBroadcaster br;
  ros::Time tplot =ros::Time::now();
  /*if(new_update && visualize && Node::DetectNewNode(nodeid, graph_map_->GetCurrentNode()))
    vis->ForcePlot();*/

  if (visualize ){
    tf::Transform tf_fuser;
    tf::transformEigenToTF(fuser_pose, tf_fuser);
    br.sendTransform(tf::StampedTransform(tf_fuser,tplot, tf_world_frame,  tf_fuser_frame));
    tf::Transform tf_gt;
    tf::transformEigenToTF(gt_pose, tf_gt);
    br.sendTransform(tf::StampedTransform(tf_gt,tplot, tf_world_frame,  gt_base_link_id));
    if (tf_world_frame != "/world") {
      tf::Transform tf_none;
      tf_none.setIdentity();
      br.sendTransform(tf::StampedTransform(tf_none, tplot, "/world", tf_world_frame));
    }
    br.sendTransform(tf::StampedTransform(Tsensor_offset_tf, tplot , tf_fuser_frame, velodyne_frame_id));
  }

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

    if(step_control)
      step_controller->Step(counter);
  }
}
void SaveAll(){
  eval_files->Close();
  metadata_ofs.close();
}


void SegmentGroundAndPublishCloud(const pcl::PointCloud<pcl::PointXYZ> &cloud, const Eigen::Affine3d &pose_est, pcl::PointCloud<pcl::PointXYZ> &output) { //pose est is given in a fixed frame //cloud is given in sensor frame
  pcl::PointCloud<pcl::PointXYZ> cloud_transformed = cloud;
  Eigen::Affine3d tmp = pose_est;
  Eigen::Affine3d robot_pose = pose_est*Tsensor_offset.inverse();
  perception_oru::transformPointCloudInPlace(tmp, cloud_transformed);
  //static ndt_generic::PointCloudQueue<PointT> points_filtered(10);

  //th_segment //with respect to the robot position. 0 hight of odometry frame
  double wheel_radius=0.12;
  output.clear();
  for(int i=0;i<cloud.size();i++){
    if(cloud_transformed[i].z>robot_pose.translation()(2)-wheel_radius+th_segment )
      output.push_back(cloud_transformed[i]);
  }
  output.header.stamp = cloud.header.stamp;
  output.header.frame_id = "/world";
  //cloud_segmented_pub->publish(output);
}
void PlotNNEstimate(const Eigen::Affine3d &Tpose, const graph_map::Matrix6d &cov){
  nav_msgs::Odometry nn_estimate;
  nn_estimate.header.frame_id = "/world";
  nn_estimate.header.stamp = ros::Time::now();
  tf::poseEigenToMsg(Tpose, nn_estimate.pose.pose);
  for(int i=0;i<36;i++)
    nn_estimate.pose.covariance[i]=cov.data()[i];
  nn_est_pub->publish(nn_estimate);

}

/*bool RequestPose(ndt_generic::PointCloudQueue<pcl::PointXYZ> &queue,  Eigen::Affine3d &Tlatest_scan, Eigen::Affine3d &Tnn_est, graph_map::Matrix6d &cov){

  pcl::PointCloud<pcl::PointXYZ> aggregated_scans;

  queue.GetCloud(aggregated_scans);
  Eigen::Affine3d Tlocal_frame = Eigen::Affine3d::Identity();
  Tlocal_frame = Tlatest_scan.inverse(); // ndt_generic::xyzrpyToAffine3d(init[0],init[1],init[2],init[3],init[4],init[5]).inverse()*(Tlatest_scan.inverse());
  transformPointCloudInPlace(Tlocal_frame, aggregated_scans);
  toPCL(ros::Time::now(), aggregated_scans.header.stamp);
  aggregated_scans.header.frame_id = "/world";
  toPCL(ros::Time::now(), aggregated_scans.header.stamp);


  self_localization::NNPose srv;
  cout<<"Request with cloud of size: "<<aggregated_scans.size()<<endl;
  pcl::toROSMsg(aggregated_scans,srv.request.vels );
  cloud_segmented_pub->publish(srv.request.vels);

  if (client->call(srv)){
    tf::poseMsgToEigen(srv.response.pose.pose.pose, Tnn_est);
    graph_map::Matrix6d m(srv.response.pose.pose.covariance.data());
    cov = m;
    return true;
  }
  else{
    ROS_ERROR("Could not get a response");
    return false;
  }
}*/
bool Initialize(){

  Todom_init = Todom_base;
  Todom_base_prev = Todom_base;
  Tgt_base_prev = Tgt_base;
  Tinit = ndt_generic::xyzrpyToAffine3d(init[0],init[1],init[2],init[3],init[4],init[5])*Tgt_base;
  //Tinit.translation() = Tinit.translation() + Eigen::Vector3d(0,0);
  Tgt_t0 = Tgt_base;
  graph_map_->m_graph.lock();
  Eigen::Affine3d Tinitialize = Tinit;
  if(random_init_pose){
    std::random_device rd{};
    std::mt19937 gen{rd()};

    // values near the mean are the most likely
    // standard deviation affects the dispersion of generated values from the mean
    std::normal_distribution<> d{0 ,100};
    Eigen::Affine3d random_init = Eigen::Affine3d::Identity();
    random_init.translation()<<-100+2*d(gen), 400+2*d(gen), 0;
    Tinitialize = random_init;
  }

  //Tinitialize.translation() = Tinitialize.translation() + Eigen::Vector3d(-3,-3,0);
  if(uniform_initialization){
    if( SubmapMCLTypePtr mcl_ptr = boost::dynamic_pointer_cast<SubmapMCLType>(localisation_type_ptr) ){
      Vector6d spread;
      spread    <<0.5, 0.5, 0.0, 0.00, 0.0, M_PI*2;
      Vector6d resolution;
      resolution<<0.5, 0.5, 0.2, 0.1 , 0.1, 2*M_PI/8;
      mcl_ptr->UniformInitialization(spread, resolution);
    }
  }
  else
    localisation_type_ptr->InitializeLocalization(Tinitialize, initial_noise);

  graph_map_->m_graph.unlock();
  cout<<"initialized"<<endl;
}
void processData() {

  //for(int i = bag_start_index ; i<10000*bag_index_spacing; i+=bag_index_spacing) //generate a vector of 100 elements
  //test_indecies.push_back(i);

  unsigned int current_test_index = bag_start_index;
  unsigned int init_attempt = 0;
  LoadGraphMap(map_file_path,graph_map_);
  localisation_param_ptr->graph_map_=graph_map_;
  localisation_type_ptr = LocalisationFactory::CreateLocalisationType(localisation_param_ptr);
  string init_type = uniform_initialization? "uniform":"DL-MCL";
  if(graph_map_==NULL ||localisation_type_ptr==NULL){
    cout<<"problem opening map"<<endl;
    exit(0);
  }
  vis = new graphVisualization(graph_map_, visualize, visualize_map, true);
  vis->SetParameters(10, 10);

  cout<<"-------------------------- Map and Localisation parameter ----------------------------"<<endl;
  cout<<localisation_type_ptr->ToString()<<endl;
  cout<<"--------------------------------------------------------"<<endl;

  int counter = 1;

  ndt_offline::readPointCloud reader(bagfilename, Tsensor_offset, odom_type , lidar_topic, min_range, max_range, velodyne_config_file, 0, tf_topic, tf_world_frame, gt_base_link_id);

  printParameters();
  pcl::PointCloud<pcl::PointXYZ> cloud, filtered;

  bool initialized = false;
  ros::Time read = ros::Time::now();
  ros::Rate r(ros::Duration(ms_delay/1000.0));
  reader.SkipToCloudIndex(current_test_index);

  //ndt_generic::PointCloudQueue<pcl::PointXYZ> queue(10);

  bool new_attempt = true;
  AsyncNNClient::AsyncNNClient nn_client;
  if(use_nn_estimates)
    nn_client.Activate();
  else if(uniform_initialization)
    nn_client.Deactivate();

  ros::Time t_start_init  = ros::Time::now();
  ros::Duration t_total(0);
  while(reader.readNextMeasurement(cloud) && ros::ok()){
    ros::Time t_start_itr = ros::Time::now();



    if (cloud.size() == 0) continue; // Check that we have something to work with depending on the FOV filter here...
    Tgt_base = Todom_base = Eigen::Affine3d::Identity();
    bool odom_valid=false, gt_valid=false;

    odom_valid = reader.GetOdomPose(reader.GetTimeOfLastCloud(), base_link_id, Todom_base);
    gt_valid = reader.GetOdomPose(reader.GetTimeOfLastCloud(), gt_base_link_id, Tgt_base);

    if( !(!(use_gt_data && !gt_valid  ||  use_odometry_source && !odom_valid)) ){//everything is valid
      cout<<"skipping this frame"<<endl;
      continue;
    }
    if(!initialized || new_attempt){
      if( SubmapMCLTypePtr mcl_ptr = boost::dynamic_pointer_cast<SubmapMCLType>(localisation_type_ptr) )
        mcl_ptr->Reset();

      t_start_init = ros::Time::now();
      t_total = ros::Duration(0);
      Initialize();
      counter = 1;
      nn_client.Clear();

      initialized = true;
      new_attempt = false;


      output_file_name = sequence_name+"_"+bag_file+"_init_type="+init_type+"_fixed="+toString(fixed_covariance)+"_attempt="+toString(init_attempt)+"_lresf="+toString(resolution_local_factor);
      metadata_ofs.open(  (output_dir_name +"/"+ output_file_name + std::string("_metadata.txt")).c_str());
      metadata_ofs<<"Frame_nr"<<" "<<"Error"<<" "<<"Tcomputation Elapsed Convergance_rate Status method x y z qx qy qz qw"<<endl;
      eval_files = new ndt_generic::CreateEvalFiles(output_dir_name, output_file_name, save_eval_results);

      if(uniform_initialization){
        if( SubmapMCLTypePtr mcl_ptr = boost::dynamic_pointer_cast<SubmapMCLType>(localisation_type_ptr) ){
          Vector6d spread;
          spread<<2.0, 2.0, 0.2, 0.0, 0.0, M_PI*2;
          //spread<<1.0, 1.0, 0.2, 0.0, 0.0, M_PI*2;
          Vector6d resolution;
          resolution<<2.0, 2.0, 0.2 , 0.1, 0.1, 2*M_PI/8;
          mcl_ptr->UniformInitialization(spread, resolution);
        }
      }
      continue;
    }

    Eigen::Affine3d Tgt_world = Tinit*Tgt_t0.inverse()*Tgt_base*Tsensor_offset;

    SegmentGroundAndPublishCloud(cloud, Tgt_world, filtered );

    //queue.Push(filtered);

    static bool new_update = false;
    if( new_update && !uniform_initialization && use_nn_estimates  ){
      if( SubmapMCLTypePtr mcl_ptr = boost::dynamic_pointer_cast<SubmapMCLType>(localisation_type_ptr) )  {
        Eigen::Affine3d nn_pose;
        graph_map::Matrix6d nn_cov;
        Tgt_world = Tgt_world*Tsensor_offset.inverse();
        nn_client.AddLatest(filtered, Tgt_world);
        if(nn_client.GetLatest(nn_pose, nn_cov)){

          Eigen::Matrix<double,6,1> vec;
          nn_cov.block<3,3>(3,3) = Eigen::Matrix3d::Identity();
          if(fixed_covariance){
            //double area = nn_cov(0,0)*nn_cov(1,1);
            vec<<70, 70, 3,  0.0225 , 0.25,  0.0225 ;
            //cout<<"Vec: "<<vec.transpose()<<endl;
          }
          else{
            Vector6d scale;
            scale<<1000, 1000, 1000,  0.0225 , 0.25, 0.225;
            nn_cov = nn_cov*scale.asDiagonal();
            vec = nn_cov.diagonal();
            //cout<<"Vec from nn: "<<vec.transpose()<<endl;
            //cout<<"nn cov: "<<nn_cov<<endl;
          }
          PlotNNEstimate(nn_pose, nn_cov);

          graph_map_->m_graph.lock();
          //Eigen::Affine3d Tcombined = ndt_generic::MuxAffineComponents(nn_pose, Eigen::Affine3d::Identity(),true,true,true,true,true,true);  //use pitch and roll from odometry
          //mcl_ptr->DelayedInitialization(cov_adjusted, vec, nn_pose);
          mcl_ptr->DelayedInitialization(nn_pose, vec);
          graph_map_->m_graph.unlock();
        }
      }
    }


    graph_map_->m_graph.lock();
    perception_oru::transformPointCloudInPlace(Tsensor_offset, cloud);
    Eigen::Affine3d Tmotion = Todom_base_prev.inverse()*Todom_base;
    new_update = UpdateAndPredict(cloud, Tmotion, Tsensor_offset);


    if(gt_localisation)
      fuser_pose = Tinit*Tgt_t0.inverse()*Tgt_base;
    else
      fuser_pose = localisation_type_ptr->GetPose();

    graph_map_->m_graph.unlock();
    gt_pose = Tinit*Tgt_t0.inverse()*Tgt_base;

    ros::Time t_end_itr = ros::Time::now();
    ros::Duration tframe(t_end_itr-t_start_itr);
    t_total +=tframe;

    double  error = (gt_pose.translation()-fuser_pose.translation()).norm();

    PlotAll(cloud, tframe, new_update, reader.GetTimeOfLastCloud(), counter);
    ros::Duration t_plot(ros::Time::now()-t_end_itr);

    if(new_update){
      int status = 0;
      cout<<"Attempt: "<<init_attempt<<"frame: "<<counter<<", T this frame: "<<tframe <<", Tplot: "<<t_plot<<", error="<<error<<endl;

      if( stop_at_convergence && (error<convergence_th ||counter == max_frames) ){

        if(counter == max_frames && error >= convergence_th){
          cout<<"Convergence not reached"<<endl;
          status = -1;
        }
        else{
          cout<<"Convergence reached"<<endl;
          status = 1;
        }
        metadata_ofs<<counter<<" "<<error<<" "<<tframe<<" "<<t_total<<" "<<convergance_rate<<" "<<status<<" "<<init_type<<" "<<ndt_generic::transformToEvalString(Tgt_world);
        eval_files->Write( reader.GetTimeOfLastCloud(), Tgt_base, fuser_pose, fuser_pose, fuser_pose*Tsensor_offset);
        SaveAll();
        current_test_index += bag_index_spacing;
        if(!reader.SkipToCloudIndex(current_test_index))
          exit(0);
        new_attempt = true;
        init_attempt++;
      }
      else{
        metadata_ofs<<counter<<" "<<error<<" "<<tframe<<" "<<t_total<<" "<<convergance_rate<<" "<<status<<" "<<init_type<<" "<<ndt_generic::transformToEvalString(Tgt_world);
        eval_files->Write( reader.GetTimeOfLastCloud(), Tgt_world, fuser_pose, fuser_pose, fuser_pose*Tsensor_offset);
      }
      counter++;
    }

    Tgt_base_prev = Tgt_base;
    Todom_base_prev = Todom_base;
    cloud.clear();
    read = ros::Time::now();

    ros::spinOnce();
  }
  SaveAll();

}


/////////////////////////////////////////////////////////////////////////////////7
/////////////////////////////////////////////////////////////////////////////////7
/// *!!MAIN!!*
/////////////////////////////////////////////////////////////////////////////////7
/////////////////////////////////////////////////////////////////////////////////7
///


int main(int argc, char **argv){



  ros::init(argc, argv, "convergence_test");
  po::options_description desc("Allowed options");
  n_=new ros::NodeHandle("~");
  ros::Time::init();
  initializeRosPublishers();
  ReadAllParameters(desc,argc,&argv);

  /*if (use_pointtype_xyzir) {
    processData<velodyne_pointcloud::PointXYZIR>();
  }*/

  processData();



  cout<<"end of program"<<endl;

}


