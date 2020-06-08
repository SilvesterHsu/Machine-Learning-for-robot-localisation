#include "self_localization/asyncnnclient.h"

namespace AsyncNNClient{

AsyncNNClient::AsyncNNClient():nh_("~"),loop_rate_(100), queue_(10){
  InitPublishers();
  Activate();
  input_th_ = new std::thread(&AsyncNNClient::ClientThread, this);
}
void AsyncNNClient::InitPublishers(){
  client_ = nh_.serviceClient<self_localization::NNPose>(topic_);
  pub = nh_.advertise<sensor_msgs::PointCloud2>("/points2", 1);
}
void AsyncNNClient::push_front(const poseWithCovariance &est){
  if(est_.size() >= max_buffer_size){
    poseWithCovariance trash;
    pop_back(trash); //the least interesting element
  }
  est_.push_front(est);
}

bool AsyncNNClient::pop_front( poseWithCovariance &est){
  if(est_.size()>0){
    est = est_.front();
    est_.pop_front();
    return true;
  }
  else return false;
}
bool AsyncNNClient::pop_back( poseWithCovariance &est){
  if(est_.size()>0){
    est = est_.back();
    est_.pop_back();
    return true;
  }
  else return false;
}

bool AsyncNNClient::Last( poseWithCovariance &est){
  bool found = false;
  if(!est_.empty()){
    found = true;
    est = est_.front();
  }
  return found;
}

bool AsyncNNClient::GetLatest( Eigen::Affine3d &Tnn_est, Eigen::Matrix<double,6,6> &cov){

  poseWithCovariance latest_estimate;
  bool new_update = false;
  m.lock();
  new_update =pop_front(latest_estimate);
  m.unlock();
  if(new_update){
    Tnn_est = latest_estimate.first;
    cov = latest_estimate.second;
    return true;
  }
  else
    return false;

}
bool AsyncNNClient::AddLatest( pcl::PointCloud<pcl::PointXYZ> &cloud, const Eigen::Affine3d &Test){
  m.lock();
  Tlatest_scan_ = Test;
  queue_.Push(cloud);
  cloud_queue_updated_ = true;
  m.unlock();
}

void AsyncNNClient::Clear(){
  m_clear_.lock(); // major lock, wait for client to end
  queue_.Clear();
  Tlatest_scan_ = Eigen::Affine3d::Identity();
  std::list<poseWithCovariance> empty;
  std::swap( est_, empty );
  m_clear_.unlock();
}
void AsyncNNClient::ClientThread( ){

  while (ros::ok()) {
    ros::spinOnce();
    loop_rate_.sleep();
    if(!gt_nn_estimats_)
      continue;
    m_clear_.lock();
    m.lock(); //following commands are protected from add latest and Get latest
    pcl::PointCloud<pcl::PointXYZ> aggregated_scans;
    Eigen::Affine3d Tlocal_frame;
    if(!cloud_queue_updated_ ){
      m.unlock();
      m_clear_.unlock();
      continue;
    }
    else{
      queue_.GetCloud(aggregated_scans);
      Tlocal_frame = Tlatest_scan_.inverse();
      if(aggregated_scans.size()==0){
        m.unlock();
        m_clear_.unlock();
        continue;
      }
    }
    cloud_queue_updated_ = false;
    m.unlock();

    perception_oru::transformPointCloudInPlace(Tlocal_frame, aggregated_scans);
    pcl_conversions::toPCL(ros::Time::now(), aggregated_scans.header.stamp);
    aggregated_scans.header.frame_id = "/world";

    self_localization::NNPose srv;
    pcl::toROSMsg(aggregated_scans,srv.request.vels );
    pub.publish(srv.request.vels);
    Eigen::Affine3d Tnn_est;
    if (client_.call(srv)){
      tf::poseMsgToEigen(srv.response.pose.pose.pose, Tnn_est);
      Matrix6d Tnn_cov(srv.response.pose.pose.covariance.data());
      m.lock();
      //cout<<"Async nn client received pose"<<endl;
      push_front( std::make_pair(Tnn_est, Tnn_cov) );
      m.unlock();

    }
    m_clear_.unlock();
  }
}


}
