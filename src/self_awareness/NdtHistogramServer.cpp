#include <ros/ros.h>
#include "self_localization/NNPoseResponse.h"
#include "self_localization/NNPose.h"
#include "self_localization/NNPoseRequest.h"
#include "Eigen/Dense"
#include "tf_conversions/tf_eigen.h"
#include "eigen_conversions/eigen_msg.h"
#include "ros/ros.h"
#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include "pcl_conversions/pcl_conversions.h"
#include <pcl/point_types.h>
using std::cout;
using std::endl;

class HistServer
{
public:
    HistServer(ros::NodeHandle &nh/* Add parameters if needed */) {
        service = nh.advertiseService("NDTSimilarity", &HistServer::ServiceCallback, this);
        // hist = new NDtHistogramInterface();

        ros::spin();
    }
    void RosToPcl(const sensor_msgs::PointCloud2 &msg, pcl::PointCloud<pcl::PointXYZ> &cloud){
        pcl::PCLPointCloud2 pcl2;
        pcl_conversions::toPCL(msg, pcl2);  //make sure this is actually converting something
        pcl::fromPCLPointCloud2(pcl2,cloud);
    }
    bool ServiceCallback(self_localization::NNPoseRequest  &req,
                         self_localization::NNPoseResponse &res){
        if(req.vels.data.size()==0){
            ROS_INFO("Empty cloud received");
            return false;
        }
        pcl::PointCloud<pcl::PointXYZ> cloud;
        RosToPcl(req.vels, cloud);
        Eigen::Affine3d eig_pose; //Pose of most similar scan

        // <Insert a function which matches cloud to the database of histograms> E.g. score = hist.match(cloud, matching); cout<<"Most similar scan has a score of: "<<score<<endl; eig_pose = hist.getPose(timestamp);

        tf::poseEigenToMsg(eig_pose, res.pose.pose.pose);
        return true;
    }

    ros::NodeHandle nh;
    ros::ServiceServer service;
    //NDtHistogramInterface *hist;
};


int main(int argc, char **argv)
{
    ros::init(argc, argv, "NdtHistogramServer");
    ros::NodeHandle nh("~");
    HistServer h(nh);

}
