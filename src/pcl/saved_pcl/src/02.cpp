/**
 * savd pointcloud as pcl type
*/
#include<ros/ros.h>
#include<sensor_msgs/PointCloud2.h>
#include<pcl_conversions/pcl_conversions.h>
#include<pcl/point_types.h>

#include<pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>

ros::Publisher pub;
/*******************************************/

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& input)
{

    pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2());
    pcl::PCLPointCloud2::Ptr cloud_filtered(new pcl::PCLPointCloud2());
    
    

    pcl_conversions::toPCL(*input, *cloud);//将ROS信息转换为PCL信息。

    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;//体素滤波
    sor.setInputCloud(cloud);
    sor.setLeafSize(0.03, 0.03, 0.03);
    sor.filter(*cloud_filtered);

    pcl::visualization::PCLVisualizer viewer("3D viewer");
    viewer.setBackgroundColor(1, 1, 1);   
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> range_image_color_handler(cloud_filtered, 0, 200, 0); //设置自定义颜色
    viewer.addPointCloud(cloud_filtered, range_image_color_handler, "range image");
}

int main(int argc, char**argv)
{
    ros::init(argc, argv, "pcl_tutorial_01");
    ros::NodeHandle nh_;


    //订阅深度点云信息。
    ros::Subscriber sub = nh_.subscribe("camera/depth/color/points", 1, cloud_cb);

    // pub = nh_.advertise<sensor_msgs::PointCloud2>("output", 1);

    ros::spin();

}
