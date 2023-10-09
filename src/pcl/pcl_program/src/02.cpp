/**
 * savd pointcloud as pcl type
 */
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>                  //体素滤波器
#include <pcl/filters/statistical_outlier_removal.h> //半径滤波器
#include <pcl/segmentation/sac_segmentation.h>       //分割器
#include <pcl/filters/extract_indices.h>             //索引滤波器

#include <pcl/visualization/pcl_visualizer.h> //点云可视化
#include <pcl/visualization/cloud_viewer.h>

pcl::visualization::CloudViewer viewer("PCL Viewer");
pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2());
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered2(new pcl::PointCloud<pcl::PointXYZ>);
/*******************************************/

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr &input)
{

    if (input->data.empty())
    {
        printf("Received empty cloud skipping processing.\n");
        return;
    }

    pcl_conversions::toPCL(*input, *cloud); // 将ROS信息转换为PCL信息。
    pcl::fromPCLPointCloud2(*cloud, *cloud_xyz);

    if (cloud_xyz->empty())
    {
        printf("Received empty cloud skipping processing.\n");
        return;
    }

    // 体素滤波
    pcl::VoxelGrid<pcl::PointXYZ> vox;
    vox.setInputCloud(cloud_xyz);
    vox.setLeafSize(0.01, 0.01, 0.01);
    vox.filter(*cloud_filtered);

    // 邻近点个数滤波
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud_filtered);
    sor.setMeanK(5);             // 设置平均距离估计的最近距离数量K
    sor.setStddevMulThresh(1.0);  // 设置标准差异阀值系数
    sor.filter(*cloud_filtered2); // 执行过滤

    // 平面分割
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE); // 平面
    seg.setDistanceThreshold(0.01);
    seg.setInputCloud(cloud_filtered2);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices); // 点云索引
    seg.segment(*inliers, *coefficients);

    // 索引滤波
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud_filtered2);
    extract.setIndices(inliers);
    extract.setNegative(true);

    pcl::PointCloud<pcl::PointXYZ>::Ptr extracted_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    extract.filter(*extracted_cloud);

    // 邻近点个数滤波
    sor.setInputCloud(extracted_cloud);
    sor.setMeanK(10);             // 设置平均距离估计的最近距离数量K
    sor.setStddevMulThresh(1.0);  // 设置标准差异阀值系数
    sor.filter(*extracted_cloud); // 执行过滤

    viewer.showCloud(extracted_cloud);
}

int main(int argc, char **argv)
{

    ros::init(argc, argv, "pcl_tutorial_01");
    ros::NodeHandle nh_;
    pcl::PCDWriter writer;

    // 订阅深度点云信息。
    /// sunny_topic/tof_frame/pointcloud
    /// berxel_camera/depth/berxel_cloudpoint
    ros::Subscriber sub = nh_.subscribe("berxel_camera/depth/berxel_cloudpoint", 10, cloud_cb);

    while (ros::ok())
    {
        ros::spinOnce();
        if (viewer.wasStopped())
        {
            // writer.write("/home/jhr/saved_point1.pcd", *cloud_filtered2, false);
            printf("saved point\n");
            break;
        }
    }
}
