#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <numeric>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <ctime>

class LiDARCalibration
{
public:
    LiDARCalibration()
    {
        // 订阅主 LiDAR 和次 LiDAR 点云
        sub_target_lidar_ = nh_.subscribe("/velodyne_points", 1, &LiDARCalibration::targetLidarCallback, this);
        sub_source_lidar_ = nh_.subscribe("/lidar_points", 1, &LiDARCalibration::sourceLidarCallback, this);

        // 发布主点云和变换后的次点云
        pub_master_points_ = nh_.advertise<sensor_msgs::PointCloud2>("/master_points", 1);
        pub_transformed_points_ = nh_.advertise<sensor_msgs::PointCloud2>("/transformed_lidar_points", 1);

        // 初始化初始外参
        initializeTransformation();

        ROS_INFO("LiDAR Calibration Node Initialized.");
        ROS_INFO("Press Enter to perform calibration...");
    }

    // 主 LiDAR 点云回调
    void targetLidarCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        pcl::fromROSMsg(*msg, *cloud_target_);
        ROS_INFO_ONCE("Received first target LiDAR (velodyne) point cloud with %lu points.", cloud_target_->size());

        // 实时发布主 LiDAR 点云
        sensor_msgs::PointCloud2 master_msg;
        pcl::toROSMsg(*cloud_target_, master_msg);
        master_msg.header.frame_id = "velodyne_frame";
        master_msg.header.stamp = ros::Time::now();
        pub_master_points_.publish(master_msg);
    }

    // 次 LiDAR 点云回调
    void sourceLidarCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        pcl::fromROSMsg(*msg, *cloud_source_);
        ROS_INFO_ONCE("Received first source LiDAR (lidar_points) point cloud with %lu points.", cloud_source_->size());

        // 实时变换次 LiDAR 点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::transformPointCloud(*cloud_source_, *transformed_cloud, transformation_);

        // 发布变换后的点云
        sensor_msgs::PointCloud2 transformed_msg;
        pcl::toROSMsg(*transformed_cloud, transformed_msg);
        transformed_msg.header.frame_id = "velodyne_frame"; // 变换后的点云在主 LiDAR 的坐标系下
        transformed_msg.header.stamp = ros::Time::now();
        pub_transformed_points_.publish(transformed_msg);
    }

    void performCalibration()
    {
        if (cloud_target_->empty() || cloud_source_->empty())
        {
            ROS_WARN("One or both point clouds are empty. Cannot perform calibration.");
            return;
        }

        // 点云预处理：降采样
        pcl::VoxelGrid<pcl::PointXYZ> voxel;
        voxel.setLeafSize(0.05, 0.05, 0.05);  // 更小的叶子大小（5cm）
        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        voxel.setInputCloud(cloud_source_);
        voxel.filter(*downsampled_cloud);

        // 点云预处理：剔除离群点
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(downsampled_cloud);
        sor.setMeanK(100);             // 增大 K 值
        sor.setStddevMulThresh(0.5);   // 减小阈值
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        sor.filter(*filtered_cloud);

        // 初始化 NDT
        pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
        ndt.setResolution(0.5);               // 减小分辨率
        ndt.setStepSize(0.1);                 // 保持步长
        ndt.setTransformationEpsilon(1e-6);   // 放宽收敛条件
        ndt.setMaximumIterations(500);        // 增加最大迭代次数

        // 使用初始外参作为初始猜测
        ndt.setInputSource(filtered_cloud);
        ndt.setInputTarget(cloud_target_);
        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        ndt.align(*aligned_cloud, transformation_);

        // 打印 NDT 状态
        ROS_INFO_STREAM("NDT Convergence Status: " << ndt.hasConverged());
        ROS_INFO_STREAM("NDT Final Score: " << ndt.getFitnessScore());

        if (!ndt.hasConverged())
        {
            ROS_WARN("NDT did not converge.");
            return;
        }

        // 计算残差数据
        std::vector<double> residuals = calculateResiduals(cloud_target_, aligned_cloud);

        // 统计残差信息
        double total_residual = std::accumulate(residuals.begin(), residuals.end(), 0.0);
        double mean_residual = total_residual / residuals.size();
        ROS_INFO("Mean Residual Error: %f meters", mean_residual);

        // 保存标定结果（每次覆盖 calibration_result.txt）
        saveTransformation(transformation_);

        // 保存残差数据（生成带时间戳的新文件）
        std::string residuals_file = generateTimestampedFilename("residuals", ".txt");
        saveResiduals(residuals, residuals_file);
    }

    void saveTransformation(const Eigen::Matrix4f &transformation)
    {
        // 获取包路径并设置 result 文件夹路径
        std::string package_path = ros::package::getPath("lidar_lidar_cal");
        std::string result_dir = package_path + "/result/";
        std::string file_path = result_dir + "calibration_result.txt";

        // 确保 result 目录存在
        struct stat info;
        if (stat(result_dir.c_str(), &info) != 0) {
            mkdir(result_dir.c_str(), 0775); // 创建 result 文件夹
        }

        // 打开文件
        std::ofstream outfile(file_path, std::ios::out);
        if (!outfile.is_open())
        {
            ROS_ERROR("Failed to open calibration_result.txt for writing. PATH: ");
            std::cout << file_path << std::endl;
            return;
        }

        // 分解变换矩阵
        Eigen::Matrix3f rotation_matrix = transformation.block<3, 3>(0, 0);
        Eigen::Vector3f translation = transformation.block<3, 1>(0, 3);
        Eigen::Vector3f euler_angles_rad = rotation_matrix.eulerAngles(2, 1, 0); // ZYX 顺序 (弧度)
        Eigen::Vector3f euler_angles_deg = euler_angles_rad * (180.0 / M_PI);    // 转为角度

        // 格式化输出
        outfile << std::fixed << std::setprecision(6);

        // 保存平移向量
        outfile << "Translation (X Y Z):\n";
        outfile << translation[0] << " " << translation[1] << " " << translation[2] << "\n";

        // 保存旋转角度（弧度）
        outfile << "\nRotation in radians (Z Y X):\n";
        outfile << euler_angles_rad[0] << " " << euler_angles_rad[1] << " " << euler_angles_rad[2] << "\n";

        // 保存旋转角度（角度）
        outfile << "\nRotation in angles (Z Y X):\n";
        outfile << euler_angles_deg[0] << " " << euler_angles_deg[1] << " " << euler_angles_deg[2] << "\n";

        // 保存 4x4 变换矩阵
        outfile << "\n4x4 Transformation Matrix:\n";
        outfile << transformation << "\n";

        // 保存 static_transform_publisher 命令
        outfile << "\nStatic Transform Publisher Command:\n";
        outfile << "rosrun tf static_transform_publisher "
                << translation[0] << " " << translation[1] << " " << translation[2] << " "
                << euler_angles_rad[2] << " " << euler_angles_rad[1] << " " << euler_angles_rad[0]
                << " /map /lidar_frame 10\n";

        // 关闭文件
        outfile.close();

        ROS_INFO("Calibration results saved to calibration_result.txt. PATH: ");
        std::cout << file_path << std::endl;
    }

    std::vector<double> calculateResiduals(const pcl::PointCloud<pcl::PointXYZ>::Ptr &target,
                                           const pcl::PointCloud<pcl::PointXYZ>::Ptr &aligned)
    {
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(target);

        std::vector<double> residuals; // 用于存储每个点的残差

        for (const auto &point : aligned->points)
        {
            std::vector<int> nearest_index(1);
            std::vector<float> nearest_distance(1);
            if (kdtree.nearestKSearch(point, 1, nearest_index, nearest_distance) > 0)
            {
                double error = std::sqrt(nearest_distance[0]); // 点到最近点的欧氏距离
                residuals.push_back(error);                   // 保存到残差列表
            }
        }

        return residuals;
    }

    void saveResiduals(const std::vector<double> &residuals, const std::string &file_name)
    {
        // 获取包路径并设置 result 文件夹路径
        std::string package_path = ros::package::getPath("lidar_lidar_cal");
        std::string result_dir = package_path + "/result/";
        std::string file_path = result_dir + file_name;

        // 确保 result 目录存在
        struct stat info;
        if (stat(result_dir.c_str(), &info) != 0) {
            mkdir(result_dir.c_str(), 0775); // 创建 result 文件夹
        }

        // 打开文件
        std::ofstream outfile(file_path, std::ios::out);
        if (!outfile.is_open())
        {
            ROS_ERROR("Failed to open residuals file for writing. PATH: ");
            std::cout << file_path << std::endl;
            return;
        }

        // 保存残差数据
        outfile << "Residuals (meters):\n";
        for (size_t i = 0; i < residuals.size(); ++i)
        {
            outfile << residuals[i] << "\n";
        }

        // 关闭文件
        outfile.close();

        ROS_INFO("Residuals saved to %s", file_path.c_str());
    }

    bool kbhit()
    {
        struct termios oldt, newt;
        int ch;
        int oldf;

        // 获取终端设置
        tcgetattr(STDIN_FILENO, &oldt);
        newt = oldt;
        newt.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);
        oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

        ch = getchar();

        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
        fcntl(STDIN_FILENO, F_SETFL, oldf);

        if (ch != EOF)
        {
            ungetc(ch, stdin);
            return true;
        }

        return false;
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_target_lidar_;  // 主 LiDAR 点云订阅器
    ros::Subscriber sub_source_lidar_; // 次 LiDAR 点云订阅器
    ros::Publisher pub_master_points_; // 主点云发布器
    ros::Publisher pub_transformed_points_; // 变换后的次点云发布器

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target_{new pcl::PointCloud<pcl::PointXYZ>()};
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_{new pcl::PointCloud<pcl::PointXYZ>()};

    Eigen::Matrix4f transformation_ = Eigen::Matrix4f::Identity(); // 初始外参矩阵

    // 生成带时间戳的文件名
    std::string generateTimestampedFilename(const std::string &base_name, const std::string &extension)
    {
        // 获取当前时间
        std::time_t now = std::time(nullptr);
        char buffer[80];
        std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", std::localtime(&now));

        // 拼接文件名
        return base_name + "_" + buffer + extension;
    }

    void initializeTransformation()
    {
        // 设置平移部分
        transformation_(0, 3) = 0.016920;  // X 平移 0.01
        transformation_(1, 3) = 0.761223;  // Y 平移 0.73
        transformation_(2, 3) = -0.195695;  // Z 平移 -0.2

        // 设置旋转部分（ZYX 欧拉角转旋转矩阵）
        double yaw = 3.131114;     // Yaw (绕 Z 轴) 3.13159
        double pitch = 0.000149;       // Pitch (绕 Y 轴) 0.0
        double roll = 0.377767;    // Roll (绕 X 轴) 0.37991

        Eigen::Matrix3f rotation_matrix;
        rotation_matrix = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) *   // Yaw (Z)
                        Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) * // Pitch (Y)
                        Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX());   // Roll (X)

        // 设置旋转矩阵到变换矩阵中
        transformation_.block<3, 3>(0, 0) = rotation_matrix;

        // 打印变换矩阵用于调试
        ROS_INFO_STREAM("Initialized Transformation Matrix:\n" << transformation_);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lidar_calibration");
    LiDARCalibration calibrator;

    ros::Rate rate(10); // 10 Hz
    while (ros::ok())
    {
        ros::spinOnce();
        if (calibrator.kbhit())
        {
            char c = getchar();
            if (c == '\n') // 按下回车键
            {
                calibrator.performCalibration();
            }
        }
        rate.sleep();
    }
    return 0;
}