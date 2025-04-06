cmake_minimum_required(VERSION 3.0.2)
project(lidar_lidar_cal)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖包
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  pcl_ros
  pcl_conversions
)

# 查找 PCL 库
find_package(PCL REQUIRED)

# 包含头文件路径
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# 链接库路径
link_directories(${PCL_LIBRARY_DIRS})

# 消除 PCL 编译的定义冲突
add_definitions(${PCL_DEFINITIONS})

# 添加可执行文件
add_executable(calibration_node src/calibration_node.cpp)

# 强制设置输出路径到 ROS 的开发空间
set_target_properties(calibration_node PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/lib/${PROJECT_NAME}
)

# 链接库
target_link_libraries(calibration_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)