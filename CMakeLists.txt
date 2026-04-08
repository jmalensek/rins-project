cmake_minimum_required(VERSION 3.5)
project(rins_project)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

install(DIRECTORY
  launch
  config
  worlds
  maps
  urdf
  DESTINATION share/${PROJECT_NAME}/
)

install(PROGRAMS
  scripts/detect_people.py
  scripts/robot_commander.py
  scripts/robot_explorer.py
  scripts/detect_people2.py
  scripts/detect_rings27.py
  scripts/greet_people.py
  scripts/greet_people_faster.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()
