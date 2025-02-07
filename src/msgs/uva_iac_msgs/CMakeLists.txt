cmake_minimum_required(VERSION 3.5)
project(uva_iac_msgs)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(autoware_auto_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# declare the message files to generate code for
file(GLOB msg_files
  LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  "msg/*.msg"
  )
file(GLOB srv_files
  LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  "srv/*.srv"
  )
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES autoware_auto_msgs std_msgs 
)
ament_export_dependencies(autoware_auto_msgs)
ament_export_dependencies(nav_msgs)
ament_export_dependencies(geometry_msgs)
ament_export_dependencies(sensor_msgs)
ament_export_dependencies(rosidl_default_runtime)
ament_export_dependencies(std_msgs)

ament_package()
