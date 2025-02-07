#The current odom of the target vehicle at the time this input was generated
nav_msgs/Odometry current_target_vehicle_odom
#The current acceleration estimate of the target vehicle at the time this input was generated
geometry_msgs/AccelWithCovarianceStamped current_target_vehicle_accel
#The pose of the reference frame the history and inner boundary are expressed in.
#It's position is the same as current_target_vehicle_pose, with an orientation that is tangent to the inner boundary
geometry_msgs/PoseStamped reference_frame
#3.0 second long history of positions, only X and Y are relevant,  but other fields may be included for debugging.
#Starts 3 seconds before current_pose.header.stamp and ends at current_pose.pose.position
sensor_msgs/PointCloud2 position_history
#400.0 meter long slice of the inner boundary starting at the closest point on the inner boundary to current_pose
sensor_msgs/PointCloud2 inner_boundary
#400.0 meter long slice of the outer boundary starting at the closest point on the outer boundary to current_pose
sensor_msgs/PointCloud2 outer_boundary