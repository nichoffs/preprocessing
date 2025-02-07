# A single track published by the tracking algorithms

# Track properties
uint32 track_id # unique identifier for a vehicle, e.g. cav car might have track_id 9
float32 reputation

# Store track state
geometry_msgs/PoseWithCovariance pose
geometry_msgs/TwistWithCovariance twist
geometry_msgs/TwistWithCovariance accel
