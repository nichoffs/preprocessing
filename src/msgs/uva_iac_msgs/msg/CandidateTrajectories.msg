# The actual candidate trajectories derived from the raceline.
autoware_auto_msgs/Trajectory[] candidates
# How far off the raceline each candidate is. Positive means away from the outer wall. Negative means towards the outer wall.
float64[] offsets_from_raceline
# The index of the first collision for each candidate. -1 means no collision.
int32[] collision_indices