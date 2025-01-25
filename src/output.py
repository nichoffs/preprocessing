from math import atan2, cos, sin

import numpy as np

# Constants
GRID_SIZE = 200
METER_RANGE = 200  # Total range in meters (both X and Y axes)
LENGTH = 4.91  # Length of the vehicle in meters
WIDTH = 1.905  # Width of the vehicle in meters

# Vehicle corner coordinates (rectangle centered at origin)
VEHICLE_CORNERS = np.array(
    [
        [LENGTH / 2, WIDTH / 2],
        [LENGTH / 2, -WIDTH / 2],
        [-LENGTH / 2, -WIDTH / 2],
        [-LENGTH / 2, WIDTH / 2],
    ]
)


def calculate_yaw(qw, qx, qy, qz):
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy**2 + qz**2)
    yaw = atan2(siny_cosp, cosy_cosp)
    return yaw


def preprocess_output(ego_odom, opponent_odoms):
    ego_x = ego_odom.pose.pose.position.x
    ego_y = ego_odom.pose.pose.position.y
    ego_yaw = calculate_yaw(
        ego_odom.pose.pose.orientation.w,
        ego_odom.pose.pose.orientation.x,
        ego_odom.pose.pose.orientation.y,
        ego_odom.pose.pose.orientation.z,
    )

    transformed_corners = []
    for vehicle_id, odom in opponent_odoms.items():
        opp_yaw = calculate_yaw(
            odom.pose.pose.orientation.w,
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
        )
        rel_yaw = opp_yaw - ego_yaw

        dx_global = odom.pose.pose.position.x - ego_x
        dy_global = odom.pose.pose.position.y - ego_y

        # Transform to ego-relative coordinates
        dx_rel = dx_global * cos(ego_yaw) + dy_global * sin(ego_yaw)
        dy_rel = -dx_global * sin(ego_yaw) + dy_global * cos(ego_yaw)

        # Transform vehicle corners
        transform = np.array(
            [
                [cos(rel_yaw), -sin(rel_yaw), dx_rel],
                [sin(rel_yaw), cos(rel_yaw), dy_rel],
                [0, 0, 1],
            ]
        )

        corners_homogeneous = np.hstack(
            [VEHICLE_CORNERS, np.ones((len(VEHICLE_CORNERS), 1))]
        )
        transformed_corners.append((transform @ corners_homogeneous.T).T[:, :2])

    return np.stack(transformed_corners, axis=0)
