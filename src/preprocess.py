#!/usr/bin/python3

import os
import re
from collections import defaultdict, deque, namedtuple

import cv2
import matplotlib.pyplot as plt  # Ensure matplotlib is imported
import numpy as np
from tqdm import tqdm

from input import preprocess_input
from output import preprocess_output
from reader import RosBagReader
from utils import timestamp_to_sec

# Constants
MCAP_DIR = "/mnt/sdc-wdc/bags/uva_tum_8_4.mcap"
IMG_SAVE_PATH = "/mnt/sdc-wdc/radar_net_dataset/images"
LABEL_SAVE_PATH = "/mnt/sdc-wdc/radar_net_dataset/labelTxt"
WINDOW_DURATION = 0.2
OPPONENT_SYNC_THRESHOLD = 0.05

# don't want exist_ok to be True
os.mkdir(IMG_SAVE_PATH)
os.mkdir(LABEL_SAVE_PATH)

TOPIC_NAMES = [
    "/vehicle/uva_odometry",
    "/vehicle_3/odometry",
    "/radar_rear/ars548_process/detections",
    "/radar_front/ars548_process/detections",
]

RADAR_SENSOR_MAPPING = {
    "/radar_front/ars548_process/detections": "front",
    "/radar_rear/ars548_process/detections": "rear",
}

RadarData = namedtuple("RadarData", ["msg", "sensor_type"])


def plot_sample(radar_grid, bounding_box):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    radar_y, radar_x = list(zip(np.nonzero(np.any(radar_grid != 0, axis=2))))
    axs[0].scatter(radar_x, radar_y, s=10, alpha=0.5)
    axs[0].set_aspect("equal")  # Make radar plot square
    axs[0].set_title("Radar Grid")

    # axs[1].pcolor(bounding_box)
    axs[1].set_aspect("equal")  # Make instance plot square
    axs[1].set_title("Instance Grid")

    plt.tight_layout()
    plt.show()
    fig.savefig("radar_grid.png")


def main():
    # Initialize the RosBagReader with the specified topics
    reader = RosBagReader(MCAP_DIR, TOPIC_NAMES)

    # Accumulated radar detections
    radar_window = deque()
    radar_counts = defaultdict(int)  # Tracks count of each radar type in window
    opponent_odoms = defaultdict(deque)  # Maps vehicle_id to deque of odometries

    # Regex pattern to match opponent odometry topics
    vehicle_odom_pattern = re.compile(r"^/vehicle_(\d+)/odometry$")

    label_num = 0
    # Iterate through all messages in the ROS bag
    for topic, msg, timestamp in tqdm(reader.read_messages(), total=217741):
        if reader.msgs_read == 1:
            first_timestamp = timestamp * 1e-9

        timestamp_sec = timestamp * 1e-9  # Convert nanoseconds to seconds

        if "radar" in topic:
            # Handle radar detections
            sensor_location = RADAR_SENSOR_MAPPING.get(topic, "unknown")
            radar_data = RadarData(msg=msg, sensor_type=sensor_location)
            radar_window.append(radar_data)
            radar_counts[sensor_location] += 1

            # Remove outdated radar detections
            while (
                radar_window
                and timestamp_to_sec(radar_window[0].msg)
                < timestamp_sec - WINDOW_DURATION
            ):
                old_radar = radar_window.popleft()
                radar_counts[old_radar.sensor_type] -= 1

        elif vehicle_odom_pattern.match(topic):
            # Handle opponent odometry
            match = vehicle_odom_pattern.match(topic)
            vehicle_id = match.group(1)
            opponent_odoms[vehicle_id].append(msg)

            # Trim only the current vehicle's queue based on its own message timestamp
            odom_queue = opponent_odoms[vehicle_id]
            while (
                odom_queue
                and timestamp_to_sec(odom_queue[0])
                < timestamp_sec - OPPONENT_SYNC_THRESHOLD
            ):
                odom_queue.popleft()
            # Remove the vehicle entry if queue is empty
            if not odom_queue:
                del opponent_odoms[vehicle_id]

        elif "/vehicle/uva_odometry" in topic:
            ego_time = timestamp_sec

            while (
                radar_window
                and timestamp_to_sec(radar_window[0].msg) < ego_time - WINDOW_DURATION
            ):
                old_radar = radar_window.popleft()
                radar_counts[old_radar.sensor_type] -= 1

            # Check if we have both radar types
            if not (radar_counts["front"] > 0 and radar_counts["rear"] > 0):
                continue

            # TODO: don't add when ego odom too far from ego position
            synced_opponent_odoms = {}
            for vehicle_id, odom_queue in opponent_odoms.items():
                for odom_msg in reversed(odom_queue):
                    odom_time = timestamp_to_sec(odom_msg)
                    if abs(ego_time - odom_time) <= OPPONENT_SYNC_THRESHOLD:
                        synced_opponent_odoms[vehicle_id] = odom_msg
                        break  # Only the most recent is needed
                    elif odom_time < ego_time - OPPONENT_SYNC_THRESHOLD:
                        break  # Older messages are out of threshold

            if radar_window and synced_opponent_odoms:
                # (1,4,2) -> (8)
                bounding_box = preprocess_output(msg, synced_opponent_odoms)

                bb_distances = np.sqrt(np.sum(np.square(bounding_box), axis=-1))
                print(bb_distances)

                # if not empty -- change!
                if not np.sum(bounding_box):
                    continue

                bounding_box_str = ""
                for coordinate in bounding_box.ravel():
                    bounding_box_str += f"{coordinate} "
                bounding_box_str += "0 0"

                radar_grid = preprocess_input(msg, radar_window)
                # Convert to image format
                radar_grid = (radar_grid * 255.0).astype(np.uint8)
                cv2.imwrite(f"{IMG_SAVE_PATH}/{label_num}.png", radar_grid)
                # bounding box array to text
                with open(f"{LABEL_SAVE_PATH}/{label_num}.txt", "w") as f:
                    f.write(bounding_box_str)
                label_num += 1


if __name__ == "__main__":
    main()
