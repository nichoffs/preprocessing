#!/usr/bin/python3

from argparse import ArgumentParser

import cv2
import numpy as np

GRID_SIZE = 800
METER_RANGE = 200
PIXEL_RESOLUTION = METER_RANGE / GRID_SIZE


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "-d", "--dataset-pth", default="/mnt/sdc-wdc/radar_net_dataset", required=False
    )
    parser.add_argument("-i", "--sample-idx", default=15000)

    args = parser.parse_args()

    radar_grid = cv2.imread(f"{args.dataset_pth}/images/{args.sample_idx}.png")

    print(radar_grid.shape)

    with open(f"{args.dataset_pth}/labels/{args.sample_idx}.txt") as f:
        for bounding_box_str in f:
            bounding_box = np.array(
                list(map(float, bounding_box_str.split()[1:]))
            ).reshape(-1, 4, 2)

            bounding_box = np.flip((bounding_box * GRID_SIZE).astype(np.int32), axis=-1)
            print(bounding_box)

            color = (0, 255, 0)

            # Line thickness of 8 px
            thickness = 1

            image = cv2.polylines(radar_grid, bounding_box, True, color, thickness)

            while 1:
                cv2.imshow("image", image)

                if cv2.waitKey(20) & 0xFF == 27:
                    break
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
