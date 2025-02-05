#!/usr/bin/env python3
"""
This script processes radar images and overlays bounding boxes from associated label files.
It accepts a dataset directory (which must contain images/ and labels/ subdirectories), an
image extension, and start/end indices. If start == end, a single image is processed and saved;
otherwise, a video is created from the sequence of processed images.

Each label file should contain one line per vehicle in the following format:
    class x1 y1 x2 y2 x3 y3 x4 y4
The first token (class) is ignored (all vehicles), and the following eight numbers represent
the four (x, y) pairs of the bounding box.
"""

import argparse
import os

import cv2
import numpy as np

# Global constants
GRID_SIZE = 800  # Size of the radar grid in pixels
METER_RANGE = 200  # Range in meters
PIXEL_RESOLUTION = METER_RANGE / GRID_SIZE  # Meters per pixel


def draw_bounding_boxes(image, label_path):
    """
    Reads all bounding boxes from the given label file and overlays them on the image.

    Each line in the label file is expected to have 9 entries:
      class x1 y1 x2 y2 x3 y3 x4 y4

    The first entry is skipped.
    """
    if not os.path.exists(label_path):
        print(f"Warning: Label file not found: {label_path}")
        return image

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue  # Not enough data for a bounding box, skip

            # Skip the first element (class) and parse the rest as floats.
            try:
                coords = list(map(float, parts[1:9]))
            except ValueError:
                print(f"Error parsing line in {label_path}: {line}")
                continue

            # Reshape the coordinates into a (4,2) array.
            try:
                bbox = np.array(coords).reshape(4, 2)
            except ValueError:
                print(f"Error reshaping coordinates from {label_path}: {coords}")
                continue

            # Scale the coordinates to the GRID_SIZE (i.e., convert from normalized or relative
            # coordinates to pixel coordinates, if necessary).
            bbox = (bbox * GRID_SIZE).astype(np.int32)

            # Draw the bounding box. Color is green (BGR: (0, 255, 0)) with a thickness of 1.
            image = cv2.polylines(
                image, [bbox], isClosed=True, color=(0, 255, 0), thickness=1
            )

    return image


def process_single_image(dataset_path, idx, ext):
    """
    Process a single image (with index idx): overlay all bounding boxes and save the result.
    """
    image_path = os.path.join(dataset_path, "images", f"{idx}.{ext}")
    label_path = os.path.join(dataset_path, "labels", f"{idx}.txt")

    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image file: {image_path}")
        return

    processed_image = draw_bounding_boxes(image, label_path)

    # Save the processed image
    output_image_path = os.path.join(dataset_path, f"processed_{idx}.{ext}")
    cv2.imwrite(output_image_path, processed_image)
    print(
        f"Processed single image with index {idx}. Saved output to: {output_image_path}"
    )

    # Optionally display the image. Press any key to close.
    cv2.imshow("Processed Image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(dataset_path, start_idx, end_idx, ext, output_video):
    """
    Process a range of images, overlay bounding boxes, and write the sequence to a video.
    """
    indices = list(range(start_idx, end_idx + 1))
    if not indices:
        print("No indices to process.")
        return

    # Read the first image to determine video properties.
    first_image_path = os.path.join(dataset_path, "images", f"{indices[0]}.{ext}")
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Error: Could not read image file: {first_image_path}")
        return

    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for mp4
    out = cv2.VideoWriter(output_video, fourcc, 20.0, (width, height))

    print(
        f"Creating video from index {start_idx} to {end_idx} and saving to: {output_video}"
    )

    for idx in indices:
        image_path = os.path.join(dataset_path, "images", f"{idx}.{ext}")
        label_path = os.path.join(dataset_path, "labels", f"{idx}.txt")

        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image file: {image_path}. Skipping.")
            continue

        processed_frame = draw_bounding_boxes(image, label_path)

        # Write the processed frame to video.
        out.write(processed_frame)

        # Optionally display the frame.
        cv2.imshow("Video Frame", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Video processing interrupted by user.")
            break

    out.release()
    cv2.destroyAllWindows()
    print(f"Finished video processing. Output video saved to: {output_video}")


def main():
    parser = argparse.ArgumentParser(
        description="Overlay bounding boxes on radar images and save as an image or video."
    )

    parser.add_argument(
        "-p",
        "--dataset-path",
        required=True,
        help="Path to directory containing 'images/' and 'labels/' subdirectories.",
    )
    parser.add_argument(
        "-ext",
        "--extension",
        default="png",
        help="Image file extension (default: png).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output.mp4",
        help="Output video file (used only when processing multiple images).",
    )
    parser.add_argument(
        "-s",
        "--start-idx",
        type=int,
        default=0,
        help="Start index (integer).",
    )
    parser.add_argument(
        "-e",
        "--end-idx",
        type=int,
        default=0,
        help="End index (integer). If equal to start index, a single image is processed.",
    )

    args = parser.parse_args()

    dataset_path = args.dataset_path
    ext = args.extension
    output_video = args.output
    start_idx = args.start_idx
    end_idx = args.end_idx

    # Check if dataset path exists.
    if not os.path.isdir(dataset_path):
        print(f"Error: Provided dataset path does not exist: {dataset_path}")
        return

    # If start and end indices are the same, process a single image.
    if start_idx == end_idx:
        process_single_image(dataset_path, start_idx, ext)
    else:
        process_video(dataset_path, start_idx, end_idx, ext, output_video)


if __name__ == "__main__":
    main()
