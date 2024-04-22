import cv2
import numpy as np
import glob
import random

# Load the cropped region
cropped_roi = cv2.imread('cropped_roi.png')
cropped_roi_gray = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if comparing in grayscale

# List all frames in the directory
all_frames = glob.glob('video_frames/*.png')

# Randomly pick 10 frames
selected_frames_paths = random.sample(all_frames, 10)


# Go through each selected frame and compare
for frame_path in selected_frames_paths:
    # Load the frame
    frame = cv2.imread(frame_path)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if comparing in grayscale

    x, y = 1824, 216
    width, height = 48, 135

    # Ensure the ROI and the frame section you're comparing are the same size
    # You might need to adjust x, y to match where the ROI would be in this new frame
    frame_section = frame_gray[y:y+height, x:x+width]

    # Calculate SSD between the cropped ROI and the frame section
    ssd = np.sum((cropped_roi_gray - frame_section) ** 2)

    # Print out the SSD value for each comparison
    print(f"SSD for {frame_path}: {ssd}")
