import cv2
import numpy as np
import glob
import random
import os


def extract_frames(video_path, save_dir):

    # Check if the directory where we want to save the frames exists; if not, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Initialize a frame count
    frame_count = 0

    # Frame extraction and saving
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Save frame as image file
        cv2.imwrite(os.path.join(save_dir, f'frame_{frame_count:04d}.png'), frame)
        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Extracted {frame_count} frames and saved to directory: '{save_dir}'")


# Function to add a grid to an image
def add_grid_to_image(image_path, grid_horizontal, grid_vertical):
    image = cv2.imread(image_path)
    step_size_x = image.shape[1] // grid_horizontal
    step_size_y = image.shape[0] // grid_vertical
    image_with_grid = image.copy()

    for x in range(0, image.shape[1], step_size_x):
        cv2.line(image_with_grid, (x, 0), (x, image.shape[0]), color=(0, 255, 0), thickness=1)
    for y in range(0, image.shape[0], step_size_y):
        cv2.line(image_with_grid, (0, y), (image.shape[1], y), color=(0, 255, 0), thickness=1)

    grid_image_path = 'image_with_grid.png'
    cv2.imwrite(grid_image_path, image_with_grid)
    return grid_image_path


# Function to crop a region of interest from an image
def crop_roi(image_path, x, y, width, height):
    image = cv2.imread(image_path)
    roi = image[y:y+height, x:x+width]
    cropped_roi_path = 'cropped_roi.png'
    cv2.imwrite(cropped_roi_path, roi)
    return cropped_roi_path


# Function to compare a cropped ROI with a list of images using SSD
def compare_images(roi_path, frame_paths, x, y, width, height):
    cropped_roi = cv2.imread(roi_path)
    cropped_roi_gray = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2GRAY)

    ssd_results = {}
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_section = frame_gray[y:y+height, x:x+width]
        ssd = np.sum((cropped_roi_gray - frame_section) ** 2)
        ssd_results[frame_path] = ssd

    return ssd_results

# Example usage of the functions

'''''
# Path to your video
video_path = 'assignment_3.mp4'

# Directory where you want to save the frames
save_dir = 'video_frames'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load the video
cap = cv2.VideoCapture(video_path)

# Initialize a frame count
frame_count = 0

# Frame extraction and saving
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Save frame as image file
    cv2.imwrite(os.path.join(save_dir, f'frame_{frame_count:04d}.png'), frame)
    frame_count += 1

cap.release()


image = cv2.imread('video_frames/frame_0197.png')


# Get the dimensions of the image
height, width, channels = image.shape

print(f"The image width is: {width}")
print(f"The image height is: {height}")
print(f"The number of channels is: {channels}")
'''

extract_frames('assignment_3.mp4', 'video_frames')
# Add grid to the image and get path of the new image with grid
grid_image_path = add_grid_to_image('video_frames/frame_0197.png', 40, 40)

# Crop the ROI and get path of the cropped image
cropped_image_path = crop_roi('video_frames/frame_0197.png', 1824, 216, 48, 135)

# Get all frames in the directory and randomly pick 10 frames
all_frames = glob.glob('video_frames/*.png')
selected_frames_paths = random.sample(all_frames, 10)

# Compare the cropped ROI with the selected frames and get SSD results
ssd_results = compare_images(cropped_image_path, selected_frames_paths, 1824, 216, 48, 135)

# Print SSD results
for frame_path, ssd in ssd_results.items():
    print(f"SSD for {frame_path}: {ssd}")
