import cv2
import numpy as np
import glob
import random


image = cv2.imread('video_frames/frame_0197.png')

# Get the dimensions of the image
height, width, channels = image.shape

print(f"The image width is: {width}")
print(f"The image height is: {height}")
print(f"The number of channels is: {channels}")


#----------------------------------- Add Grid ------------------------#

# Define the number of grid cells
grid_cells_horizontal = 40
grid_cells_vertical = 40

# Calculate the step size for each cell
step_size_x = image.shape[1] // grid_cells_horizontal  # Image width divided by number of cells horizontally
step_size_y = image.shape[0] // grid_cells_vertical    # Image height divided by number of cells vertically

# Create a copy of the image to draw the grid on
image_with_grid = image.copy()

# Draw the vertical grid lines
for x in range(0, image.shape[1], step_size_x):
    cv2.line(image_with_grid, (x, 0), (x, image.shape[0]), color=(0, 255, 0), thickness=1)

# Draw the horizontal grid lines
for y in range(0, image.shape[0], step_size_y):
    cv2.line(image_with_grid, (0, y), (image.shape[1], y), color=(0, 255, 0), thickness=1)

# Save image with grid
cv2.imwrite('image_with_grid.png', image_with_grid)

# Show the image with the grid
cv2.imshow('Image with Grid', image_with_grid)
cv2.waitKey(0)
cv2.destroyAllWindows()

#----------------------------------- Crop Object ------------------------#
x, y = 1824, 216
width, height = 48, 135

# Crop the ROI from the image
roi = image[y:y+height, x:x+width]

cv2.imwrite('cropped_roi.png', roi)
# Show the cropped ROI
cv2.imshow('Cropped ROI', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()


#----------------------------------- Image comparison ------------------------#
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


