import cv2
import numpy as np

# Load the stereo images
img_left = cv2.imread('captured_image.png', 0)  # Assuming grayscale
img_right = cv2.imread('captured_image_2.png', 0)  # Assuming grayscale

# Check if the images are loaded properly
if img_left is None or img_right is None:
    raise ValueError("Could not open one or both images. Check the file paths.")

# Initialize the stereo block matcher with the correct blockSize
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# Compute the disparity map
disparity = stereo.compute(img_left, img_right)

# Convert the raw disparity to a visible image (scaling it to the 0-255 range)
disparity_visual = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# You might want to select a specific point, here's how you could do it for a central point
h, w = img_left.shape
center_x, center_y = w // 2, h // 2

# Assume the marker is around the center, adjust the x and y accordingly if needed
marker_disparity = disparity[center_y, center_x]

print(f"The disparity for the marker is: {marker_disparity}")

# If you want to print out a region (e.g., 10x10 pixels around the center)
region_size = 10  # Define the region size
half_size = region_size // 2
region_disparity = disparity[center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size]

print(f"The disparity for the region around the marker is:\n{region_disparity}")

# Show the disparity map
cv2.imshow('Disparity Map', disparity_visual)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the visual disparity map
cv2.imwrite('disparity_map.png', disparity_visual)
