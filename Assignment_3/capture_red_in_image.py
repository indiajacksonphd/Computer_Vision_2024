import cv2
import numpy as np

# Load the image
image = cv2.imread('video_frames/frame_0193.png')

'''''
# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define range for red color in HSV
# Adjust these values based on the shade of red
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])
'''''

# Convert the RGB color to HSV color space
# OpenCV uses H: 0-179, S: 0-255, V: 0-255
color = np.uint8([[[78, 10, 24]]])  # RGB
hsv_color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)

# Get the HSV value for the color and define a range around it
hsv_value = hsv_color[0][0]
hue = hsv_value[0]
saturation = hsv_value[1]
value = hsv_value[2]

# Define a range for the color
# You may need to adjust these ranges to get the best result
lower_range = np.array([hue-10, max(saturation-50, 0), max(value-50, 0)])
upper_range = np.array([hue+10, min(saturation+50, 255), min(value+50, 255)])

# Create a mask for red color
# mask = cv2.inRange(hsv, lower_red, upper_red)

mask = cv2.inRange(hsv_color, lower_range, upper_range)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour based on area and get bounding box
largest_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest_contour)

# Crop the ROI from the image
roi = image[y:y+h, x:x+w]

# Show the cropped ROI
cv2.imshow('Cropped ROI', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the cropped ROI
cv2.imwrite('cropped_roi.png', roi)
