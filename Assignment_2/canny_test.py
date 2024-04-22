import cv2
import numpy as np

# Load the image frame
frame = cv2.imread('frames/frame_0000.jpg')  # Replace 'frame_0000.jpg' with the path to your extracted frame

# Select a region of interest (ROI) in the image
# For demonstration, let's define a rectangular ROI manually
roi_x, roi_y, roi_width, roi_height = 100, 100, 200, 200  # Example ROI coordinates and dimensions
roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]

# Apply Canny edge detection to the ROI manually
edges_manual = cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), threshold1=50, threshold2=150)  # Adjust thresholds as needed

# Display the original image and the manually detected edges
cv2.imshow('Original Image', frame)
cv2.imshow('Manually Detected Edges', edges_manual)
cv2.waitKey(0)
cv2.destroyAllWindows()
