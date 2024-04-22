import cv2
import numpy as np
import matplotlib.pyplot as plt

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=1000)

reference_image = cv2.imread('video_frames/frame_0197.png', cv2.IMREAD_GRAYSCALE)
target_image = cv2.imread('video_frames/frame_0198.png', cv2.IMREAD_GRAYSCALE)

# Ensure images loaded correctly
if reference_image is None or target_image is None:
    raise ValueError("One of the images didn't load correctly. Check the file paths.")

# Detect keypoints and compute descriptors
reference_keypoints, reference_descriptors = orb.detectAndCompute(reference_image, None)
target_keypoints, target_descriptors = orb.detectAndCompute(target_image, None)

# Create a BFMatcher object with Hamming distance and crossCheck
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = matcher.match(reference_descriptors, target_descriptors)

# Ensure 'matches' is a list for sorting
if not isinstance(matches, list):
    matches = list(matches)

# Sort matches by score
matches.sort(key=lambda x: x.distance)

# Draw top N matches
N = 50
matched_image = cv2.drawMatches(reference_image, reference_keypoints, target_image, target_keypoints, matches[:N], None)

# Display the matched image
cv2.imshow('Matched Features', matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

