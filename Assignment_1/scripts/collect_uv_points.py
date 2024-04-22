import cv2
import numpy as np

# Load images
img1 = cv2.imread('../depth_map/left_frame.png')
img2 = cv2.imread('../depth_map/right_frame.png')

# Detect keypoints and extract descriptors
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# Match keypoints
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Filter matches based on distance
matches = sorted(matches, key=lambda x: x.distance)
good_matches = matches[:50]

# Extract matched keypoints
points1 = np.float32([keypoints1[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
points2 = np.float32([keypoints2[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)

# Estimate essential matrix
F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_LMEDS)

# Decompose essential matrix to obtain R, t
_, R, t, _ = cv2.recoverPose(F, points1, points2)

print("Rotation matrix:")
print(R)
print("Translation vector:")
print(t)
