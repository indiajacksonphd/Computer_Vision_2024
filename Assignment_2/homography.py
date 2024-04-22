import cv2
import numpy as np

# Load images
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')

# Define pixel patches (super-pixel patches)
patch1 = image1[100:150, 100:150]  # Example patch in image 1
patch2 = image2[120:170, 120:170]  # Corresponding patch in image 2

# Convert patches to grayscale
gray_patch1 = cv2.cvtColor(patch1, cv2.COLOR_BGR2GRAY)
gray_patch2 = cv2.cvtColor(patch2, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Compute SIFT features for patches
kp1, des1 = sift.detectAndCompute(gray_patch1, None)
kp2, des2 = sift.detectAndCompute(gray_patch2, None)

# Compute SSD between SIFT descriptors
ssd = np.sum((des1 - des2) ** 2)

print("SSD between SIFT descriptors:", ssd)

# Compute homography matrix
# Assuming you have corresponding points (pts1, pts2) detected using feature matching
pts1 = np.array([[x.pt[0], x.pt[1]] for x in kp1], dtype=np.float32).reshape(-1, 1, 2)
pts2 = np.array([[x.pt[0], x.pt[1]] for x in kp2], dtype=np.float32).reshape(-1, 1, 2)

H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)

# Compute inverse of homography matrix
H_inv = np.linalg.inv(H)

print("Homography matrix:")
print(H)
print("Inverse of homography matrix:")
print(H_inv)
