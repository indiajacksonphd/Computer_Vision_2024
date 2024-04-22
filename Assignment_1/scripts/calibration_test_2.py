import numpy as np
import cv2
import glob

# Define the checkerboard dimensions
checkerboard_size = (7, 7)  # (width-1, height-1)
square_size = 49.21  # Set the actual size of the squares on your checkerboard (mm or inches)

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (6,9,0)
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all images
objpoints = []  # 3d points in real-world space
imgpoints = []  # 2d points in image plane

# List of paths to calibration images
images = glob.glob('calibration_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
    print(f"Processing {fname}: Corners found = {ret}")

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        print(f"All corners for {fname}: {corners.reshape(-1, 2)}")
        print(f"Refined corners for {fname}: {corners2.reshape(-1, 2)}")
        print(f"World coordinates (object points) for {fname}: {objp.reshape(-1, 3)}")
    else:
        print(f"No corners found in {fname}.")

# After detecting corners and refining them:
for fname, objp_single, corners2_single in zip(images, objpoints, imgpoints):
    # Assuming your camera is properly calibrated
    H, _ = cv2.findHomography(objp_single[:, :2], corners2_single.squeeze(), method=cv2.RANSAC)
    print(f"Homography for {fname}: \n{H}\n")

# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print out the calibration matrix (mtx) and distortion coefficients (dist)
print("Camera matrix:", mtx)
print("Distortion coefficients:", dist)

rvec = rvecs[8]  # Example index
tvec = tvecs[8]  # Example index

# Convert rotation vector to rotation matrix
R, _ = cv2.Rodrigues(rvec)

# Compute Euler angles from R, the method depends on the rotation order
# e.g., using some function euler_from_matrix(R, 'sxyz') which you might define based on the chosen convention

# Print results
print("Rotation matrix:", R)
# print("Euler angles:", euler_angles)  # Assuming you have a method to convert

