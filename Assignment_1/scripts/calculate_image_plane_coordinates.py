import numpy as np

# Define the intrinsic matrix K
K = np.array([
    [1.47678135e+03, 0.00000000e+00, 9.87986065e+02],
    [0.00000000e+00, 1.47926479e+03, 5.81230952e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

# Define the rotation matrix R and translation matrix t for image 8
R = np.array([
    [0.9630102, -0.25518618, -0.08655271],
    [0.26926717, 0.92361576, 0.27281702],
    [0.01032231, -0.28603137, 0.95816465]
])
t = np.array([
    [-4.87948859],
    [-255.25656046],
    [831.88903297]
])

# Combine R and t to form the extrinsic matrix
extrinsic_matrix = np.hstack((R, t))
print(extrinsic_matrix)

# Example world coordinate (the first point)
X_world = np.array([[0], [0], [0], [1]])  # Homogeneous coordinates
print(X_world)

print(extrinsic_matrix.dot(X_world))

# Calculate the image plane coordinates (u, v)
uv_homogeneous = K.dot(extrinsic_matrix.dot(X_world))
print(uv_homogeneous)
uv = uv_homogeneous / uv_homogeneous[2]  # Normalize by w to get (u, v, 1)

print(f"Image coordinates (u, v): {uv_homogeneous[0][0]}, {uv_homogeneous[1][0]}")
print(f"Image coordinates (u, v): {uv[0][0]}, {uv[1][0]}")
