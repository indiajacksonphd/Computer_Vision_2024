import cv2
import numpy as np
import depthai as dai


# Function to get the frame from the camera queue
def get_frame(queue):
    frame = queue.get()  # Get frame from queue
    return frame.getCvFrame()  # Convert frame to OpenCV format and return

# Function to get the mono camera node for specified board socket
def get_mono_camera(pipeline, is_left):
    mono = pipeline.createMonoCamera()  # Configure mono camera
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)  # Set camera resolution

    if is_left:
        mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
    else:
        mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    return mono

# Function to compute the world coordinates from pixel coordinates and depth value
def compute_world_coordinates(u, v, Z, K, R, t):
    # Convert pixel coordinates to camera coordinates
    x_cam = (u - K[0, 2]) * Z / K[0, 0]
    y_cam = (v - K[1, 2]) * Z / K[1, 1]

    # Convert camera coordinates to world coordinates
    X_world, Y_world, Z_world = np.dot(np.linalg.inv(R), np.array([x_cam, y_cam, Z])) - np.squeeze(t)

    return X_world, Y_world, Z_world


def compute_rotation_translation(img1, img2):
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

    # Estimate fundamental matrix
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_LMEDS)

    # Decompose essential matrix to obtain R, t
    _, R, t, _ = cv2.recoverPose(F, points1, points2)

    return R, t


# Load images
img1 = cv2.imread('../depth_map/left_frame.png')
img2 = cv2.imread('../depth_map/right_frame.png')

# Compute rotation matrix R and translation vector t
R, t = compute_rotation_translation(img1, img2)

# Now you can use the R and t matrices in the compute_world_coordinates function
# Define compute_world_coordinates function here

# Example usage
# u, v, Z = ... # Define pixel coordinates and depth value
X_world, Y_world, Z_world = compute_world_coordinates(u, v, Z, K, R, t)

