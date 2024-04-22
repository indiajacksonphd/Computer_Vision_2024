import cv2
import depthai as dai
import numpy as np

# Load your calibrated parameters
# Replace these with your actual calibration parameters
intrinsic_matrix = np.load('../intrinsic_matrix.npy')
distortion_coeffs = np.load('../distortion_coefficients.npy')


# Function to undistort the image using camera calibration parameters
def undistort_image(image, intrinsic_matrix, distortion_coeffs):
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coeffs, (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(image, intrinsic_matrix, distortion_coeffs, None, new_camera_matrix)
    return undistorted_image


# Function to compute real-world coordinates
def compute_real_world_coordinates(u, v, Z, intrinsic_matrix, extrinsic_matrix):
    fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    X_cam = (u - cx) * Z / fx
    Y_cam = (v - cy) * Z / fy
    camera_coordinates = np.array([X_cam, Y_cam, Z, 1])
    world_coordinates = np.dot(np.linalg.inv(extrinsic_matrix), camera_coordinates)
    return world_coordinates[:3]  # Return only X, Y, Z coordinates


# Setup the Oak-D camera
pipeline = dai.Pipeline()
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("rgb")
camRgb.video.link(xout.input)

# Connect to the device and start the pipeline
with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        frame = qRgb.get()  # Blocking call, will wait until a new data has arrived
        img = frame.getCvFrame()

        # Undistort the image
        undistorted_img = undistort_image(img, intrinsic_matrix, distortion_coeffs)

        # Display the undistorted frame
        cv2.imshow("Undistorted RGB Image", undistorted_img)
        key = cv2.waitKey(1)
        if key == ord('s'):  # Press 's' to save and exit
            # Save the undistorted image
            cv2.imwrite('../undistorted_captured_image.jpg', undistorted_img)
            print("Undistorted image saved!")
            break

'''''
# Assuming you want to convert image points to real-world points,
# you need to have depth information (Z). For example, let's say we have a depth map:
# depth_map = ... (obtain your depth map from Oak-D Lite)

# For demonstration, let's convert the center pixel to real-world coordinates
u, v = img.shape[1] // 2, img.shape[0] // 2  # Center pixel in image
Z = depth_map[v, u]  # Depth value of the center pixel

# Convert from image coordinates to camera coordinates
x_cam = (u - cx) * Z / fx
y_cam = (v - cy) * Z / fy

# Convert from camera coordinates to world coordinates
X_world, Y_world, Z_world, _ = np.dot(np.linalg.inv(extrinsic_matrix), np.array([x_cam, y_cam, Z, 1]))

print(f"Real-world coordinates: X={X_world}, Y={Y_world}, Z={Z_world}")
'''
