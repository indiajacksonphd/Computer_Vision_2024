import cv2
import depthai as dai
import numpy as np

# Load camera calibration parameters
intrinsic_matrix = np.load('../intrinsic_matrix.npy')
distortion_coeffs = np.load('../distortion_coefficients.npy')

# Setup the DepthAI camera
pipeline = dai.Pipeline()

# Define a node to capture RGB frames
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.video.link(xoutRgb.input)

# Define a StereoDepth node to obtain depth information
depth = pipeline.create(dai.node.StereoDepth)
depth.initialConfig.setConfidenceThreshold(255)
depth.setRectifyEdgeFillColor(0)  # Black fill color
depth.setDepthAlign(dai.CameraBoardSocket.RGB)
camRgb.video.link(depth.depth)
depth.out.link(xoutRgb.input)

# Connect to the device and start the pipeline
with dai.Device(pipeline) as device:
    # Get camera output queues
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        # Get the RGB frame
        rgbFrame = qRgb.get()  # Blocking call, will wait until a new data has arrived
        img = rgbFrame.getCvFrame()

        # Undistort the image
        undistorted_img = cv2.undistort(img, intrinsic_matrix, distortion_coeffs)

        # Get the depth map
        depthFrame = depthQueue.get()
        depthMap = depthFrame.getFrame()

        # Convert image coordinates to real-world coordinates
        # For demonstration, let's consider the center pixel in the image
        u, v = img.shape[1] // 2, img.shape[0] // 2  # Center pixel in image
        depthValue = depthMap[v, u]  # Depth value of the center pixel

        # Assuming you know the baseline distance and focal length
        baselineDistance = 0.1  # Example baseline distance in meters
        focalLength = intrinsic_matrix[0, 0]  # Focal length in pixels

        # Convert to real-world coordinates
        x_world = (u - intrinsic_matrix[0, 2]) * depthValue / focalLength
        y_world = (v - intrinsic_matrix[1, 2]) * depthValue / focalLength
        z_world = depthValue

        print("Real-world coordinates (X, Y, Z):", x_world, y_world, z_world)

        # Display the undistorted frame
        cv2.imshow("Undistorted RGB Image", undistorted_img)
        key = cv2.waitKey(1)
        if key == ord('s'):  # Press 's' to save and exit
            break

cv2.destroyAllWindows()
