import depthai as dai
import cv2

def live_camera_preview():
    """
    Starts a live preview from the OAK-D camera and allows the user to save images by pressing 's' or quit by pressing 'q'.
    """
    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Create a Color camera node
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    # Create XLink output to send RGB frames to the host (the computer)
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    # Pipeline is defined, now we can connect to the device
    with dai.Device(pipeline) as device:
        print("Starting camera preview...")
        # Output queue will be used to get the frames from the camera output
        q_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)

        while True:
            in_rgb = q_rgb.tryGet()

            if in_rgb is not None:
                # Retrieve 'bgr' (OpenCV format) frame
                cv_frame = in_rgb.getCvFrame()
                cv2.imshow("Camera Preview", cv_frame)

            key = cv2.waitKey(1)
            if key == ord('s'):
                # Save the frame when 's' key is pressed
                cv2.imwrite("captured_image_rotate.png", cv_frame)
                print("Image captured and saved!")
            elif key == ord('q'):
                # Quit the preview when 'q' key is pressed
                break

        cv2.destroyAllWindows()


def compute_disparity(left_image_path, right_image_path):
    """
    Compute the disparity map from two stereo images.

    Args:
    left_image_path (str): Path to the left image.
    right_image_path (str): Path to the right image.

    Returns:
    None
    """
    # Load the stereo images in grayscale
    img_left = cv2.imread(left_image_path, 0)
    img_right = cv2.imread(right_image_path, 0)

    # Check if the images are loaded properly
    if img_left is None or img_right is None:
        raise ValueError("Could not open one or both images. Check the file paths.")

    # Initialize the stereo block matcher with the correct blockSize
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

    # Compute the disparity map
    disparity = stereo.compute(img_left, img_right)

    # Normalize the disparity for visualization
    disparity_visual = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Display and save the disparity map
    cv2.imshow('Disparity Map', disparity_visual)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('disparity_map.png', disparity_visual)

    # Calculate and print disparity at the center
    h, w = img_left.shape
    center_x, center_y = w // 2, h // 2
    marker_disparity = disparity[center_y, center_x]
    print(f"The disparity for the marker at the center is: {marker_disparity}")

    # Optionally, print out a region around the center
    region_size = 10  # Define the region size
    half_size = region_size // 2
    region_disparity = disparity[center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size]
    print(f"The disparity for the region around the center is:\n{region_disparity}")


live_camera_preview()
# compute_disparity('captured_image.png', 'captured_image_2.png')
# compute_disparity('captured_image_rotate.png', 'captured_image_rotate2.png')
