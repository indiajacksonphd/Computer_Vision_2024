import depthai as dai
import cv2
import numpy as np

def main():
    # Create a pipeline
    pipeline = dai.Pipeline()

    # Setup mono cameras
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    # Create a node for stereo depth estimation
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.initialConfig.setConfidenceThreshold(200)
    stereo.setSubpixel(False)
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # Create a color camera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # Create output streams
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    # Connect to device and start the pipeline
    with dai.Device(pipeline) as device:
        rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        while True:
            rgb_frame = rgb_queue.get().getCvFrame()
            depth_frame = depth_queue.get().getFrame()

            # Convert RGB frame to HSV for color segmentation
            hsv_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)
            lower_color = np.array([20, 100, 100])  # lower boundary of the color yellow
            upper_color = np.array([30, 255, 255])  # upper boundary of the color yellow
            mask = cv2.inRange(hsv_frame, lower_color, upper_color)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # Extract depth information within the object's bounding box
                    depth_roi = depth_frame[y:y+h, x:x+w]
                    average_depth = np.mean(depth_roi) * 0.1  # Convert to cm (assumes depth in mm)
                    cv2.putText(rgb_frame, f"Depth: {average_depth:.2f} cm", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Display the frames
            cv2.imshow("RGB Frame", rgb_frame)
            cv2.imshow("Mask", mask)

            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
