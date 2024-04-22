import depthai as dai
import cv2
import numpy as np

def main():
    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Set up the RGB camera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    # Create output stream
    xout_video = pipeline.create(dai.node.XLinkOut)
    xout_video.setStreamName("video")
    cam_rgb.video.link(xout_video.input)

    # Pipeline is created, now we can connect to the device
    with dai.Device(pipeline) as device:
        # Output queue will be used to get the frames from the output defined above
        queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)

        while True:
            frame_packet = queue.tryGet()  # Get frames from the video output queue
            if frame_packet is not None:
                frame = frame_packet.getCvFrame()
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect QR codes in the frame
                detector = cv2.QRCodeDetector()
                data, bbox, _ = detector.detectAndDecode(gray_frame)

                if bbox is not None:
                    # Properly format and convert the points
                    bbox = np.int0(bbox)  # Convert float coordinates to integer and remove nesting if needed
                    n_points = len(bbox)
                    for i in range(n_points):
                        # Draw polygon around the QR code using formatted points
                        cv2.line(frame, tuple(bbox[i][0]), tuple(bbox[(i + 1) % n_points][0]), (0, 255, 0), 3)

                    if data:
                        print("Detected QR code:", data)
                        # Display data near the first corner
                        cv2.putText(frame, data, (bbox[0][0][0], bbox[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                cv2.imshow("QR Code Detection", frame)

            if cv2.waitKey(1) == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
