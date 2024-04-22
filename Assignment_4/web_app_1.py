import depthai as dai
import cv2
import numpy as np

def main():
    # Create pipeline
    pipeline = dai.Pipeline()

    # Create and configure the color camera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)

    # Create output stream for video
    xout_video = pipeline.create(dai.node.XLinkOut)
    xout_video.setStreamName("video")
    cam_rgb.preview.link(xout_video.input)

    # Pipeline is defined, now we can connect to the device
    with dai.Device(pipeline) as device:
        # Output queue will be used to get the rgb frames from the output defined above
        video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)

        while True:
            frame = video_queue.get().getCvFrame()
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define range of target color in HSV
            lower_color = np.array([20, 100, 100])  # e.g., lower boundary for yellow
            upper_color = np.array([30, 255, 255])  # e.g., upper boundary for yellow

            # Threshold the HSV image to get only the target colors
            mask = cv2.inRange(hsv_frame, lower_color, upper_color)
            res = cv2.bitwise_and(frame, frame, mask=mask)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('Frame', frame)
            cv2.imshow('Mask', mask)
            cv2.imshow('Filtered Color', res)

            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
