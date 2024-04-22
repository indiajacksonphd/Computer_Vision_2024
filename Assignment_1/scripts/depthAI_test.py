import cv2
import depthai as dai


def get_frame(queue):
    frame = queue.get()  # get frame from queue
    return frame.getCvFrame()  # convert frame to OpenCV format and return


if __name__ == '__main__':
    pipeline = dai.Pipeline()  # define a pipeline

    # Set up color camera
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)  # Use the RGB (color) camera

    # Create XLinkOut for color camera
    x_out_rgb = pipeline.createXLinkOut()
    x_out_rgb.setStreamName("rgb")
    cam_rgb.video.link(x_out_rgb.input)  # Link color camera to XLinkOut

    with dai.Device(pipeline) as device:
        rgb_queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)  # get output queue for color camera

        cv2.namedWindow("Color Camera")  # set display window name

        while True:
            rgb_frame = get_frame(rgb_queue)  # get color frame
            cv2.imshow("Color Camera", rgb_frame)  # display output image

            if cv2.waitKey(1) == ord('q'):
                break  # quit when 'q' is pressed

