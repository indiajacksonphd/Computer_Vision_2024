import cv2
import depthai as dai
import numpy as np


def get_frame(queue):
    frame = queue.get()  # get frame from queue
    return frame.getCvFrame()  # convert frame to OpenCV format and return


def get_mono_camera(pipeline, is_left):
    mono = pipeline.createMonoCamera()  # configure mono camera
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)  # set camera resolution

    if is_left:
        mono.setBoardSocket(dai.CameraBoardSocket.CAM_B)  # get left camera
    else:
        mono.setBoardSocket(dai.CameraBoardSocket.CAM_C)  # get right camera

    return mono


if __name__ == '__main__':
    pipeline = dai.Pipeline()  # define a pipeline

    mono_left = get_mono_camera(pipeline, is_left=True)  # set up left camera
    mono_right = get_mono_camera(pipeline, is_left=False)  # set up right camera

    x_out_left = pipeline.createXLinkOut()  # set output xlink for left camera
    x_out_left.setStreamName("left")  # set name xlink for left camera

    x_out_right = pipeline.createXLinkOut()  # set output xlink for left camera
    x_out_right.setStreamName("right")  # set name xlink for left camera

    mono_left.out.link(x_out_left.input)  # attach left camera to left xlink
    mono_right.out.link(x_out_right.input)  # attach right camera to right xlink

    with dai.Device(pipeline) as device:

        left_queue = device.getOutputQueue(name="left", maxSize=1)  # get output queue for left camera
        right_queue = device.getOutputQueue(name="right", maxSize=1)  # get output queue for right camera

        cv2.namedWindow("Stereo Pair")  # set display window name
        side_by_side = True  # toggle left and right windows side by side

        while True:
            left_frame = get_frame(left_queue)  # get left frame
            right_frame = get_frame(right_queue)  # get right frame

            if side_by_side:
                image_out = np.hstack((left_frame, right_frame))  # show side by side view
            else:
                image_out = np.uint8(left_frame/2 + right_frame/2)  # show overlapping frames

            cv2.imshow("Stereo Pair", image_out)  # display output image

            key = cv2.waitKey(1)  # check for keyboard input
            if key == ord('q'):
                break  # quit when q is pressed
            elif key == ord('t'):
                side_by_side = not side_by_side





