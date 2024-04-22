import depthai as dai
import time

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and output
camRgb = pipeline.createColorCamera()
videoEnc = pipeline.createVideoEncoder()
xout = pipeline.createXLinkOut()

xout.setStreamName('h265')

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setFps(30)
videoEnc.setDefaultProfilePreset(camRgb.getFps(), dai.VideoEncoderProperties.Profile.H265_MAIN)

# Linking
camRgb.video.link(videoEnc.input)
videoEnc.bitstream.link(xout.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the encoded data from the output defined above
    q = device.getOutputQueue(name="h265", maxSize=30, blocking=True)

    # Start video recording
    with open('videos/output_video.h265', 'wb') as videoFile:
        print("Press Ctrl+C to stop encoding...")
        start_time = time.time()
        try:
            while time.time() - start_time < 10:  # Capture video for 10 seconds
                h265Packet = q.get()  # Blocking call, will wait until new data has arrived
                h265Packet.getData().tofile(videoFile)  # Append the packet data to the opened file

                # Implement camera panning functionality here
                # Example: Adjust camera angle by updating camRgb's settings
                # Example: time.sleep(0.1)  # Adjust sleep time as needed to control panning speed
        except KeyboardInterrupt:
            # Keyboard interrupt (Ctrl + C) detected
            pass

    print("Video capture complete. The encoded data has been saved to output_video.h265.")
