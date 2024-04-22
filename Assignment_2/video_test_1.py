import depthai as dai
import subprocess
import ffmpeg
import time

# Define the recording duration in seconds
record_duration = 10

# Create pipeline
pipeline = dai.Pipeline()

# Define ColorCamera node
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)

# Create VideoEncoder node
videoEnc = pipeline.create(dai.node.VideoEncoder)

# Create XLinkOut node
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("h265")

# Link nodes
camRgb.video.link(videoEnc.input)
videoEnc.bitstream.link(xout.input)

# Connect to the device
with dai.Device(pipeline) as device:
    # Output queue will be used to get the encoded data from the output defined above
    q = device.getOutputQueue(name="h265", maxSize=30, blocking=True)

    # Open a file to save the video
    with open('videos/video.h265', 'wb') as videoFile:
        print("Recording video...")

        # Record for the specified duration
        start_time = time.time()
        while time.time() - start_time < record_duration:
            # Get encoded video data from the output queue
            h265Packet = q.get()
            h265Packet.getData().tofile(videoFile)

    print("Recording stopped.")

# Convert the stream file (.h265) into a video file (.mp4) using ffmpeg
print("Converting the stream file into a video file...")
print("ffmpeg -framerate 30 -i video.h265 -c copy video.mp4")

# Convert the stream file (.h265) into a video file (.mp4) using ffmpeg
print("Converting the stream file into a video file...")
# subprocess.run(["ffmpeg", "-framerate", "30", "-i", "video.h265", "-c", "copy", "video.mp4"])
# Example command to convert video
ffmpeg.input('input.mp4').output('output.mp4').run()
print("Video conversion completed.")
