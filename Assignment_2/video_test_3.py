import cv2

# Path to the input video file
video_path = 'videos/video_test_2.mov'

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Frame index counter
frame_count = 0

# Loop through the video frames
while True:
    # Read the next frame from the video
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if not ret:
        break  # End of video

    # Save the frame as an image file
    frame_filename = f'frames/frame_{frame_count:04d}.jpg'  # Adjust the filename format as needed
    cv2.imwrite(frame_filename, frame)

    # Print the frame number
    print(f"Saved frame {frame_count}")

    # Increment the frame index counter
    frame_count += 1

# Release the VideoCapture object and close the video file
cap.release()

print("Frame extraction complete.")
