import cv2
import numpy as np

# Function to stitch two images together
def image_stitch(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Match descriptors using FLANN matcher
    flann = cv2.FlannBasedMatcher_create()
    matches = flann.knnMatch(des1, des2, k=2)

    # Filter matches using ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Estimate homography if enough good matches are found
    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp img1 to img2 using estimated homography
        warped_img = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))

        # Blend the warped image and img2
        warped_img[:, 0:img2.shape[1]] = img2

        return warped_img
    else:
        return None

# Open the camera
cap = cv2.VideoCapture(0)

# Capture the first frame
ret, prev_frame = cap.read()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Stitch the current frame with the previous frame
    stitched_img = image_stitch(prev_frame, frame)

    if stitched_img is not None:
        # Display the stitched image
        cv2.imshow('Stitched Image', stitched_img)

    # Update the previous frame
    prev_frame = frame.copy()

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
