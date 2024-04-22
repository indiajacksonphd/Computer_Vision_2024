import numpy as np
import cv2

# Function to perform Harris corner detection manually
def manual_harris_corner_detection(image, ksize, k, threshold):
    # Step 1: Compute gradients
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)

    # Step 2: Compute products of gradients
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Step 3: Compute the sums of products in the local neighborhood
    Sxx = cv2.GaussianBlur(Ixx, (ksize, ksize), 0)
    Syy = cv2.GaussianBlur(Iyy, (ksize, ksize), 0)
    Sxy = cv2.GaussianBlur(Ixy, (ksize, ksize), 0)

    # Step 4: Compute the Harris response
    det = Sxx * Syy - Sxy * Sxy
    trace = Sxx + Syy
    R = det - k * (trace ** 2)

    # Step 5: Thresholding and non-maximum suppression
    corners = np.zeros_like(image)
    corners[R > threshold * R.max()] = 255

    return corners.astype(np.uint8)

# Read the input image
image = cv2.imread('frames/frame_0000.jpg', cv2.IMREAD_GRAYSCALE)

# Perform Harris corner detection manually
corners = manual_harris_corner_detection(image, ksize=3, k=0.04, threshold=0.01)

# Display the result
cv2.imshow('Original Image', image)
cv2.imshow('Harris Corners', corners)
cv2.waitKey(0)
cv2.destroyAllWindows()
