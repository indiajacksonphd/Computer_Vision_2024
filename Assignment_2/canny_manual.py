import numpy as np
import cv2


# Function to perform Canny edge detection manually
def manual_canny_edge_detection(image, low_threshold, high_threshold):
    # Step 1: Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Find image gradients using Sobel operators
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Step 4: Compute gradient magnitude and direction
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_direction = np.arctan2(grad_y, grad_x)

    # Step 5: Non-maximum suppression
    edge_image = np.zeros_like(gray)
    for i in range(1, grad_magnitude.shape[0] - 1):
        for j in range(1, grad_magnitude.shape[1] - 1):
            angle = grad_direction[i, j] * 180. / np.pi
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                if (grad_magnitude[i, j] >= grad_magnitude[i, j - 1]) and (
                        grad_magnitude[i, j] >= grad_magnitude[i, j + 1]):
                    edge_image[i, j] = grad_magnitude[i, j]
            elif (22.5 <= angle < 67.5):
                if (grad_magnitude[i, j] >= grad_magnitude[i - 1, j + 1]) and (
                        grad_magnitude[i, j] >= grad_magnitude[i + 1, j - 1]):
                    edge_image[i, j] = grad_magnitude[i, j]
            elif (67.5 <= angle < 112.5):
                if (grad_magnitude[i, j] >= grad_magnitude[i - 1, j]) and (
                        grad_magnitude[i, j] >= grad_magnitude[i + 1, j]):
                    edge_image[i, j] = grad_magnitude[i, j]
            elif (112.5 <= angle < 157.5):
                if (grad_magnitude[i, j] >= grad_magnitude[i - 1, j - 1]) and (
                        grad_magnitude[i, j] >= grad_magnitude[i + 1, j + 1]):
                    edge_image[i, j] = grad_magnitude[i, j]

    # Step 6: Apply hysteresis thresholding
    low_threshold = low_threshold
    high_threshold = high_threshold

    edge_image[edge_image < low_threshold] = 0
    edge_image[edge_image > high_threshold] = 255

    return edge_image.astype(np.uint8)

# Read the input image
image = cv2.imread('frames/frame_0000.jpg')

# Perform Canny edge detection manually
edges = manual_canny_edge_detection(image, low_threshold=30, high_threshold=100)

# Display the result
cv2.imshow('Original Image', image)
cv2.imshow('Canny Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
