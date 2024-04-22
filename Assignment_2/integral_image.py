import cv2
import numpy as np


# Function to compute integral image (summed area table)
def compute_integral_image(image):
    integral_image = np.zeros_like(image, dtype=np.uint32)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            integral_image[y, x] = image[y, x] + \
                                   (integral_image[y-1, x] if y > 0 else 0) + \
                                   (integral_image[y, x-1] if x > 0 else 0) - \
                                   (integral_image[y-1, x-1] if (x > 0 and y > 0) else 0)
    return integral_image


# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute integral image
    integral_image = compute_integral_image(gray)

    # Convert integral image to uint8 for display
    integral_image_display = cv2.normalize(integral_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Display RGB feed and integral image feed side by side
    cv2.imshow('RGB Feed', frame)
    cv2.imshow('Integral Image Feed', integral_image_display)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
