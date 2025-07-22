import cv2
import numpy as np

def nothing(x):
    pass  # This is a dummy function required by createTrackbar

# Create a black image and a window
img = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow('Color Window')

# Create trackbars for R, G, B values
cv2.createTrackbar('R', 'Color Window', 0, 255, nothing)
cv2.createTrackbar('G', 'Color Window', 0, 255, nothing)
cv2.createTrackbar('B', 'Color Window', 0, 255, nothing)

while True:
    # Show the image
    cv2.imshow('Color Window', img)

    # Wait for key and break if 'ESC' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # Get current positions of all trackbars
    r = cv2.getTrackbarPos('R', 'Color Window')
    g = cv2.getTrackbarPos('G', 'Color Window')
    b = cv2.getTrackbarPos('B', 'Color Window')

    # Update image color
    img[:] = [b, g, r]  # OpenCV uses BGR order

cv2.destroyAllWindows()
