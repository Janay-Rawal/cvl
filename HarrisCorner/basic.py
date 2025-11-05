import cv2
import numpy as np

# Read image
img_bgr = cv2.imread('building.jpg')
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Compute Sobel derivatives
Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

Ixx = Ix ** 2
Iyy = Iy ** 2
Ixy = Ix * Iy

# Apply Gaussian Blur
Ixx = cv2.GaussianBlur(Ixx, (5, 5), 0)
Iyy = cv2.GaussianBlur(Iyy, (5, 5), 0)
Ixy = cv2.GaussianBlur(Ixy, (5, 5), 0)

# Harris response
k = 0.04
R = (Ixx * Iyy - Ixy ** 2) - k * (Ixx + Iyy) ** 2

# Thresholding
threshold = 0.001 * R.max()
corners = (R > threshold)

# --- Prepare 3 versions ---
# 1. Grayscale with corners
gray_with_corners = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # convert to BGR for marking
gray_with_corners[corners] = [0, 0, 255]

# 2. RGB with corners
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_rgb_with_corners = img_rgb.copy()
img_rgb_with_corners[corners] = [255, 0, 0]  # red in RGB

# 3. BGR with corners
img_bgr_with_corners = img_bgr.copy()
img_bgr_with_corners[corners] = [0, 255, 0]  # green in BGR

# --- Display ---
cv2.imshow('Harris Corners on Grayscale', gray_with_corners)
cv2.imshow('Harris Corners on RGB', img_rgb_with_corners)
cv2.imshow('Harris Corners on BGR', img_bgr_with_corners)

cv2.waitKey(0)
cv2.destroyAllWindows()