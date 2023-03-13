import cv2
import numpy as np

# Read the image
img = cv2.imread('lena.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to the image
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Perform unsharp masking
unsharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

# Perform Laplacian domain frequency filtering
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian_filtered = np.uint8(np.abs(laplacian))

# Perform selective filtering
sel_filter = np.zeros_like(gray)
sel_filter[150:250, 150:250] = 1
sel_filtered = cv2.bitwise_and(gray, gray, mask=sel_filter)

# Display the results
cv2.imshow('Original Image', img)
cv2.imshow('Unsharp Masking', unsharp)
cv2.imshow('Laplacian Domain Frequency Filtering', laplacian_filtered)
cv2.imshow('Selective Filtering', sel_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
