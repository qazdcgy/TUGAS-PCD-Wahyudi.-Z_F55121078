import numpy as np
import cv2

def max_filter(image, kernel_size):
    padded_image = np.pad(image, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2)), mode='edge')
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            filtered_image[i, j] = np.max(padded_image[i:i+kernel_size, j:j+kernel_size])
    return filtered_image

image = cv2.imread('lena.jpg', 0)

filtered_image = max_filter(image, 3)

cv2.imshow('Original', image)
cv2.imshow('Max filtered', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()