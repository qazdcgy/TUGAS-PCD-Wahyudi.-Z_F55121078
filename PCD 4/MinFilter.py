import cv2

from scipy.ndimage.filters import minimum_filter


def min_filter(image, size):
    filtered = minimum_filter(image, size)

    return filtered

img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

filtered_img = min_filter(img, size=3)

cv2.imshow('Original Image', img)
cv2.imshow('Filtered Image', filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()