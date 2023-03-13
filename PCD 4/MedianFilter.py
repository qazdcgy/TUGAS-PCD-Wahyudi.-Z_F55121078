from scipy.signal import medfilt2d
import cv2

img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

filtered_img = medfilt2d(img, kernel_size=3)

cv2.imshow('Original Image', img)
cv2.imshow('Filtered Image', filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()