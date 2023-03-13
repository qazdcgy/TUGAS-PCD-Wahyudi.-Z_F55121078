import cv2
import numpy as np

# read input image
img = cv2.imread('lena.jpg', 0)

# get image dimensions
height, width = img.shape

# create highpass filter kernels
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# create Butterworth highpass filter kernel
butterworth_highpass = np.zeros((height, width, 2), np.float32)
d0 = 50  # cutoff frequency
n = 2  # order of the filter
for i in range(height):
    for j in range(width):
        distance = np.sqrt((i - height/2)**2 + (j - width/2)**2)
        butterworth_highpass[i, j] = 1 / (1 + (distance/d0)**(2*n))

# apply Butterworth highpass filter
butterworth_highpass_filter = dft_shift * butterworth_highpass
butterworth_highpass_filter_shift = np.fft.ifftshift(butterworth_highpass_filter)
butterworth_highpass_image = cv2.idft(butterworth_highpass_filter_shift)
butterworth_highpass_image = cv2.magnitude(butterworth_highpass_image[:, :, 0], butterworth_highpass_image[:, :, 1])

# create Gaussian highpass filter kernel
gaussian_highpass = np.zeros((height, width, 2), np.float32)
d0 = 50  # cutoff frequency
for i in range(height):
    for j in range(width):
        distance = np.sqrt((i - height/2)**2 + (j - width/2)**2)
        gaussian_highpass[i, j] = 1 - np.exp(-(distance**2)/(2*d0**2))

# apply Gaussian highpass filter
gaussian_highpass_filter = dft_shift * gaussian_highpass
gaussian_highpass_filter_shift = np.fft.ifftshift(gaussian_highpass_filter)
gaussian_highpass_image = cv2.idft(gaussian_highpass_filter_shift)
gaussian_highpass_image = cv2.magnitude(gaussian_highpass_image[:, :, 0], gaussian_highpass_image[:, :, 1])

# display filtered images
cv2.imshow('Original Image', img)
cv2.imshow('Butterworth Highpass Filter', butterworth_highpass_image)
cv2.imshow('Gaussian Highpass Filter', gaussian_highpass_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
