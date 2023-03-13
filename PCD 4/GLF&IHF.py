import cv2
import numpy as np

# Load image
img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

# Define filter parameters
kernel_size = 5  # filter size
sigma = 1.5      # standard deviation for Gaussian kernel
cutoff_freq = 20 # cutoff frequency for highpass filter

# Create Gaussian lowpass filter
kernel = cv2.getGaussianKernel(kernel_size, sigma)
kernel = np.outer(kernel, kernel)

# Apply Gaussian lowpass filter to image
img_filtered = cv2.filter2D(img, -1, kernel)

# Create ideal highpass filter
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2  # center of the image
mask = np.ones((rows, cols), np.uint8)
mask[crow-cutoff_freq:crow+cutoff_freq, ccol-cutoff_freq:ccol+cutoff_freq] = 0

# Apply ideal highpass filter to image
fimg = np.fft.fft2(img)
fimg_shifted = np.fft.fftshift(fimg)
fimg_filtered = fimg_shifted * mask
fimg_filtered_shifted = np.fft.ifftshift(fimg_filtered)
img_filtered_highpass = np.abs(np.fft.ifft2(fimg_filtered_shifted))

# Display filtered images
cv2.imshow('Gaussian Lowpass Filter', img_filtered)
cv2.imshow('Ideal Highpass Filter', img_filtered_highpass)
cv2.waitKey(0)
cv2.destroyAllWindows()
