import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread('lena.jpg', 0)

# Perform DFT
dft = np.fft.fft2(img)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(np.abs(dft_shift))

# Plot the magnitude spectrum
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
