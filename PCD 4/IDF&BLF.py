import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate the FFT of the image
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Create the ideal lowpass filter
rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)
ideal_radius = 30
ideal_lowpass = np.zeros((rows, cols), np.uint8)
cv2.circle(ideal_lowpass, (ccol, crow), ideal_radius, 255, -1)

# Apply the ideal lowpass filter
f_ideal = fshift * ideal_lowpass
f_ideal_shift = np.fft.ifftshift(f_ideal)
img_ideal = np.fft.ifft2(f_ideal_shift)
img_ideal = np.abs(img_ideal)

# Create the Butterworth lowpass filter
n = 2
butter_radius = 30
butter_lowpass = np.zeros((rows, cols), np.uint8)
for i in range(rows):
    for j in range(cols):
        d = np.sqrt((i-crow)**2 + (j-ccol)**2)
        butter_lowpass[i,j] = 1 / (1 + (d/butter_radius)**(2*n))

# Apply the Butterworth lowpass filter
f_butter = fshift * butter_lowpass
f_butter_shift = np.fft.ifftshift(f_butter)
img_butter = np.fft.ifft2(f_butter_shift)
img_butter = np.abs(img_butter)

# Display the images
plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img_ideal, cmap = 'gray')
plt.title('Ideal Lowpass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(img_butter, cmap = 'gray')
plt.title('Butterworth Lowpass Filter'), plt.xticks([]), plt.yticks([])
plt.show()
