import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
from PIL import Image

# Load image
image = Image.open("lena.jpg")

# Convert image to grayscale
image = image.convert("L")

# Convert image to numpy array
image_array = np.array(image)

# Apply FFT to image array
fft2 = fftpack.fft2(image_array)

# Shift the zero-frequency component to the center of the spectrum
fft2_shifted = fftpack.fftshift(fft2)

# Compute the magnitude of the spectrum
magnitude_spectrum = 20*np.log(np.abs(fft2_shifted))

# Display the original image and the magnitude spectrum
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Input Image')
ax[0].axis('off')
ax[1].imshow(magnitude_spectrum, cmap='gray')
ax[1].set_title('Magnitude Spectrum')
ax[1].axis('off')
plt.show()
