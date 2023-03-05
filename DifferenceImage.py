import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('img01.jpg')
img2 = cv2.imread('img02.jpeg')

img1 = cv2.resize(img1, (640, 480))
img2 = cv2.resize(img2, (640, 480))

if img1.shape[2] == 3 and img2.shape[2] == 3:
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

diff = cv2.absdiff(img1, img2)

cv2.imshow('Difference Image', diff)

hist = cv2.calcHist([diff], [0], None, [256], [0, 256])
plt.plot(hist)
plt.xlim([0, 256])
plt.title('Histogram of Difference Image')
plt.show()

cv2.imwrite('diff.jpg', diff)

cv2.waitKey(0)
cv2.destroyAllWindows()