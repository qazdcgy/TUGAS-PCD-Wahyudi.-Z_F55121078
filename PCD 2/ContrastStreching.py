import numpy
import matplotlib.pyplot as plt
from copy import deepcopy
from PIL import Image

def getGrayColor(rgb) :
    return rgb[0]
def setGrayColor(color) :
    return [color,color, color]
img = Image.open('Lena.png')
img = numpy.asarray(img)
ct = deepcopy(img)
rl = 100
sl = 50
r2 = 150
s2 = 200
for i in range(len(img)):
    for j in range (len(img[i])):
        x = getGrayColor(img[i][j])
        if(0 <= x and x <= rl):
            ct[i][j] = setGrayColor(sl/rl * x)
        elif(rl < x and x <= r2):
            ct[i][j] = setGrayColor(((s2 - sl)/(r2 - rl)) * (x - rl) + sl)
        elif(r2 < x and x <= 255):
            ct[i][j] = setGrayColor(((255 - s2)/(255 - r2)) * (x - r2) + s2)
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.subplot(2, 2, 2)
plt.imshow(ct)
plt.show()