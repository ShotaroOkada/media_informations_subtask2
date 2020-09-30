from PIL import Image
import numpy as np

img = Image.open('terasawa.jpg')
imgArray = np.array(img)
imgArray[100:300, 100:300, :] = 0
img = Image.fromarray(imgArray)
img.show()
