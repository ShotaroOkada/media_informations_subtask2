from PIL import Image
import numpy as np

img = Image.open('terasawa.jpg')
img = img.convert("L")
img = img.resize((256, 256))
img = np.array(img)
w, h = img.shape

for y in range(h):
    for x in range(w):
        if x > h/2:
            img[y, x] = 255 - img[y, x]
        else:
            img[y, x] = img[y, x]/2

img = Image.fromarray(img)
img.show()
