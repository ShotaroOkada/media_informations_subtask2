import os
import cv2 as cv
import numpy as np

os.chdir('test')
image_list = os.listdir('./')
chists = np.zeros((3000, 64), np.float32)
fnames = []

for i, fname in enumerate(image_list):
    fnames.append(fname)
    hist = np.zeros((4, 4, 4), np.float32)
    print(str(i)+':'+fname)
    img = cv.imread(fname)
    Height = img.shape[0]
    Width = img.shape[1]
    for y in range(Height):
        for x in range(Width):
            b = img[y, x, 0]//64
            g = img[y, x, 1]//64
            r = img[y, x, 2]//64
            hist[r, b, g] += 1

    hist = hist.reshape((1, 64))
    hist = hist/(Height*Width)
    chists[i, :] = hist[0, :]

query_id = 373
print('query: '+fnames[query_id])
dists = np.ones((3000), np.float32)*999

for i in range(3000):
    v = chists[query_id]-chists[i]
    dists[i] = np.linalg.norm(v, ord=1)
for r in range(6):
    minval = np.min(dists)
    minidx = np.argmin(dists)
    #print('rank '+str(r)+': '+str(minidx)+' '+fnames[minidx]+', '+str(minval))
    print('rank '+str(r)+': '+fnames[minidx]+', '+str(minval))
    dists[minidx] = 999

cv.waitKey(0)
cv.destroyAllWindows()
