import os
import cv2 as cv
import numpy as np

os.chdir('test')
image_list = os.listdir('./')
data_num = 3000
vector_num = 64
rank_num = 6
query_id = 373
max_dist = 999
color_histograms = np.zeros((data_num, vector_num), np.float32)
file_names = []

for i, file_name in enumerate(image_list):
    file_names.append(file_name)
    color_histogram = np.zeros((4, 4, 4), np.float32)
    print(str(i) + ':' + file_name)
    img = cv.imread(file_name)
    Height = img.shape[0]
    Width = img.shape[1]

    for y in range(Height):
        for x in range(Width):
            b = img[y, x, 0] // vector_num
            g = img[y, x, 1] // vector_num
            r = img[y, x, 2] // vector_num
            color_histogram[r, b, g] += 1

    color_histogram = color_histogram.reshape((1, vector_num))
    color_histogram = color_histogram / (Height * Width)
    color_histograms[i, :] = color_histogram[0, :]

print('query: ' + file_names[query_id])
dists = np.ones((data_num), np.float32) * max_dist

for i in range(data_num):
    v = color_histograms[query_id] - color_histograms[i]
    dists[i] = np.linalg.norm(v, ord=1)

for r in range(rank_num):
    min_value = np.min(dists)
    min_index = np.argmin(dists)
    print('rank ' + str(r) + ': ' +
          file_names[min_index] + ', ' + str(min_value))
    dists[min_index] = max_dist

cv.waitKey(0)
cv.destroyAllWindows()
