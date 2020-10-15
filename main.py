import cv2
import matplotlib.pyplot as plt
import numpy as np


def identity(image):
    h, w = image.shape[:2]
    src = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], np.float32)
    affine = cv2.getAffineTransform(src, src)
    return cv2.warpAffine(image, affine, (w, h))


def rotate(image, angle):
    h, w = image.shape[:2]
    affine = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
    return cv2.warpAffine(image, affine, (w, h))


if __name__ == "__main__":
    image = cv2.imread("terasawa.jpg")[:, :, ::-1]
    converted = rotate(image, 20)
    plt.imshow(converted)
    plt.title("Identity")
    plt.show()
