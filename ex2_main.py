import time
import numpy as np
from ex2_utils import *
import matplotlib.pyplot as plt

from matala2 import ex2_utils


def conv1Demo():
    print("numpy result:", np.convolve(np.array([1, 2, 3]), np.array([1,2,1])/4, "full"))
    print("My result  : ", conv1D(np.array([1, 2, 3]), np.array([1,2,1])/4))


def conv2Demo():
    krnl = np.ones(shape=(5, 5)) * (1 / 25)
    img_path = cv2.imread("beach.jpg", cv2.IMREAD_GRAYSCALE)
    numpyRes = cv2.filter2D(img_path, -1, krnl, borderType=cv2.BORDER_REPLICATE)
    myRes = conv2D(img_path, krnl)

    fig, axes = plt.subplots(1, 2)  # 1 row and 2 cols
    fig.suptitle('Convolution2D', fontsize=29)
    plt.gray()
    axes[0].set_title('My Result')
    axes[0].imshow(myRes, cmap="gray")
    axes[1].set_title('Numpy Result')
    axes[1].imshow(numpyRes, cmap="gray")
    plt.show()


def derivDemo():

    img = cv2.imread("boxman.jpg", cv2.IMREAD_GRAYSCALE)
    directions, magnitude, im_derive_x, im_derive_y = convDerivative(img)
    plt.gray()
    fig, axes = plt.subplots(2, 2)  # 2 row and 2 cols
    fig.suptitle('convDerivative', fontsize=20)
    axes[0][0].set_title('Original'),axes[0][0].imshow(img)
    axes[0][1].set_title('Magnitude'),axes[0][1].imshow(magnitude)
    axes[1][0].set_title('Derivative X'),axes[1][0].imshow(im_derive_x)
    axes[1][1].set_title('Derivative Y'),axes[1][1].imshow(im_derive_y)
    plt.show()




def edgeDemo():

    img = cv2.imread("coins.jpg", cv2.IMREAD_GRAYSCALE)

    """edgeDetectionSobel test"""
    cv2_edge_dtcton_sobel, my_edge_dtcton_sobel = edgeDetectionSobel(img)
    fig, axes = plt.subplots(1, 2)  # 1 row and 2 cols
    plt.gray()
    fig.suptitle('edgeDetectionSobel', fontsize=23)
    axes[0].imshow(cv2_edge_dtcton_sobel)
    axes[0].set_title("cv2 Result")
    axes[1].imshow(my_edge_dtcton_sobel)
    axes[1].set_title("My Result")
    plt.show()

    """edgeDetectionZeroCrossingSimple test"""
    edgeDetection_img = edgeDetectionZeroCrossingSimple(img)
    plt.imshow(edgeDetection_img)
    plt.title("edgeDetectionZeroCrossingSimple")
    plt.show()

    """edgeDetectionCanny test"""
    cv2_edge_img, edge_img = edgeDetectionCanny(img, 0.09, 0.05)
    fig, axes = plt.subplots(1, 2)  # 1 row and 2 cols
    fig.suptitle('edgeDetectionCanny', fontsize=23)
    axes[0].imshow(cv2_edge_img)
    axes[0].set_title("cv2 Result")
    axes[1].imshow(edge_img)
    axes[1].set_title("My Result")
    plt.show()


def blurDemo():
    img = cv2.imread("pool_balls.jpeg", cv2.IMREAD_GRAYSCALE)
    myBlur = blurImage1(img,37)
    cv2Blur = blurImage2(img, 37)

    fig, axes = plt.subplots(1, 2)  # 1 row and 2 cols
    plt.gray()
    fig.suptitle('Blurring - Bonus', fontsize=23)
    axes[0].imshow(myBlur)
    axes[0].set_title("blurImage1 - mine")
    axes[1].imshow(cv2Blur)
    axes[1].set_title("blurImage2 - cv2")
    plt.show()



def houghDemo():
    img = cv2.imread("coins.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    lst = ex2_utils.houghCircle(img, 50, 70)
    fig, axes = plt.subplots()
    plt.gray()
    fig.suptitle('houghCircle', fontsize=23)
    axes.imshow(img)
    for i in lst:
        c = plt.Circle((i[0], i[1]), i[2], color='r', fill=False)
        axes.add_artist(c)
    plt.show()


if __name__ == '__main__':
        print("")
        print("MY ID : 207205972")
        print(" ")
        print("'conv1D' - ")
        conv1Demo()
        conv2Demo()
        derivDemo()
        edgeDemo()
        houghDemo()
        blurDemo()

