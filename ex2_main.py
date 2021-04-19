import numpy as np
from ex2_utils import *
import matplotlib.pyplot as plt

from matala2 import ex2_utils


def conv1Demo():

    krnl = np.array([1,2,1])/4
    sgnl = np.array([1, 2, 3])
    print("numpy: ", np.convolve(sgnl, krnl, "full"), " my result: ", conv1D(sgnl, krnl))


def conv2Demo():
    krnl = np.ones(shape=(5, 5)) * (1 / 25)
    img_path = cv2.imread("boxman.jpg", cv2.IMREAD_GRAYSCALE)
    numpyRes = cv2.filter2D(img_path, -1, krnl, borderType=cv2.BORDER_REPLICATE)
    myRes = conv2D(img_path, krnl)
    print("numpy: ", numpyRes)
    print(" ")
    print("my result: ", myRes)

    plt.gray()
    plt.imshow(numpyRes)
    plt.show()
    plt.imshow(myRes)
    plt.show()


#def derivDemo():  this is hila test

    # img = cv2.imread("boxman.jpg", cv2.IMREAD_GRAYSCALE)
    # directions, magnitude, im_derive_x, im_derive_y = convDerivative(img)
    #
    # plt.gray()
    # plt.subplot(2, 2, 1), plt.imshow(img)
    # plt.title('Original'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 2, 2), plt.imshow(magnitude)
    # plt.title('Magnitude'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 2, 3), plt.imshow(im_derive_x)
    # plt.title('Derivative X'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 2, 4), plt.imshow(im_derive_y)
    # plt.title('Derivative Y'), plt.xticks([]), plt.yticks([])
    # plt.show()



# blurDemo()
# edgeDemo()
# houghDemo()


if __name__ == '__main__':
        print("conv1D-")
        conv1Demo()
        print(" ")
        print("conv2D-...(לבדוק למה יצאו לי מצריצות שונות(")
        conv2Demo()
        print("")
        #derivDemo()
