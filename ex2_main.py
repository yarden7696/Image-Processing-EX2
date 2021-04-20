import numpy as np
from ex2_utils import *
import matplotlib.pyplot as plt

from matala2 import ex2_utils


def conv1Demo():
    print("numpy result:", np.convolve(np.array([1, 2, 3]), np.array([1,2,1])/4, "full"))
    print("My result  : ", conv1D(np.array([1, 2, 3]), np.array([1,2,1])/4))


def conv2Demo():
    krnl = np.ones(shape=(5, 5)) * (1 / 25)
    img_path = cv2.imread("boxman.jpg", cv2.IMREAD_GRAYSCALE)
    numpyRes = cv2.filter2D(img_path, -1, krnl, borderType=cv2.BORDER_REPLICATE)
    myRes = conv2D(img_path, krnl)

    fig, axes = plt.subplots(1, 2)  # 1 row and 2 cols
    fig.suptitle('Convolution2D', fontsize=29)
    axes[0].set_title('my result')
    axes[0].imshow(myRes, cmap="gray")
    axes[1].set_title('numpy result')
    axes[1].imshow(numpyRes, cmap="gray")
    plt.show()




def derivDemo():

    img = cv2.imread("boxman.jpg", cv2.IMREAD_GRAYSCALE)
    directions, magnitude, im_derive_x, im_derive_y = convDerivative(img)

    plt.gray()
    fig, axes = plt.subplots(2, 2)  # 2 row and 2 cols
    axes[0][0].set_title('Original'),axes[0][0].imshow(img)
    axes[0][1].set_title('Magnitude'),axes[0][1].imshow(magnitude)
    axes[1][0].set_title('Derivative X'),axes[1][0].imshow(im_derive_x)
    axes[1][1].set_title('Derivative Y'),axes[1][1].imshow(im_derive_y)
    plt.show()



# blurDemo()
# edgeDemo()
# houghDemo()


if __name__ == '__main__':

        print("conv1D-")
        conv1Demo()
        print(" ")
        conv2Demo()
        print(" ")
        derivDemo()
