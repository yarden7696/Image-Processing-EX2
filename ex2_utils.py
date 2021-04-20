import numpy as np
import cv2


def conv1D(inSignal: np.ndarray, kernel1: np.ndarray) -> np.ndarray:
    """
 Convolve a 1-D array with a given kernel
 :param inSignal: 1-D array
 :param kernel1: 1-D array as a kernel
 :return: The convolved array
  """
    revKernel1 = kernel1[::-1]  # reverse to the kernel
    addZeros = np.zeros(kernel1.size - 1)  # num of zeros to add both sides of the signal
    newSignal = np.append(addZeros, np.append(inSignal, addZeros))  # adding the zeros both of sides

    ans = np.zeros(inSignal.size + kernel1.size - 1)  # the res vector will be in this size

    for i in range(ans.size):
        ans[i] = np.dot(newSignal[i: i + (kernel1.size - 1) + 1], revKernel1)

    return ans


def conv2D(inImage: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param inImage: 2D image
    :param kernel2: A kernel
    :return: The convolved image
    """

    revKernel2 = np.flip(kernel2)  # Reverse the order of kernel
    ans = np.zeros(inImage.shape)  # init a result array in size inImage

    # caz we are in 2D space, we need to check how many zeros we should adding
    # to the rows and columns of the original image
    lenX = np.floor(revKernel2.shape[0] / 2).astype(int)  # floor takes a lower value of a given number
    if lenX < 1:
        lenX = 1
    lenY = np.floor(revKernel2.shape[1] / 2).astype(int)
    if lenY < 1:
        lenY = 1

    afterZeros = np.pad(inImage, [(lenX,), (lenY,)], mode='constant')  # The actual addition
    res = help2D(lenX, lenY, afterZeros, ans, revKernel2)

    return res


""" Auxiliary function that performs convolution in 2D space, here the kernel is
multiply with the partial signal """
def help2D(lenX: int, lenY: int, afterZeros: np.ndarray, ans: np.ndarray, revKernel2: np.ndarray) -> (np.ndarray):
    for row in range(lenX, afterZeros.shape[0] - lenX):
        for col in range(lenY, afterZeros.shape[1] - lenY):
            begin_row = row - lenX
            end_row = row - lenX + revKernel2.shape[0]
            begin_col = col - lenY
            end_col = col - lenY + revKernel2.shape[1]
            signal_part = afterZeros[begin_row:end_row, begin_col:end_col]
            ans[begin_row, begin_col] = np.sum(np.multiply(signal_part, revKernel2))
    return ans





def convDerivative(inImage: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param inImage: Grayscale iamge
    :return: (directions, magnitude,x_der,y_der)
    """

    blurImg = cv2.GaussianBlur(inImage, (5, 5), 1)  # smoothing before derivative using gaussian filter

    xKrnl = np.array([-1, 0, 1]).reshape((1, 3))
    yKrnl = xKrnl.reshape((3, 1))

    # After i blurred the image i need to calc partial derivative of X and Y
    xDerive = cv2.filter2D(blurImg, -1, xKrnl, borderType=cv2.BORDER_REPLICATE)
    yDerive = cv2.filter2D(blurImg, -1, yKrnl, borderType=cv2.BORDER_REPLICATE)

    # According to the formulas we learned in the class
    mag = np.sqrt(np.square(xDerive) + np.square(yDerive)).astype('uint8')
    directions = np.arctan2(yDerive, xDerive) * 180 / np.pi

    return directions, mag, xDerive, yDerive




def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7)-> (np.ndarray, np.ndarray):
    """
    Detects edges using the Sobel method
    :param img: Input image
    :param thresh: The minimum threshold for the edge response
    :return: opencv solution, my implementation
    """

    sobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    derivativeX = conv2D(img, sobelX)
    derivativeY = conv2D(img, sobelY)
    myMag = np.sqrt(np.square(sobelX) + np.square(sobelY))

