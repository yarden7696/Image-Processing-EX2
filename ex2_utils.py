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


def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7) -> (np.ndarray, np.ndarray):
    """
    Detects edges using the Sobel method
    :param img: Input image
    :param thresh: The minimum threshold for the edge response
    :return: opencv solution, my implementation
    """
    sobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    derivativeX = conv2D(img, sobelX)  # Blurring in the Y direction and derivative in the X direction
    derivativeY = conv2D(img, sobelY)  # Blurring in the X direction and derivative in the Y direction
    myMag = np.sqrt(np.square(derivativeX) + np.square(derivativeY))  # cal mag according the formula
    myMag[myMag < thresh * 255] = 0
    myMag[myMag >= thresh * 255] = 1

    # Blurring in the Y direction and derivative in the X direction and opposite using cv2.Sobel
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    cvMag = cv2.magnitude(grad_x, grad_y)  # cal mag according the formula
    cvMag[cvMag < thresh * 255] = 0
    cvMag[cvMag >= thresh * 255] = 1
    return cvMag, myMag


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> (np.ndarray):
    """
    Detecting edges using the "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    krnl_LPLCAN = np.array([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]])
    img = conv2D(img, krnl_LPLCAN)  # laplacian filter
    zeroCrossing = np.zeros(img.shape)  # create the final ans of the edges matrix

    # Check for a zero crossing around (x,y)
    for i in range(img.shape[0] - (krnl_LPLCAN.shape[0] - 1)):
        for j in range(img.shape[1] - (krnl_LPLCAN.shape[1] - 1)):
            cellij = img[i][j]
            cellPj = img[i][j + 1]
            cellMj = img[i][j - 1]
            cellPi = img[i + 1][j]
            cellMi = img[i - 1][j]
            if cellij == 0:
                if (cellPj > 0 and cellMj < 0) or (cellPj < 0 and cellMj < 0) or \
                        (cellPi > 0 and cellMi < 0) or (cellPi < 0 and cellMi > 0):
                    zeroCrossing[i][j] = 255
            if cellij < 0:
                if (cellPj > 0) or (cellMj > 0) or (cellPi > 0) or (cellMi > 0):
                    zeroCrossing[i][j] = 255
    return zeroCrossing



def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float)-> (np.ndarray, np.ndarray):
    """
    Detecting edges usint "Canny Edge" method
    :param img: Input image
    :param thrs_1: T1
    :param thrs_2: T2
    :return: opencv solution, my implementation
    """

    cv2Res = cv2.Canny(cv2.GaussianBlur(img, (5, 5), 0), thrs_1 * 255, thrs_2 * 255)  # cv2 canny algorithm

    sobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img = cv2.GaussianBlur(img, (5, 5), 0)  # Blurring the image using Gaussian
    grad_x = conv2D(img, sobelX)  # Calculation of a partial derivative of X
    grad_y = conv2D(img, sobelY)  # Calculation of a partial derivative of Y
    myMag = np.sqrt(np.square(grad_x) + np.square(grad_y)) # cal mag according the formula
    myDirections = (np.rad2deg(np.arctan2(grad_y, grad_x)) % 180)
    #  round the angel of the gradient
    myDirections[(157.5 <= myDirections) | (myDirections < 22.5)] = 0
    myDirections[(myDirections < 67.5) & (22.5 <= myDirections)] = 45
    myDirections[(myDirections < 112.5) & (67.5 <= myDirections)] = 90
    myDirections[(myDirections < 157.5) & (112.5 <= myDirections)] = 135

    # In the following conditions we run all over the pixels and check whether its power is stronger
    # than one pixel to the right and one pixel to the left.If so we will save it in the answer matrix
    result = np.zeros(myMag.shape)
    for i in range(1, myMag.shape[0] - 1):
        for j in range(1, myMag.shape[1] - 1):
            if myDirections[i, j] == 0:
                if (myMag[i, j] > myMag[i, j - 1]) and (myMag[i, j] > myMag[i, j + 1]):
                    result[i, j] = myMag[i, j]
            elif myDirections[i, j] == 45:
                if (myMag[i, j] > myMag[i - 1, j + 1]) and (myMag[i, j] > myMag[i + 1, j - 1]):
                    result[i, j] = myMag[i, j]
            elif myDirections[i, j] == 90:
                if (myMag[i, j] > myMag[i - 1, j]) and (myMag[i, j] > myMag[i + 1, j]):
                    result[i, j] = myMag[i, j]
            elif myDirections[i, j] == 135:
                if (myMag[i, j] > myMag[i - 1, j - 1]) and (myMag[i, j] > myMag[i + 1, j + 1]):
                    result[i, j] = myMag[i, j]

    result = hysteresis(result, thrs_1 * 255, thrs_2 * 255)

    return cv2Res, result

""" Helper function that checks all the pixels between T1 and T2 """
def hysteresis(img, weak, strong):
    img_height, img_width = img.shape
    ans = np.zeros((img_height, img_width))
    ans[img < weak] = 0  # set all pixels that smaller than T2 to 0
    ans[img >= strong] = 255  # set all pixels that grater than T1 to 255
    #  All the pixels in weak XY are between T1 and T2 so we need to check if each of them is an edge.
    weakXY = np.argwhere((img <= strong) & (img >= weak))
    for x, y in weakXY:
        ans[x, y] = 255 if 255 in [img[x - 1, y - 1], img[x - 1, y],
            img[x - 1, y + 1], img[x, y - 1],
            img[x, y + 1], img[x + 1, y - 1],
            img[x + 1, y], img[x + 1, y + 1]] else 0

    return ans



