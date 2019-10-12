import numpy as np
import cv2


def im2double(im):
    """
    Converts uint image (0-255) to double image (0.0-1.0) and generalizes
    this concept to any range.

    :param im:
    :return: normalized image
    """
    min_val = np.min(im.ravel()) # flatten the array to a 1-d and get the min value
    max_val = np.max(im.ravel()) # flatten the array to a 1-d and get the max value
    out = (im.astype('float') - min_val) / (max_val - min_val) 
    # .astype('float') basically converts the values to floats
    return out


def make_gaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.

    returns a size-by-size kernel, filled with gaussian created values
    
    @NB:- GK is a low-pass filter kernel; good for removing noise. High frequency content (edges, noise) is removed/blurred
        - 
        - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html#filtering
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    # set the center to e.g. x=5, y=5 for size=11
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    k = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    return k / np.sum(k)


def convolution_2d(img, kernel):
    """
    Computes the convolution between kernel and image

    :param img: grayscale image
    :param kernel: convolution matrix - 3x3, or 5x5 matrix
    :return: result of the convolution

    @NB: http://machinelearninguru.com/computer_vision/basics/convolution/image_convolution_1.html
    """
    # TODO write convolution of arbritrary sized convolution here
    # Hint: you need the kernelsize

    offset = int(kernel.shape[0]/2)
    newimg = np.zeros(img.shape) # picture blank, filled with zeroes, sized liked the original image
    # TODO ??? Padding ???
    # iterate pixel by pixel
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            newimg[x, y] = (kernel * img[y:y + 11, x:x + 11]).sum()

    return newimg


if __name__ == "__main__":

    filename = "Lenna.png"
    # 1. load image in grayscale
    img_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # (512, 512)
    img_col = cv2.imread(filename, cv2.IMREAD_COLOR) # (512, 512, 3)

    # cv2.cvtColor does the color conversion from gray to color
    final_img = np.concatenate((cv2.cvtColor(img_gray,cv2.COLOR_GRAY2RGB, img_col)), axis=1)
    #print(final_img.shape, final_img.size)

    # 2. convert image to 0-1 image (see im2double)
    img = im2double(img_gray)

    # image kernels
    # for calculating the gradients
    sobelmask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelmask_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gk = make_gaussian(11)
    print(gk)
    # 3 .use image kernels on normalized image
    conv_img = convolution_2d(img, gk)

    # 4. compute magnitude of gradients

    # Show resulting images
    '''
    cv2.imshow("sobel_x", sobel_x)
    cv2.imshow("sobel_y", sobel_y)
    '''
    cv2.imshow("mog", conv_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    