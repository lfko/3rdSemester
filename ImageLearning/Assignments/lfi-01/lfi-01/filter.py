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
    @NB: http://www.adeveloperdiary.com/data-science/computer-vision/applying-gaussian-smoothing-to-an-image-using-python-from-scratch/
    """
    imgRow, imgCol = img.shape
    kernRow, kernCol = kernel.shape

    print('shape kernel', kernel.shape)
    print('shape img', img.shape)

    offset = int(kernRow/2) # center of the kernel
    newimg = np.zeros(img.shape) # picture blank, filled with zeroes, sized liked the original image
    
    pad_height = int((kernRow - 1) / 2) # added area of padding in y direction
    pad_width = int((kernCol - 1) / 2) # added area of padding in X direction
    
    img_pad = np.zeros((imgRow + (2 * pad_height), imgCol + (2 * pad_width)))
    # TODO Verstehen!
    img_pad[pad_height:img_pad.shape[0] - pad_height, pad_width:img_pad.shape[1] - pad_width] = img # set the non-padded area with our original picture

    # iterate pixel by pixel
    for row in range(imgRow):
        for col in range(imgCol):
            newimg[row, col] = np.dot(kernel, img_pad[row:row + kernRow, col:col + kernCol]).sum()

    return newimg


if __name__ == "__main__":

    filename = "Lenna.png"
    # 1. load image in grayscale
    img_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # (512, 512)
    img_col = cv2.imread(filename, cv2.IMREAD_COLOR) # (512, 512, 3)

    # cv2.cvtColor does the color conversion from gray to color
    final_img = np.concatenate((cv2.cvtColor(img_gray,cv2.COLOR_GRAY2RGB, img_col)), axis=1)

    # 2. convert image to 0-1 image (see im2double)
    img = im2double(img_gray)

    # image kernels
    # for calculating the gradients
    sobelmask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelmask_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gk = make_gaussian(3)

    # 3 .use image kernels on normalized image
    conv_img = convolution_2d(img, gk)
    sobel_x_img = convolution_2d(conv_img, sobelmask_x)
    sobel_y_img = convolution_2d(conv_img, sobelmask_y)

    # 4. compute magnitude of gradients
    mog = np.sqrt(np.square(sobel_x_img) + np.square(sobel_y_img))

    # Show resulting images
    
    cv2.imshow("conv_img", conv_img)
    cv2.imshow("sobel_x", sobel_x_img)
    cv2.imshow("sobel_y", sobel_y_img)
    cv2.imshow("mog", mog)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    