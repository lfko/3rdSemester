'''
    @author lfko
'''

import cv2
import numpy as np

def median_kernel():
    '''
        apply the median kernel
    '''
    return np.array([[0, -0.25, 0], [-0.25, 1, -0.25], [0, -0.25, 0]])

def mean_kernel():
    return np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]])

def hpf_kernel():
    return np.array([np.repeat(-1, 5),
                    [-1, 1, 2, 1, -1],
                    [-1, 2, 4, 2, -1],
                    [-1, 1, 2, 1, -1],
                    np.repeat(-1,5 )])

def emboss_kernel():
    return np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])

def avg_kernel(size = 3):
    return np.ones((size, size), dtype=np.uint8)

def sharpen_kernel():
    return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

def gaussian_blur(X, sigma = 1, mu = 0):
    Y = (1/(sigma * np.sqrt(2*np.pi)))*np.exp((-(X - mu)**2)/(2*sigma**2))
    Y = Y.reshape((3, 3))
    print(Y, Y.shape)

    return Y

def convolution_2d(img, kernel):

    imgRow, imgCol = img.shape
    kernRow, kernCol = kernel.shape

    offset = int(kernRow/2) # center of the kernel
    newimg = np.zeros(img.shape) # picture blank, filled with zeroes, sized liked the original image
    
    pad_height = int((kernRow - 1) / 2) # added area of padding in y direction
    pad_width = int((kernCol - 1) / 2) # added area of padding in X direction
    
    img_pad = np.zeros((imgRow + (2 * pad_height), imgCol + (2 * pad_width)))
    img_pad[pad_height:img_pad.shape[0] - pad_height, pad_width:img_pad.shape[1] - pad_width] = img # set the non-padded area with our original picture

    for row in range(imgRow):
        for col in range(imgCol):
            newimg[row, col] = np.dot(kernel, img_pad[row:row + kernRow, col:col + kernCol]).sum()

    return newimg

if __name__ == "__main__":

    img = cv2.imread("Johanna.jpg") # per default loads a BGR color image
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img_grey.shape)
    retval, img_grey_th = cv2.threshold(img_grey, 120, 255, cv2.THRESH_BINARY)

    #w, h, ch = img.shape # retrieve width, heigth and the number of channels
    img_kernel = convolution_2d(img_grey, mean_kernel())
    blurred = cv2.GaussianBlur(img_grey, (11, 11), 0)
    print(blurred.shape)
    cv2.imshow("greyth", img_grey_th)
    cv2.imshow("img_kernel", img_kernel) # HPF
    #cv2.imshow("blurred", blurred)
    #cv2.imshow("g_hpf", img_kernel - blurred) # LPF?
    
    X = np.arange(0, 9, 1, float)
    Y = gaussian_blur(X)
    new_img = cv2.filter2D(img_grey, -1, mean_kernel())
    cv2.imshow("org_img", img)
    cv2.imshow("new_img", new_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()