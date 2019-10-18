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

def emboss_kernel():
    return np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])

def avg_kernel(size = 3):
    return np.ones((size, size), dtype=np.uint8)

def sharpen_kernel():
    return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

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

    img = cv2.imread("???.jpg") # per default loads a BGR color image
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #w, h, ch = img.shape # retrieve width, heigth and the number of channels

    cv2.imshow("img", convolution_2d(img_grey, median_kernel())) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()