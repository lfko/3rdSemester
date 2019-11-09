import numpy as np
import cv2
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

###############################################################
#
# Write your own descriptor / Histogram of Oriented Gradients
# A2.2
#
###############################################################
feature_params = dict(maxCorners=500,qualityLevel=0.01,minDistance=10)

def plot_histogram(hist, bins):
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()


def compute_simple_hog(imgcolor, keypoints):

    # convert color to gray image and extract feature in gray
    imggray = cv2.cvtColor(imgcolor, cv2.COLOR_RGB2GRAY)

    # compute x and y gradients (sobel kernel size 5)
    x_grads = cv2.Sobel(imggray, cv2.CV_32FC1, dx = 1, dy = 0, ksize = 5)
    y_grads = cv2.Sobel(imggray, cv2.CV_32FC1, dx = 0, dy = 1, ksize = 5)

    print(x_grads.shape)
    print(y_grads.shape)

    # compute magnitude and angle of the gradients
    phase = cv2.phase(x_grads, y_grads, angleInDegrees = False)
    magnitude = cv2.magnitude(x_grads, y_grads)

    # go through all keypoints and and compute feature vector
    descr = np.zeros((len(keypoints), 8), np.float32)
    count = 0
    for kp in keypoints:
        # print kp.pt, kp.size
        print(kp.pt, kp.size)
        # extract angle in keypoint sub window
        x_sub, y_sub = kp.pt

        block_edge_size = int(kp.size/2)+1 # +1 for the offset, since we are looking at an ellipsoid 
        angles = phase[int(x_sub-block_edge_size):int(x_sub+block_edge_size),
                        int(y_sub-block_edge_size):int(y_sub+block_edge_size)]

        # extract gradient magnitude in keypoint subwindow
        mags = magnitude[int(x_sub-block_edge_size):int(x_sub+block_edge_size),
                        int(y_sub-block_edge_size):int(y_sub+block_edge_size)]

        # create histogram of angle in subwindow BUT only where magnitude of gradients is non zero! Why? Find an
        # answer to that question use np.histogram
        angles = angles[mags > .0]
        #bins = np.arange(angles.min(), np.ceil(angles.max()), step = angles.max()/8) 
        (hist, bins) = np.histogram(angles, bins = 8, density = True, range = (0, 2*np.pi)) # we are using angles so max range with 2*PI

        plot_histogram(hist, bins)

        descr[count] = hist

    return descr

# creates a KP at x = 15, y = 15, size = 11
keypoints = [cv2.KeyPoint(15, 15, 11)]

# test for all test images
# all of them look good except for the vert.jpg
test = cv2.imread('./images/hog_test/circle.jpg')
descriptor = compute_simple_hog(test, keypoints)