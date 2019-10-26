import numpy as np
import cv2
import math
import sys

import random

'''
    @NB: https://lmcaraig.com/color-quantization-using-k-means
'''


############################################################
#
#                       KMEANS
#
############################################################

# implement distance metric - e.g. squared distances between pixels
def distance(a, b):
    return np.linalg.norm(a-b) # basic euclidean distance
    

# k-means works in 3 steps
# 1. initialize
# 2. assign each data element to current mean (cluster center)
# 3. update mean
# then iterate between 2 and 3 until convergence, i.e. until ~smaller than 5% change rate in the error

def update_mean(img, clustermask):
    """This function should compute the new cluster center, i.e. numcluster mean colors"""

    for k in range(numclusters):
        # check if there are any pixels at all belonging to this cluster
        if len(img[clustermask[:, :, 0] == k]) == 0:
            continue
        # using the cluster assignments for boolean indexing
        clust_mean = np.mean(img[clustermask[:, :, 0] == k], axis = 0)
        current_cluster_centers[k, 0] = np.asarray(clust_mean)

    print('updated cluster_centers: ')
    print(current_cluster_centers)

def assign_to_current_mean(img, result, clustermask):
    """The function expects the img, the resulting image and a clustermask.
    After each call the pixels in result should contain a cluster_color corresponding to the cluster
    it is assigned to. clustermask contains the cluster id (int [0...num_clusters]
    Return: the overall error (distance) for all pixels to there closest cluster center (mindistance px - cluster center).
    """
    overall_dist = 0

    for row in range(h1):
        for col in range(w1):
            dist_to_c = {} # {"cluster", "distance"}
            for k in range(numclusters):
                # calculate distance to the colors
                d = distance(img[row, col], current_cluster_centers[k, 0])
                # save them in a dictionary, with k as key and the distance to c as value
                dist_to_c[k] = d

            # smallest distance wins, i.e. pixel will be assigned to this cluster
            # cluster assignment (idx)
            new_clust = min(dist_to_c, key = dist_to_c.get)
            clustermask[row, col] = new_clust
            # colour assignment
            result[row, col] = current_cluster_centers[new_clust, 0]

            # add the min distance to the overall count
            overall_dist += dist_to_c.get(new_clust)

    return overall_dist



def initialize(img):
    """inittialize the current_cluster_centers array for each cluster with a random pixel position"""

    for k in range(numclusters):
        col = cluster_colors[np.random.randint(0, len(cluster_colors))]
        current_cluster_centers[k, 0] = np.asarray(col)
        cluster_colors.remove(col) # remove selected color afterwards to avoid having same cluster centers
        
    print(current_cluster_centers)


def kmeans(img):
    """Main k-means function iterating over max_iterations and stopping if
    the error rate of change is less then 2% for consecutive iterations, i.e. the
    algorithm converges. In our case the overall error might go up and down a little
    since there is no guarantee we find a global minimum.
    """
    max_iter = 10
    max_change_rate = 0.02
    dist = sys.float_info.max
    prev_dist = sys.float_info.max

    clustermask = np.zeros((h1, w1, 1), np.uint8)
    result = np.zeros((h1, w1, 3), np.uint8)

    initialize(img)
    # initial cluster assignment
    prev_dist = assign_to_current_mean(img, result, clustermask)
    update_mean(img, clustermask)

    for i in range(max_iter):

        # calculate total variance and check if change to the previous iteration
        # is below the threshold
        dist = assign_to_current_mean(img, result, clustermask)
        update_mean(img, clustermask)

        print('overall dist after iteration', i, ': ', dist)
        curr_change_rate = np.abs((dist - prev_dist)/prev_dist) 
        print(curr_change_rate)
        if curr_change_rate < max_change_rate:
            break

        prev_dist = dist

    return result

# num of cluster
numclusters = 3
# corresponding colors for each cluster
cluster_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 255], [0, 0, 0], [128, 128, 128]]
# initialize current cluster centers (i.e. the pixels that represent a cluster center)
current_cluster_centers = np.zeros((numclusters, 1, 3), np.float32)

# load image
imgraw = cv2.imread('Lenna.png')
scaling_factor = 0.5
imgraw = cv2.resize(imgraw, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

image = imgraw # BGR
#image = cv2.cvtColor(imgraw, cv2.COLOR_BGR2HSV)
#image = cv2.cvtColor(imgraw, cv2.COLOR_BGR2LAB)
#image = cv2.cvtColor(imgraw, cv2.COLOR_BGR2YUV)

h1, w1 = image.shape[:2]

# execute k-means over the image
# it returns a result image where each pixel is color with one of the cluster_colors
# depending on its cluster assignment
res = kmeans(image)

h1, w1 = res.shape[:2]
h2, w2 = image.shape[:2]
vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
vis[:h1, :w1] = res
vis[:h2, w1:w1 + w2] = image

cv2.imshow("Color-based Segmentation Kmeans-Clustering", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()