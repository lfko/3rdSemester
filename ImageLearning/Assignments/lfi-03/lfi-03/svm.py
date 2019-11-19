import numpy as np
import cv2
import glob
import time
from sklearn import svm


############################################################
#
#              Support Vector Machine
#              Image Classification
#              A3.1
#
############################################################

def createKeypoints(w = 256, h = 256, kpSize = 15):
    return [cv2.KeyPoint(x, y, kpSize) for x in range(w) for y in range(h)]

def extractSIFTFeatures(keypoints, images):
    sift = cv2.xfeatures2d.SIFT_create()
    return [sift.compute(cv2.imread(img, 0), keypoints) for img in images]


# 1. Implement a SIFT feature extraction for a set of training images ./images/db/train/** (see 2.3 image retrieval)
# use 256x256 keypoints on each image with subwindow of 15x15px
img_path = './images/db'
images = glob.glob(img_path + '/train/*/*.jpg')
print(len(images), 'images loaded!')

t0 = time.time()

keypoints = createKeypoints(1, 1)
descriptors = []
descriptors = extractSIFTFeatures(keypoints, images)

t1 = time.time()

print(t1-t0)
print(len(descriptors), "descriptors extracted!")
print(descriptors[19][1].shape)
# 2. each descriptor (set of features) need to be flattened in one vector
# That means you need a X_train matrix containing a shape of (num_train_images, num_keypoints*num_entry_per_keypoint)
# num_entry_per_keypoint = histogram orientations as talked about in class
# You also need a y_train vector containing the labels encoded as integers
X_train = np.empty((20, 128))
for i in range(len(images)):
    X_train = np.hstack(descriptors[i][1].flatten())

print(X_train.shape)
print(X_train[1,])

# class encoding
# car = 0
# face = 1
# flower = 2
Y_train = np.asarray([0 if img.find('car') > 0 else 1 if img.find('face') > 0 else 2 for img in images])
#print(Y_train)

# 3. We use scikit-learn to train a SVM classifier - however you need to test with different kernel options to get
# good results for our dataset.


# 4. We test on a variety of test images ./images/db/test/ by extracting an image descriptor
# the same way we did for the training (except for a single image now) and use .predict()
# to classify the image
#images_test = glob.glob(img_path + '/test/*/*.jpg')
#descriptors_test = extractSIFTFeatures(keypoints, images_test[0])

# 5. output the class + corresponding name
