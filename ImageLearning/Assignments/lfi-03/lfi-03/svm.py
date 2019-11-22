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
    '''
        TODO there must still be a bug here; seems images is handled as a list of strings and the length is incorrect
    '''
    sift = cv2.xfeatures2d.SIFT_create()
    return [sift.compute(cv2.imread(img, 0), keypoints) for img in images]


# 1. Implement a SIFT feature extraction for a set of training images ./images/db/train/** (see 2.3 image retrieval)
# use 256x256 keypoints on each image with subwindow of 15x15px
img_path = './images/db'
images_train = glob.glob(img_path + '/train/*/*.jpg')
print(len(images_train), 'images loaded!')

t0 = time.time()

keypoints = createKeypoints(1, 1)
descriptors = []
descriptors = extractSIFTFeatures(keypoints, images_train)

t1 = time.time()

print(t1-t0)
print(len(descriptors), "descriptors extracted!")

# 2. each descriptor (set of features) need to be flattened in one vector
# That means you need a X_train matrix containing a shape of (num_train_images, num_keypoints*num_entry_per_keypoint)
# num_entry_per_keypoint = histogram orientations as talked about in class
# You also need a y_train vector containing the labels encoded as integers
X_train = np.empty((20, 128))
for i in range(len(images_train)):
    X_train[i] = descriptors[i][1]

#print(X_train.shape)
#print(X_train[1,])

# class encoding
# car = 0
# face = 1
# flower = 2
Y_train = np.asarray([0 if img.find('car') > 0 else 1 if img.find('face') > 0 else 2 for img in images_train])
classes = {0: 'car', 1 : 'face', 2 : 'flower'}
print(Y_train)
print(images_train)

# 3. We use scikit-learn to train a SVM classifier - however you need to test with different kernel options to get
# good results for our dataset.
svm = svm.SVC(kernel = "linear", C = 1E10) # TODO use another kernel
svm.fit(X_train, Y_train)

# 4. We test on a variety of test images ./images/db/test/ by extracting an image descriptor
# the same way we did for the training (except for a single image now) and use .predict()
# to classify the image
images_test = glob.glob(img_path + '/test/flower.jpg')
print(len(images_test), 'images loaded!')

# exctract features of the test picture
# TODO use defined function
#descriptors_test = extractSIFTFeatures(keypoints, images_test[0]) # car.jpg
sift = cv2.xfeatures2d.SIFT_create()
descriptors_test = sift.compute(cv2.imread(images_test[0], 0), keypoints)
print(len(descriptors_test), "descriptors for the test extracted")

X_test = np.empty((1, 128))
X_test[0] = descriptors_test[1]

y_pred = svm.predict(X_test)
# 5. output the class + corresponding name
print(y_pred, "Class '", classes[y_pred[0]], "' predicted")
