import numpy as np
import cv2
import glob
import re
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


# 1. Implement a SIFT feature extraction for a set of training images ./images/db/train/** (see 2.3 image retrieval)
# use 256x256 keypoints on each image with subwindow of 15x15px
sift = cv2.xfeatures2d.SIFT_create()
img_path = './images/db'
images_train = glob.glob(img_path + '/train/*/*.jpg')
print(len(images_train), 'images loaded!')

keypoints = createKeypoints(256, 256)

descriptors = []
for i, img in enumerate(images_train):
    _, des = sift.compute(cv2.imread(img, 0), keypoints)
    descriptors.append(des)

print(len(descriptors), "descriptors extracted!")

# 2. each descriptor (set of features) need to be flattened in one vector
# That means you need a X_train matrix containing a shape of (num_train_images, num_keypoints*num_entry_per_keypoint)
# num_entry_per_keypoint = histogram orientations as talked about in class
# You also need a y_train vector containing the labels encoded as integers
X_train = np.zeros((20, 256 * 256 * 128)) # 128 because total of bin values
for i in range(len(images_train)):
    X_train[i, :] = np.ravel(descriptors[i])

print('X_train.shape: ', X_train.shape)
#print(X_train[1,])

# class encoding
# car = 0
# face = 1
# flower = 2
Y_train = np.asarray([0 if img.find('car') > 0 else 1 if img.find('face') > 0 else 2 for img in images_train])
classes = {0: 'car', 1 : 'face', 2 : 'flower'}
print(Y_train)

# 3. We use scikit-learn to train a SVM classifier - however you need to test with different kernel options to get
# good results for our dataset.
svm = svm.SVC(kernel = "linear")
svm.fit(X_train, Y_train)

# 4. We test on a variety of test images ./images/db/test/ by extracting an image descriptor
# the same way we did for the training (except for a single image now) and use .predict()
# to classify the image
images_test = glob.glob(img_path + '/test/*.jpg')
print(len(images_test), 'images loaded!')
print("\n")

# exctract features of the test picture
descriptors_test = []
true_classes = []
for img in images_test:
    true_class = re.findall('\w+.jpg', img)[0].replace('.jpg', '')
    true_class = re.sub('\d', '', true_class)
    true_classes.append(true_class)

    _, des = sift.compute(cv2.imread(img, 0), keypoints)
    descriptors_test.append(des)

X_test = np.zeros((4, 256 * 256 * 128))
for i in range(len(descriptors_test)):
    X_test[i, :] = np.ravel(descriptors_test[i])

# 5. output the class + corresponding name
y_pred = svm.predict(X_test)
for i, p in enumerate(y_pred):
    print('True class:', true_classes[i])
    print("Class '", classes[y_pred[i]], "' predicted")
