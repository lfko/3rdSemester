import cv2
import glob
import sys
import numpy as np
from queue import PriorityQueue

############################################################
#
#              Simple Image Retrieval
#              A2.1
#
############################################################


# implement distance function
def distance(a, b):
    return np.linalg.norm(a-b)


def create_keypoints(w, h):
    keypoints = []
    keypointSize = 11

    for x in range(w):
        for y in range(h):
            kp = cv2.KeyPoint(x, y, keypointSize)
            keypoints.append(kp)

    return keypoints


# 1. preprocessing and load
images = glob.glob('./images/db/train/*/*.jpg')
print(len(images), "images are loaded!")

# 2. create keypoints on a regular grid (cv2.KeyPoint(r, c, keypointSize), as keypoint size use e.g. 11)
descriptors = []
keypoints = create_keypoints(256, 256)

# 3. use the keypoints for each image and compute SIFT descriptors
#    for each keypoint. this compute one descriptor for each image.
sift = cv2.xfeatures2d.SIFT_create() # instantiate a SIFT for feature extraction

for image in images:
    img = cv2.imread(image, 0) # read in the image (in GRAY)
    kp, descr = sift.compute(img, keypoints)
    descriptors.append(descr) # one descriptor per image
    
print(len(descriptors))
# 4. use one of the query input image to query the 'image database' that
#    now compress to a single area. Therefore extract the descriptor and
#    compare the descriptor to each image in the database using the L2-norm
#    and save the result into a priority queue (q = PriorityQueue())

# load test/query image
imgtest = cv2.imread("./images/db/test/car.jpg", 0)
kp, descr = sift.compute(imgtest, keypoints)

q = PriorityQueue() # internally uses heappush() to sort via distance

# compare distances using L2-Norm
for i, d in enumerate(descriptors):
    dist = distance(d, descr)
    q.put((dist, images[i]))

# 5. output (save and/or display) the query results in the order of smallest distance
ranking = open("./ranking.txt", "w+")
ranking.write("Rank \t Distance \t Filename \n")
rank = 1
while not q.empty():
    next_item = q.get()
    similarity = next_item[0]
    img_name = next_item[1]
    ranking.write(str(rank))
    ranking.write("\t")
    ranking.write(str(similarity))
    ranking.write("\t")
    ranking.write(img_name)
    ranking.write("\n")
    rank += 1
    #cv2.imshow("query result", cv2.imread(img_name))
    #cv2.waitKey(0)

ranking.close()
#cv2.destroyAllWindows()