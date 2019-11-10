import cv2
import numpy as np

###
## A2.3 Harris Corner Detector
###

# Load image and convert to gray and floating point
img = cv2.imread('./images/Lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
rows, cols = gray.shape
window = 3 # k in .cornerHarris()

# Define sobel filter and use cv2.filter2D to filter the grayscale image
ksize = 3
I_x = cv2.Sobel(gray, -1, dx = 1, dy = 0, ksize = ksize)
I_y = cv2.Sobel(gray, -1, dx = 0, dy = 1, ksize = ksize)

# Compute G_xx, G_yy, G_xy and sum over all G_xx etc. 3x3 neighbors to compute
# entries of the matrix M = \sum_{3x3} [ G_xx Gxy; Gxy Gyy ]
# Note1: this results again in 3 images sumGxx, sumGyy, sumGxy
# Hint: to sum the neighbor values you can again use cv2.filter2D to do this efficiently

# these are the products of the gradient components (structure tensors)
I_xx = np.multiply(I_x, I_x)  
I_yy = np.multiply(I_y, I_y)
I_xy = np.multiply(I_x, I_y)

offset = int(window/2)
print("window:", window, "offset:", offset)
print("Sobel kernel size:", ksize)

# Define parameter
k = 0.04
threshold = 0.01
harris_thres = np.zeros(gray.shape, np.uint8)

sumKernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
sumIxx = cv2.filter2D(I_xx, -1, kernel = sumKernel)
sumIyy = cv2.filter2D(I_yy, -1, kernel = sumKernel)
sumIxy = cv2.filter2D(I_xy, -1, kernel = sumKernel)

# iterate through each pixel position and calculate the change of intensity
# at this position
'''
for y in range(rows):
    for x in range(cols):
        # y - row; x - column
        #sumIxx = I_xx[y-offset:y+offset+1, x-offset:x+offset+1].sum()
        #sumIyy = I_yy[y-offset:y+offset+1, x-offset:x+offset+1].sum()
        #sumIxy = I_xy[y-offset:y+offset+1, x-offset:x+offset+1].sum()

        # Compute the determinat and trace of M using sumGxx, sumGyy, sumGxy. 
        M = np.array([[sumIxx, sumIxy], [sumIxy, sumIyy]])
        R = np.linalg.det(M) - k*(np.trace(M)**2) # response at the current pixel location
        harris_thres[y, x] = R

        #if R > threshold:
            # we detected a corner
            #img.itemset((y, x, 0), 0)
            #img.itemset((y, x, 1), 0)
            #img.itemset((y, x, 2), 255) # set the corner pixel to red
'''
det_M = (sumIxx*sumIyy) - sumIxy**2
trace_M = sumIxx + sumIyy
harris = det_M - k*(trace_M**2)
print(harris_thres.shape)
# Filter the harris 'image' with 'harris > threshold*harris.max()'
# this will give you the indices where values are above the threshold.
# These are the corner pixel you want to use
harris_thres[harris > threshold*harris.max()] = [255] # corners are white

# The OpenCV implementation looks like this - please do not change
harris_cv = cv2.cornerHarris(gray,3,3,k)

# intialize in black - set pixels with corners in with
harris_cv_thres = np.zeros(harris_cv.shape)
harris_cv_thres[harris_cv>threshold*harris_cv.max()]=[255]

# just for debugging to create such an image as seen
# in the assignment figure.
img[harris_thres>threshold*harris_thres.max()]=[255,0,0]



# please leave this - adjust variable name if desired
print("====================================")
print("DIFF:", np.sum(np.absolute(harris_thres - harris_cv_thres)))
print("====================================")


cv2.imwrite("Harris_own.png", harris_thres)
cv2.imwrite("Harris_cv.png", harris_cv_thres)
cv2.imwrite("Image_with_Harris.png", img)
'''
cv2.namedWindow('Interactive Systems: Harris Corner')
while True:
    ch = cv2.waitKey(0)
    if ch == 27: # ESC
        break

    cv2.imshow('harris_thres',harris_thres)
    cv2.imshow('harris_cv',harris_cv_thres)
'''