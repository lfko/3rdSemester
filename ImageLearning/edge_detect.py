import cv2 
import numpy as np

img = cv2.imread("Johanna.jpg")

'''
    Apply Blur, then do Laplacian
'''
img_blurred = cv2.GaussianBlur(img, (5, 5), 0) # 5x5 kernel
img_blurred_grey = cv2.cvtColor(img_blurred, cv2.COLOR_RGB2GRAY)
cv2.Laplacian(img_blurred_grey, cv2.CV_BU, img_blurred_grey, ksize = 5)

