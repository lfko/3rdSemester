import numpy as np
import cv2


'''
@NB
    Thresholding helps us in simplifying our picture, e.g. by applying a threshold to a grayscale picture and thus 
    converting every value below 125 to 0 and above to 1 respectively. For colored images the threshold is applied
    to every channel independently

@NB
    Edge Detection

@NB
    Blurs and Filters: https://www.youtube.com/watch?v=C_zFhWdM4ic&list=PLhhhoPkZzWjZlXDz7AeIl4HfyGMPlsWK7&index=11
    ED: https://www.youtube.com/watch?v=uihBwtPIBxM&list=PLhhhoPkZzWjZlXDz7AeIl4HfyGMPlsWK7&index=13 (Sobel)
    Impl: https://pythonprogramming.net/thresholding-image-analysis-python-opencv-tutorial/
'''

cap = cv2.VideoCapture(0)
mode = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # wait for key and switch to mode
    ch = cv2.waitKey(1) & 0xFF
    
    if ch == ord('1'):
        mode = 1 
        # HSV

    if ch == ord('2'):
        mode = 2 
        # LAB

    if ch == ord('3'):
        mode = 3 
        # YUV

    if ch == ord('4'):
        mode = 4
        # Canny Edge Detection (the easy way) 

    if ch == ord('5'):
        # Thresholding Gaussian
        mode = 5 

    if ch == ord('6'):
        # Thresholding Otsu
        mode = 6 

    if ch == ord('q'):
        break

    '''
        Modes Implementation
    '''

    if mode == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # look for certain colored objects
        # define lower and upper thresholds for blue colors
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # inRange checks if elements of the array lie between the elements of our boundaries
        mask = cv2.inRange(frame, lower_blue, upper_blue)        
        # use the mask to filter only for blue(ish) objects
        res = cv2.bitwise_and(frame, frame, mask = mask)

    if mode == 2:
        # LAB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    if mode == 3:
        # YUF
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    
    if mode == 4:
        # Canny Edge
        frame = cv2.Canny(frame, 100, 200, L2gradient = True)

    if mode == 5:
        # Thresholding Gaussian, 255 max value, either-or threshold, 50 pixel vicinity for the calculation
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    
    if mode == 6:
        # Thresholding Otsu, 255 max value, either-or threshold, 50 pixel vicinity for the calculation
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        retval, frame = cv2.threshold(frame, 125, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #cv2.imshow('threshold', threshold)
    # Display the resulting frame
    cv2.imshow('frame', frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()