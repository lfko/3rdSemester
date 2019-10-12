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
        hsv, frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # look for certain colored objects
        # define lower and upper thresholds for blue colors
        lower_blue = np.arrary([110, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # inRange checks if elements of the array lie between the elements of our boundaries
        mask = cv2.inRange(hsv, lower_blue, upper_blue)        
        # use the mask to filter only for blue(ish) objects
        res = cv2.bitwise_and(frame, frame, mask = mask)

    if ch == ord('2'):
        mode == 2 
        # LAB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    if ch == ord('3'):
        mode == 3 
        # YUV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    
    if ch == ord('4'):
        mode == 4 
        # Canny Edge Detection (the easy way)
        frame = cv2.Canny(frame, 100, 200, L2gradient = True)

    
    if ch == ord('5'):
        mode == 5 
        # Thresholding Gaussian, 255 max value, either-or threshold, 50 pixel vicinity for the calculation
        retval, frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 50, 1)
    
    if ch == ord('6'):
        mode == 6 
        # Thresholding Otsu, 255 max value, either-or threshold, 50 pixel vicinity for the calculation
        retval, frame = cv2.threshold(frame, 125, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    
    if ch == ord('q'):
        break

    #if mode == 1:
        # just example code
        # your code should implement
        #frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Display the resulting frame
    cv2.imshow('frame', frame)






# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()