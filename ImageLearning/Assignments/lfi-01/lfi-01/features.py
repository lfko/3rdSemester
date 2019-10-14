import cv2

'''
    @NB: some theoretical background http://aishack.in/tutorials/sift-scale-invariant-feature-transform-introduction/
    @NB: https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
'''

cap = cv2.VideoCapture(0)
cv2.namedWindow('Learning from images: SIFT feature visualization')
while True:

    # 1. read each frame from the camera (if necessary resize the image)
    #    and extract the SIFT features using OpenCV methods
    #    Note: use the gray image - so you need to convert the image
    # 2. draw the keypoints using cv2.drawKeypoints
    #    There are several flags for visualization - e.g. DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

    # close the window and application by pressing a key

    # Capture frame-by-frame
    ret, frame = cap.read()

    # wait for key and switch to mode
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('q'):
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # exctract SIFT features, using a SIFT object
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(frame, None)

    # Display the resulting frame
    cv2.imshow('frame', frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()