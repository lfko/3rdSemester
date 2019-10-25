"""
    @author lfko
    @summary 
"""

import cv2
import numpy as np
import time

class CaptureManager(object):

    def __init__(self, capture, previewWindowManager = None, shouldMirrorPreview = False):

       self.previewWindowManager = previewWindowManager
       self.shouldMirrorPreview = shouldMirrorPreview 

       self._capture = capture
       self._channel = 0
       self._enteredFrame = False
       self._imageFilename = None
       self._videoFilename = None
       self._videoEncoding = None
       self._videoWriter = None

       self._startTime = None
       self._framesElapsed = int(0)
       self._fpsEstimate = None

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrieve()

        return self._frame

    @property
    def isWritingImage(self):
        return self._imageFilename is not None

    @property
    def isWritingVideo(self):
        return self._videoFilename is not None

    def writeImage(self, filename):
        self._imageFilename = filename
        
    def enterFrame(self):
        """ Captures the next frame, if any """
        # assert tests and - if failed - prints out a debug message
        assert not self._enteredFrame, 'previous enterFrame() had no matching exitFrame()'

        # grab the frame, i.e. synchronize it, but do not read it yet
        if self._capture is not None:
            self._enteredFrame = self._capture.grab()

    def exitFrame(self):
        """ Draw to window. Write to files. Release the frame. """

        if self.frame is None:
            self._enteredFrame = False
            return
        
        if self._framesElapsed == 0:
            self._startTime = time.time()
        else:
            timeElapsed = time.time() - self._startTime
            self._fpsEstimate = self._framesElapsed / timeElapsed
        self._framesElapsed += 1

        # Draw to window
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirroredFrame = np.fliplr(self._frame).copy()
                self.previewWindowManager.show(mirroredFrame)
            else:
                self.previewWindowManager.show(self._frame)
        
        if self.isWritingImage:
            cv2.imWrite(self._imageFilename, self._frame)
            self._imageFilename = None

        self._writeVideoFrame()

        # Release resources
        self._frame = None
        self._enteredFrame = False


class WindowManger(object):

    def __init__(self, windowName, keypressCallback = None):
        self.keypressCallback = keypressCallback
        # non public properties
        self._windowName = windowName
        self._isWindowCreated = False

    @property
    def isWindowCreated(self):
        return self._isWindowCreated

    def createWindow(self):
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True

    def show(self, frame):
        cv2.imshow(self._windowName, frame)

    def destroyWindow(self):
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def processEvents(self):
        keycode = cv2.waitKey(1)
        if self.keypressCallback is not None and keycode != -1:
            # Discard any non-ASCII info encoded by GTK.
            keycode &= 0xFF
            self.keypressCallback(keycode)
