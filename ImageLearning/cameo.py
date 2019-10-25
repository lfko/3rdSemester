import cv2
from managers import WindowManger, CaptureManager

""" Main application class """
class Cameo(object):

    def __init__(self):
        # onKeypress as callback
        self._windowManager = WindowManger('Cameo', self.onKeypress)
        # Video device as I/O
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, True)

    def run(self):
        """ Run the main loop """
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        if keycode == 32: # space
            self._captureManager.writeImage("screenshot.png")
        elif keycode == 9: # tab
            if not self._captureManager.isWritingVideo:
                # TODO
                pass
            else:
                # TODO
                pass
        elif keycode == 27: # escape
            self._windowManager.destroyWindow()

if __name__ == "__main__":
    Cameo().run()