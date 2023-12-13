import cv2 as cv
import numpy as np
import pyautogui
from PIL import ImageGrab
from time import sleep

# Taken from https://stackoverflow.com/questions/75620398/live-opencv-window-capture-screenshot-on-macos-darwin-using-python
import numpy as np
import Quartz as QZ


screenshot = []

class WindowCapture:

    # properties
    window_name = None
    window = None
    window_id = None
    window_width = 0
    window_height = 0

    # constructor
    def __init__(self, given_window_name=None):
        if given_window_name is not None:

            self.window_name = given_window_name
            self.window = self.get_window()

            if self.window is None:
                raise Exception('Unable to find window: {}'.format(given_window_name))

            self.window_id = self.get_window_id()

            self.window_width = self.get_window_width()
            self.window_height = self.get_window_height()

            self.window_x = self.get_window_pos_x()
            self.window_y = self.get_window_pos_y()
        else:
            raise Exception('No window name given')

    def get_window(self):
        windows = QZ.CGWindowListCopyWindowInfo(QZ.kCGWindowListOptionAll, QZ.kCGNullWindowID)
        for window in windows:
            name = window.get('kCGWindowName', 'Unknown')
            print('window name, ', name)
            if name and self.window_name in name:
                
                return window
        return None
    
    def get_window_id(self):
        return self.window['kCGWindowNumber']

    def get_window_width(self):
        return int(self.window['kCGWindowBounds']['Width'])
    
    def get_window_height(self):
        return int(self.window['kCGWindowBounds']['Height'])

    def get_window_pos_x(self):
        return int(self.window['kCGWindowBounds']['X'])

    def get_window_pos_y(self):
        return int(self.window['kCGWindowBounds']['Y'])
    
    def get_image_from_window(self):
        core_graphics_image = QZ.CGWindowListCreateImage(
            QZ.CGRectNull,
            QZ.kCGWindowListOptionIncludingWindow,
            self.window_id,
            QZ.kCGWindowImageBoundsIgnoreFraming | QZ.kCGWindowImageNominalResolution
        )

        bytes_per_row = QZ.CGImageGetBytesPerRow(core_graphics_image)
        width = QZ.CGImageGetWidth(core_graphics_image)
        height = QZ.CGImageGetHeight(core_graphics_image)

        core_graphics_data_provider = QZ.CGImageGetDataProvider(core_graphics_image)
        core_graphics_data = QZ.CGDataProviderCopyData(core_graphics_data_provider)

        np_raw_data = np.frombuffer(core_graphics_data, dtype=np.uint8)

        numpy_data = np.lib.stride_tricks.as_strided(np_raw_data,
                                                shape=(height, width, 3),
                                                strides=(bytes_per_row, 4, 1),
                                                writeable=False)
        
        final_output = np.ascontiguousarray(numpy_data, dtype=np.uint8)

        return final_output




# Detect the enemy (triangle object)
def enemy_detect():

    # TODO: Change to downsize the image so that there's not so much processing power
    enemy_img = cv.imread('enemy.png', cv.IMREAD_UNCHANGED)
    result = cv.matchTemplate(enemy_img, '', cv.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result) # best match position

    threshold = 0.85

    # if I wanted to get multiple locations
    locations = np.where(result >= threshold)
    locations = list(zip(*locations[::-1]))    

    if max_val >= threshold:
        print("enemy spotted")

        top_left = max_loc
        bottom_rigth = (top_left[0] + enemy_img.shape[1], top_left[1] + enemy_img[0])

        # Draw a rectangle around the detected enemy
        cv.rectangle('', max_loc, 0, color=(0, 255, 0), thickness=2, lineType=cv.LINE_4)
        cv.imshow('Result', '')
        cv.waitKey()

    cv.destroyAllWindows()

def capture_window():
    global screenshot

    # Find window with name 'Doing something'
    wincap = WindowCapture('Doing something')
    print('Now monitoring the window for enemies')


    while(True):
        screenshot = wincap.get_image_from_window()
        # In the tutorial he uses pywin32, but I will do something else for my system
        # convert screenshot


        screenshot = np.array(screenshot)
        screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2BGR)

        cv.imshow('Doing something', screenshot)

        # Ok this now monitors the window in real time. Now I need to draw the square on the
        # location where we find the enemy to show that enemy has been detected
        
        # press 'q' with the output window focused to exit
        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            break



    
