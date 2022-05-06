import cv2 as cv

max_value = 255
max_type = 1
max_binary_value = 255
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted'
trackbar_value = 'Thresh'
window_name = 'Threshold'
img_original = None
img_res = None

def threshold(img):
    global img_original, img_res
    img_original = img
    img_res = img_original

    if (len(img_original.shape) == 3):
        img_original = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)

    cv.namedWindow(window_name)

    cv.createTrackbar(trackbar_type, window_name , 0, max_type, on_change)
    cv.createTrackbar(trackbar_value, window_name , 50, max_value, on_change)

    on_change(0)   

def on_change(val):
    #0: Binary
    #1: Binary Inverted
    global img_res

    threshold_type = cv.getTrackbarPos(trackbar_type, window_name)
    threshold_value = cv.getTrackbarPos(trackbar_value, window_name)
    _, img_res = cv.threshold(img_original, threshold_value, max_binary_value, threshold_type )
    cv.imshow(window_name, img_res)


def get_thresholded_image():
    return img_res

