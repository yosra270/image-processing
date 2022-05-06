import cv2 as cv

from basics import *

def high_pass_filter(img, kernel_size=3):
    if kernel_size%2 == 0:
        kernel_size = kernel_size + 1
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edged_img = img - gaussian_blur(img, kernel_size)
    return edged_img

def sobel_detector(img, kernel_size=11):
    if kernel_size%2 == 0:
        kernel_size = kernel_size + 1
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edged_img = cv.Sobel(src=gaussian_blur(img, 11), ddepth=cv.CV_64F, dx=1, dy=1, ksize=kernel_size)
    return edged_img

def canny_detector(img, threshold1=100, threshold2=200):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edged_img = cv.Canny(image=img, threshold1=threshold1, threshold2=threshold2)
    return edged_img

def draw_contours(img, edged_img):
    contours, _=cv.findContours(edged_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img,contours,-1,(0,255,0),3)
    return img


# Canny detection
max_threshold1 = 300
max_threshold2 = 600

threshold1_init = 100
threshold2_init = 200

trackbar_threshold1 = 'threshold1'
trackbar_threshold2 = 'threshold2'

window_name_canny = 'Canny Edge Detector'

img_original = None

def canny_detection(img):
    global img_original
    img_original = img

    cv.namedWindow(window_name_canny)

    cv.createTrackbar(trackbar_threshold1, window_name_canny , threshold1_init, max_threshold1, on_change_canny)
    cv.createTrackbar(trackbar_threshold2, window_name_canny , threshold2_init, max_threshold2, on_change_canny)

    on_change_canny(0)   

def on_change_canny(val):
    threshold1 = cv.getTrackbarPos(trackbar_threshold1, window_name_canny)
    threshold2 = cv.getTrackbarPos(trackbar_threshold2, window_name_canny)
    edged_img = canny_detector(img_original, threshold1, threshold2)
    img_res = draw_contours(img_original.copy(), edged_img)
    cv.imshow(window_name_canny, img_res)


# Sobel Detection
max_kernel_size = 31

kernel_size_init = 11

trackbar_ksize = 'kernel size'

window_name_sobel = 'Sobel Edge Detector'

img_original = None

def sobel_detection(img):
    global img_original
    img_original = img

    cv.namedWindow(window_name_sobel)

    cv.createTrackbar(trackbar_ksize, window_name_sobel , kernel_size_init, max_kernel_size, on_change_sobel)

    on_change_sobel(11)   

def on_change_sobel(val):
    edged_img = sobel_detector(img_original, val)
    cv.imshow(window_name_sobel, edged_img)