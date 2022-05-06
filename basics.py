import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def average(img):
    return np.average(img) if img is not None else 0.00

def std(img):
    return np.std(img) if img is not None else 0.00

# Conversions
def to_gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img is not None else None


def to_hsv(img):
    return cv.cvtColor(img, cv.COLOR_BGR2HSV) if img is not None else None

# Histograms
def calculate_histogram(img):
    if (len(img.shape) < 3):
        color = ('k',)
    else:
        color = ('b','g','r')
    histSize = 256
    histRange = (0, 256)
    plt.clf()
    for i,col in enumerate(color):
        histr = cv.calcHist([img],[i],None,[histSize],histRange)
        plt.plot(histr,color = col)
        plt.xlim([0,256])

def histogram_cumulative(img):
    if (len(img.shape) < 3):
        color = ('k',)
    else:
        color = ('b','g','r')
    histSize = 256
    histRange = (0, 256)
    plt.clf()
    for i,col in enumerate(color):
        histr = cv.calcHist([img],[i],None,[histSize],histRange)
        hc = np.zeros(histSize,int)
        hc[0] = histr[0]
        for index in range(1,len(histr)): 
            hc[index]= histr[index] + hc[index -1]
        plt.plot(hc,color = col)
        plt.xlim(histRange)

def histogram_equalization(src):
    if (len(src.shape) == 3):
       src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # Apply Histogram Equalization
    dst = cv.equalizeHist(src)

    return dst

# Contrast and Brightness
alpha = 1.0
alpha_max = 500
beta = 0
beta_max = 200
img_original = None
img_res = None

def basicLinearTransform():
    global img_res
    img_res = cv.convertScaleAbs(img_original, alpha=alpha, beta=beta)
    img_corrected = cv.hconcat([img_original, img_res])
    cv.imshow("Brightness and contrast adjustments", img_corrected)

def on_linear_transform_alpha_trackbar(val):
    global alpha
    alpha = val / 100
    basicLinearTransform()

def on_linear_transform_beta_trackbar(val):
    global beta
    beta = val - 100
    basicLinearTransform()

def adjust_contrast_and_brightness(img):
    global img_original, img_res
    img_original = img
    img_res = img_original
    img_corrected = cv.hconcat([img_original, img_original])
    cv.namedWindow('Brightness and contrast adjustments')

    alpha_init = int(alpha *100)
    cv.createTrackbar('Alpha gain (contrast)', 'Brightness and contrast adjustments', alpha_init, alpha_max, on_linear_transform_alpha_trackbar)
    beta_init = beta + 100
    cv.createTrackbar('Beta bias (brightness)', 'Brightness and contrast adjustments', beta_init, beta_max, on_linear_transform_beta_trackbar)

    on_linear_transform_alpha_trackbar(alpha_init)
    return img_corrected

def get_corrected_image():
    return img_res

# Smoothing(Blurring)

## Average Blur
def average_blur(img, kernel_size=3):
    smoothed_img = cv.blur(img,ksize=(kernel_size,kernel_size))
    return smoothed_img

## Gaussian Blur
def gaussian_blur(img, kernel_size=3):
    smoothed_img = cv.GaussianBlur(img,(kernel_size,kernel_size), cv.BORDER_DEFAULT)
    return smoothed_img

## Median Blur
def median_blur(img, kernel_size = 3):
    smoothed_img = cv.medianBlur(img,kernel_size)
    return smoothed_img

noised_img = None
kernel_size_init = 3
kernel_size_max = 101
median_kernel_size = kernel_size_init
gaussian_kernel_size = kernel_size_init
average_kernel_size = kernel_size_init
window_name = ""


def on_median_kernel_size_trackbar(val):
    global median_kernel_size
    on_noise_kernel_size_trackbar(val, median_blur, median_kernel_size)

def on_gaussian_kernel_size_trackbar(val):
    global gaussian_kernel_size
    on_noise_kernel_size_trackbar(val, gaussian_blur, gaussian_kernel_size)

def on_average_kernel_size_trackbar(val):
    global average_kernel_size
    on_noise_kernel_size_trackbar(val, average_blur, average_kernel_size)

def on_noise_kernel_size_trackbar(val, blurring_algorithm, kernel_size):
    global img_res
    if val%2 == 0:
        val = val + 1
    kernel_size = val
    img_res = blurring_algorithm(noised_img, kernel_size)
    smoothed_img = cv.hconcat([noised_img, img_res])
    cv.imshow(window_name, smoothed_img)

def blur(img):
    global noised_img, window_name
    noised_img = img
    window_name = "Blurring (Smoothing)"

    smoothed_img = cv.hconcat([noised_img, noised_img])
    cv.namedWindow(window_name)

    cv.createTrackbar('Gaussian Blur - kernel size', window_name, kernel_size_init, kernel_size_max, on_gaussian_kernel_size_trackbar)
    cv.createTrackbar('Median Blur - kernel size', window_name, kernel_size_init, kernel_size_max, on_median_kernel_size_trackbar)
    cv.createTrackbar('Average Blur - kernel size', window_name, kernel_size_init, kernel_size_max, on_average_kernel_size_trackbar)
    
    on_gaussian_kernel_size_trackbar(kernel_size_init)
    on_median_kernel_size_trackbar(kernel_size_init)
    on_average_kernel_size_trackbar(kernel_size_init)
    return smoothed_img


# Noise removal
def remove_noise(img, iterations=2, kernel_size=3):
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    denoised_img = cv.morphologyEx(img,cv.MORPH_OPEN,kernel, iterations = iterations)
    return denoised_img

# Contours



# Threshold

# Erosion & Dilatation

# Segmentation

# Face detection
