import numpy as np 
import cv2 as cv

max_value = 10
trackbar_value = 'k'
window_name = 'k-Means Segmentation'
img_original = None
img_res = None

def segmentation(img):
    global img_original, img_res
    img_original = img
    img_res = img_original

    img_original = cv.cvtColor(img_original, cv.COLOR_BGR2RGB)

    cv.namedWindow(window_name)

    cv.createTrackbar(trackbar_value, window_name , 2, max_value, on_change)

    on_change(10)   

def on_change(val):
    global img_res

    # Reshaping the image into a 2D array of pixels and 3 color values (RGB) 
    pixel_vals = img_original.reshape((-1,3))
    # Convert to float type only for supporting cv2.kmean
    pixel_vals = np.float32(pixel_vals)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)

    if val == 0:
        val = val + 1
    _, labels, centers = cv.kmeans(pixel_vals, val, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS) 
    
    # convert data into 8-bit values 
    centers = np.uint8(centers) 
    segmented_data = centers[labels.flatten()] # Mapping labels to center points( RGB Value)

    # reshape data into the original image dimensions 
    segmented_image = segmented_data.reshape((img_original.shape)) 

    img_res = segmented_image
    cv.imshow(window_name, segmented_image)


def get_segmented_image():
    return img_res
