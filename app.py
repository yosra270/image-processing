import cv2 as cv
import sys
import matplotlib.pyplot as plt
from basics import *
from threshold import *
from segmentation import *
from edge_detection import *


img = cv.imread(cv.samples.findFile("img/heart.jpg"))

if img is None:
    sys.exit("Could not open or find the image.")


canny_detection(img )

k = cv.waitKey(0)

if k == ord("s"):
    cv.imwrite("img/cat2.jpg", img)