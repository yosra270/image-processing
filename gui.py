from PyQt5.QtCore import  Qt, QSize
from PyQt5.QtWidgets import (QApplication, QCheckBox, QDialog, QGridLayout,
        QGroupBox, QHBoxLayout, QLabel, QPushButton, QStyleFactory,
        QVBoxLayout, QFileDialog, QMessageBox)

from PyQt5.QtGui import (QPixmap, QImage)
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar


from basics import *
from erosion_dilation import erosion_dilation
from threshold import threshold, get_thresholded_image
from segmentation import segmentation
from edge_detection import canny_detection, sobel_detection
from face_and_eyes_detection import detectAndDisplayHumanFaceAndEyes, detectAndDisplayCatFace
from pedestrians_detection import pedestrians_detection



class WidgetGallery(QDialog):
    def __init__(self, parent=None):
        super(WidgetGallery, self).__init__(parent)

        QApplication.setStyle(QStyleFactory.create('Fusion'))
        QApplication.setPalette(QApplication.style().standardPalette())

        self.createSmoothingContrastErosionGroupBox()
        self.createFaceAndEdgeDetectionGroupBox()
        self.createPedestriansDetectionAndSegmentationGroupBox()
        self.createOperationsGroupBox()
        self.createBottomLeftImage()
        self.createBottomRightGroupBox()
        self.createBottomLayout()



        mainLayout = QGridLayout()
        mainLayout.addWidget(self.operationsGroupBox, 0, 0, 1, 2)
        mainLayout.addWidget(self.bottomLeftImage, 1, 0, alignment=Qt.AlignCenter)
        mainLayout.addWidget(self.bottomRightGroupBox, 1, 1)
        mainLayout.setRowStretch(0, 1)
        mainLayout.setRowStretch(1, 1)
        mainLayout.setColumnStretch(0, 1)
        mainLayout.setColumnStretch(1, 1)
        mainLayout.addLayout(self.downLayout, 2, 0, 1, 2)

        self.setLayout(mainLayout)

        self.setWindowTitle("Image Processing Tool")


    # Open image file
    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, _ = QFileDialog.getOpenFileName(self,"Open Image", "","Image Files (*.JPEG *.JPG *.PNG *.PGM)", options=options)
        if self.fileName is None:
            sys.exit("Could not open or find the image.")
        self.openImage(self.fileName)


    # Open & Save image operations + Conversions
    def createBottomLayout(self):
        openButton = QPushButton("Open")
        openButton.setDefault(True)
        openButton.clicked.connect(self.openFileNameDialog)

        saveButton = QPushButton("Save")
        saveButton.setDefault(True)
        saveButton.clicked.connect(self.saveFile)

        self.toGrayScaleCheckBox = QCheckBox("To Gray Scale")
        self.toGrayScaleCheckBox.stateChanged.connect(self.toGrayScale)

        self.toHSVCheckBox = QCheckBox("To HSV")
        self.toHSVCheckBox.stateChanged.connect(self.toHSV)


        self.downLayout = QHBoxLayout()
        self.downLayout.addWidget(openButton)
        self.downLayout.addStretch(1)
        self.downLayout.addWidget(self.toGrayScaleCheckBox)
        self.downLayout.addWidget(self.toHSVCheckBox)
        self.downLayout.addStretch(1)
        self.downLayout.addWidget(saveButton)

    def saveFile(self):
        newFileName = self.fileName
        if (len(self.currentImg.shape) < 3):
            extensionPos = self.fileName.rfind('.')
            newFileName = self.fileName[:extensionPos]+'.pgm' 
        cv.imwrite(newFileName, self.currentImg)

    # Image processing tools : Contrast and Brightness, Smoothing, Erosion and Dilatation, Thresholding
    def createSmoothingContrastErosionGroupBox(self):
        self.topLeftGroupBox = QGroupBox("")

        
        contrastAndBrightnessButton = QPushButton("Contrast and Brightness")
        contrastAndBrightnessButton.setDefault(True)
        contrastAndBrightnessButton.setFixedSize(QSize(270, 40))
        contrastAndBrightnessButton.clicked.connect(self.startContrastAndBrightnessAdjustements)


        smoothingButton = QPushButton("Smoothing (Blurring)")
        smoothingButton.setDefault(True)
        smoothingButton.setFixedSize(QSize(270, 40))
        smoothingButton.clicked.connect(self.startBlurring)

        erosionAndDilatationButton = QPushButton("Erosion and Dilation")
        erosionAndDilatationButton.setDefault(True)
        erosionAndDilatationButton.setFixedSize(QSize(270, 40))
        erosionAndDilatationButton.clicked.connect(self.startErosionAndDilation)

        layout = QVBoxLayout()
        layout.addStretch(1)
        layout.addWidget(contrastAndBrightnessButton)
        layout.addWidget(smoothingButton)
        layout.addWidget(erosionAndDilatationButton)
        layout.addStretch(1)
        self.topLeftGroupBox.setLayout(layout)  

    # Image processing tools : Contrast and Brightness, Smoothing, Erosion and Dilatation, Thresholding
    def createPedestriansDetectionAndSegmentationGroupBox(self):
        self.pedestriansDetectionGroupBox = QGroupBox("")

        thresholdButton = QPushButton("Thresholding")
        thresholdButton.setDefault(True)
        thresholdButton.setFixedSize(QSize(270, 40))
        thresholdButton.clicked.connect(self.startThresholding)

        segmentationButton = QPushButton("Segmentation")
        segmentationButton.setDefault(True)
        segmentationButton.setFixedSize(QSize(270, 40))
        segmentationButton.clicked.connect(self.startSegmentation)


        pedestriansDetectionButton = QPushButton("Pedestrians Detection")
        pedestriansDetectionButton.setDefault(True)
        pedestriansDetectionButton.setFixedSize(QSize(270, 40))
        pedestriansDetectionButton.clicked.connect(pedestrians_detection)

        layout = QVBoxLayout()
        layout.addStretch(1)
        layout.addWidget(thresholdButton)
        layout.addWidget(segmentationButton)
        layout.addWidget(pedestriansDetectionButton)
        layout.addStretch(1)
        self.pedestriansDetectionGroupBox.setLayout(layout)   

    # Image processing tools : Segmentation, Edge Detection, Face and Eyes Detection
    def createFaceAndEdgeDetectionGroupBox(self):

        self.edgeDetectionGroupBox = QGroupBox("Edge Detection")
        cannyEdgeDetectionButton = QPushButton("Canny Detector")
        cannyEdgeDetectionButton.setDefault(True)
        cannyEdgeDetectionButton.clicked.connect(self.startCannyDetection)
        sobelEdgeDetectionButton = QPushButton("Sobel Detector")
        sobelEdgeDetectionButton.setDefault(True)
        sobelEdgeDetectionButton.clicked.connect(self.startSobelDetection)

        layout = QVBoxLayout()
        layout.addWidget(cannyEdgeDetectionButton)
        layout.addWidget(sobelEdgeDetectionButton)

        self.edgeDetectionGroupBox.setLayout(layout)



        self.faceDetectionGroupBox = QGroupBox("Face Detection")
        humanFaceDetectionButton = QPushButton("Human Face and Eyes Detection")
        humanFaceDetectionButton.setDefault(True)
        humanFaceDetectionButton.clicked.connect(self.startFaceAndEyesDetection)
        catFaceDetectionButton = QPushButton("Cat Face Detection")
        catFaceDetectionButton.setDefault(True)
        catFaceDetectionButton.clicked.connect(self.startCatFaceDetection)

        layout = QVBoxLayout()
        layout.addWidget(humanFaceDetectionButton)
        layout.addWidget(catFaceDetectionButton)

        self.faceDetectionGroupBox.setLayout(layout)

    

    # All processing  tools
    def createOperationsGroupBox(self):
        self.operationsGroupBox = QGroupBox("Operations")
        operationsLayout = QGridLayout()
        operationsLayout.addWidget(self.topLeftGroupBox, 0, 0, 2, 1)
        operationsLayout.addWidget(self.edgeDetectionGroupBox, 0, 1)
        operationsLayout.addWidget(self.faceDetectionGroupBox, 1, 1)
        operationsLayout.addWidget(self.pedestriansDetectionGroupBox, 0, 2, 2, 1)
        self.operationsGroupBox.setLayout(operationsLayout)

    # Image preview
    def createBottomLeftImage(self):
        self.bottomLeftImage = QGroupBox("")
        self.imageBox = QLabel()

        self.openImage("img/default.png", False)
        self.fileName = "img/default.png"

        
        self.avgLabel = QLabel(f"Avg. : {round(average(self.originalImg),2)}\t")
        self.avgLabel.setStyleSheet("color: purple")
        self.stdLabel = QLabel(f"Std. : {round(std(self.originalImg),2)}")
        self.stdLabel.setStyleSheet("color: purple")

        layout = QGridLayout()
        layout.addWidget(self.imageBox, 0, 0, 1, 2)
        layout.addWidget(self.avgLabel, 1, 0)
        layout.addWidget(self.stdLabel, 1, 1)
        self.bottomLeftImage.setLayout(layout)
    
    def openImage(self, fileName="img/default.png", replace_existant=True):

        if fileName.find('.pgm') > 0:
            self.originalImg = cv.imread(cv.samples.findFile(fileName), cv.IMREAD_GRAYSCALE)
        else:
            self.originalImg = cv.imread(cv.samples.findFile(fileName))
        self.currentImg = self.originalImg
        
        pixmap = self.prepareImage(self.originalImg)
        self.imageBox.setPixmap(pixmap)

        if replace_existant is True:
            self.histogramEqualizationCheckBox.setChecked(False)  
            self.toGrayScaleCheckBox.setChecked(False)
            self.toHSVCheckBox.setChecked(False)
            self.updateImageData()

    def updateImage(self):
        pixmap = self.prepareImage(self.currentImg)
        self.imageBox.setPixmap(pixmap)

    def prepareImage(self, img):
        width = 430
        height = 320
        dim = (width, height)
        
        resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)

        if (len(img.shape) < 3):
            format = QImage.Format_Indexed8
            channel = 1
        else:
            format = QImage.Format_RGB888
            channel = 3

        bytesPerLine = channel * width
        qImg = QImage(resized.data, width, height, bytesPerLine, format).rgbSwapped()
        
        pixmap = QPixmap(qImg)
        return pixmap

    # Histogram
    def createBottomRightGroupBox(self):
        self.bottomRightGroupBox = QGroupBox("Histogram")

        self.figure = plt.figure()

        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)


        self.histogramEqualizationCheckBox = QCheckBox("Histogram Equalization")
        self.histogramEqualizationCheckBox.stateChanged.connect(self.updateHistogramEqualization)


        self.cumulativeHistogramCheckBox = QCheckBox("Cumulative Histogram")
        self.cumulativeHistogramCheckBox.stateChanged.connect(self.setCumulativeHistogram)

        layout = QGridLayout()
        layout.addWidget(self.toolbar, 0, 0, 1, 2)
        layout.addWidget(self.canvas, 1, 0, 1, 2)
        layout.addWidget(self.histogramEqualizationCheckBox, 2, 0)
        layout.addWidget(self.cumulativeHistogramCheckBox, 2, 1)
        self.bottomRightGroupBox.setLayout(layout)

        calculate_histogram(self.originalImg)
        self.canvas.draw()

    def updateImageData(self):
        self.avgLabel.setText(f"Avg. : {round(average(self.currentImg),2)}\t")
        self.stdLabel.setText(f"Std. : {round(std(self.currentImg),2)}")
      

        calculate_histogram(self.currentImg)
        self.canvas.draw()  

    def updateHistogramEqualization(self):
        if self.histogramEqualizationCheckBox.isChecked() is True:
            self.imgBeforeEqualization = self.currentImg
            self.currentImg = histogram_equalization(self.currentImg)
            self.updateImageData()
            self.updateImage()
        else :
            self.currentImg = self.imgBeforeEqualization
            self.updateImageData()
            self.updateImage()


    def setCumulativeHistogram(self):
        if self.cumulativeHistogramCheckBox.isChecked() is True:
            histogram_cumulative(self.currentImg)
        else :
            calculate_histogram(self.currentImg)

        self.canvas.draw()  

    # Conversions
    def convert(self, algorithm, checkbox):
        if checkbox.isChecked() is True:
            self.imgBeforeConversion = self.currentImg
            self.currentImg = algorithm(self.currentImg)
            self.updateImageData()
            self.updateImage()
        else :
            self.currentImg = self.imgBeforeConversion
            self.updateImageData()
            self.updateImage()
    
    def toGrayScale(self):
        self.convert(to_gray, self.toGrayScaleCheckBox)
    def toHSV(self):
        self.convert(to_hsv, self.toHSVCheckBox)

    def startContrastAndBrightnessAdjustements(self):
        self.infoMsg()

        adjust_contrast_and_brightness(self.currentImg)
        k = cv.waitKey(0)
        cv.destroyAllWindows()

        self.saveAdjustements(k, get_corrected_image)
    
    def startBlurring(self):
        self.infoMsg()

        blur(self.currentImg)
        k = cv.waitKey(0)
        cv.destroyAllWindows()

        self.saveAdjustements(k, get_corrected_image)
    
    def startErosionAndDilation(self):
        erosion_dilation(self.currentImg)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def startThresholding(self):
        self.infoMsg()

        threshold(self.currentImg)
        k = cv.waitKey(0)
        cv.destroyAllWindows()

        self.saveAdjustements(k, get_thresholded_image)

    def startSegmentation(self):
        segmentation(self.currentImg)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def startCannyDetection(self):
        canny_detection(self.currentImg)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def startSobelDetection(self):
        sobel_detection(self.currentImg)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def startFaceAndEyesDetection(self):
        detectAndDisplayHumanFaceAndEyes(self.currentImg)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def startCatFaceDetection(self):
        detectAndDisplayCatFace(self.currentImg)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def infoMsg(self):
        infoMsgBox = QMessageBox()
        infoMsgBox.setIcon(QMessageBox.Information)
        infoMsgBox.setText("After adjusting the image :\n\n- Press Escape to discard changes\n- Press S to save the corrected image")
        infoMsgBox.setWindowTitle("Finalization options")
        infoMsgBox.setStandardButtons(QMessageBox.Ok)
        infoMsgBox.exec()

    def saveAdjustements(self, k, get_new_image_function): 
        if k == ord("s"):
            newFileName = self.fileName
            self.currentImg = get_new_image_function()
            if (len(self.currentImg.shape) < 3):
                extensionPos = self.fileName.rfind('.')
                newFileName = self.fileName[:extensionPos]+'.pgm' 
            cv.imwrite(newFileName, self.currentImg)
            self.updateImageData()
            self.updateImage()

if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    gallery = WidgetGallery()
    gallery.show()
    sys.exit(app.exec()) 
