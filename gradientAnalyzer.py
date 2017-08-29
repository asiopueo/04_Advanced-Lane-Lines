from __future__ import print_function
import sys

import sip
sip.setapi('QVariant', 2)

from PyQt4.QtCore import *
from PyQt4.QtGui import *

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

from functools import partial

from gradients import magThresh, dirThresh, absSobelThresh



absSobelThresh_X = partial(absSobelThresh, orient='x')
absSobelThresh_Y = partial(absSobelThresh, orient='y')





class SliderUnit(QWidget):
    def __init__(self, function, name):
        self.function = function

        self.layout = QGridLayout()
        layoutLow = QHBoxLayout()
        self.button = QRadioButton()
        self.button.setChecked(False)
        
        self.sliderLow = QSlider(Qt.Vertical)
        self.sliderLow.setMinimum(0)
        self.sliderLow.setMaximum(255)
        self.sliderLow.setValue(50)
        self.sliderLow.setTickPosition(QSlider.TicksBelow)
        self.sliderLow.setTickInterval(32)

        self.sliderHigh = QSlider(Qt.Vertical)
        self.sliderHigh.setMinimum(0)
        self.sliderHigh.setMaximum(255)
        self.sliderHigh.setValue(200)
        self.sliderHigh.setTickPosition(QSlider.TicksBelow)
        self.sliderHigh.setTickInterval(32)

        layoutLow.addWidget(self.sliderLow)
        layoutLow.addWidget(self.sliderHigh)
        self.layout.addWidget(self.button, 1, 1, Qt.AlignCenter)
        self.layout.addLayout(layoutLow, 2, 1)

        self.label = QLabel(name)
        self.layout.addWidget(self.label, 3, 1, Qt.AlignCenter)

        self.displayHigh = QLineEdit()
        self.displayHigh.setText(str(self.sliderHigh.value()))
        self.displayHigh.setFixedWidth(35)
        self.displayHigh.setAlignment(Qt.AlignCenter)

        self.displayLow = QLineEdit()
        self.displayLow.setText(str(self.sliderLow.value()))
        self.displayLow.setFixedWidth(35)
        self.displayLow.setAlignment(Qt.AlignCenter)
        
        self.layout.addWidget(self.displayHigh, 4, 1, Qt.AlignCenter)
        self.layout.addWidget(self.displayLow, 5, 1, Qt.AlignCenter)
        



class Sliders(QWidget):
    def __init__(self, parent = None):
        super(Sliders, self).__init__(parent)

        primaryLayout = QHBoxLayout()
        leftLayout = QVBoxLayout()

        self.imageCB = QComboBox()

        testImageList = [   'test1.jpg', 'test2.jpg', 
                            'test3.jpg', 'test4.jpg', 
                            'test5.jpg', 'test6.jpg']
        
        self.imageCB.addItems(testImageList)
        self.imageCB.currentIndexChanged.connect(self.valueChange_imageCB)
        leftLayout.addWidget(self.imageCB)
        
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        leftLayout.addWidget(self.label) 
       
        self.textBox = QLineEdit()
        #self.textBox.setText("((binaryMag==1)|(binaryDir==1))")
        self.textBox.setText("((binaryGradx==0)|(binaryGrady==0)) & ((binaryMag==0)|(binaryDir==0))")

        leftLayout.addWidget(self.textBox)
        self.textBox.returnPressed.connect(self.execLogic)

        primaryLayout.addLayout(leftLayout)


        self.magSlider = SliderUnit(magThresh, "Mag")
        self.dirSlider = SliderUnit(dirThresh, "Dir")
        self.sobelxSlider = SliderUnit(absSobelThresh_X, "Sobel_x")
        self.sobelySlider = SliderUnit(absSobelThresh_Y, "Sobel_y")

        primaryLayout.addLayout(self.magSlider.layout)
        primaryLayout.addLayout(self.dirSlider.layout)
        primaryLayout.addLayout(self.sobelxSlider.layout)
        primaryLayout.addLayout(self.sobelySlider.layout)

               
        self.setLayout(primaryLayout)

        self.setWindowTitle("Gradient Analyzer")
        self.currentIndex = 0
        self.currentSlider = self.magSlider
        self.currentSlider.button.setChecked(True)
        self.loadImages(testImageList)

        self.magSlider.button.toggled.connect(lambda x: self.buttonchange(self.magSlider))
        self.magSlider.sliderLow.valueChanged.connect(lambda x: self.valuechange(self.magSlider))
        self.magSlider.sliderHigh.valueChanged.connect(lambda x: self.valuechange(self.magSlider))

        self.dirSlider.button.toggled.connect(lambda x: self.buttonchange(self.dirSlider))
        self.dirSlider.sliderLow.valueChanged.connect(lambda x: self.valuechange(self.dirSlider))
        self.dirSlider.sliderHigh.valueChanged.connect(lambda x: self.valuechange(self.dirSlider))

        self.sobelxSlider.button.toggled.connect(lambda x: self.buttonchange(self.sobelxSlider))
        self.sobelxSlider.sliderLow.valueChanged.connect(lambda x: self.valuechange(self.sobelxSlider))
        self.sobelxSlider.sliderHigh.valueChanged.connect(lambda x: self.valuechange(self.sobelxSlider))

        self.sobelySlider.button.toggled.connect(lambda x: self.buttonchange(self.sobelySlider))
        self.sobelySlider.sliderLow.valueChanged.connect(lambda x: self.valuechange(self.sobelySlider))
        self.sobelySlider.sliderHigh.valueChanged.connect(lambda x: self.valuechange(self.sobelySlider))

        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)



    def loadImages(self, fileList):
        self.images = []

        for i in range(6):
            self.images.append(cv2.imread("./test_images/" + fileList[i]))

        self.valueChange_imageCB(0)



    def valuechange(self, sl):  
        #self.pixmap.fill(Qt.white)
        sl.button.setChecked(True)
        sl.displayHigh.setText(str(sl.sliderHigh.value()))
        sl.displayLow.setText(str(sl.sliderLow.value()))
        self.currentSlider = sl
        img = sl.function(self.images[self.currentIndex], 3, (sl.sliderLow.value(), sl.sliderHigh.value())) * 255
        self.refreshImage(img)


    def buttonchange(self, sl):  
        self.currentSlider = sl
        img = sl.function(self.images[self.currentIndex], 3, (sl.sliderLow.value(), sl.sliderHigh.value())) * 255
        self.refreshImage(img)


    def refreshImage(self, img):
        width = img.shape[1]
        img = np.dstack((img, img, img))
        img = QImage(img, img.shape[1], img.shape[0], img.shape[1]*3, QImage.Format_RGB888)
        self.pixmap = QPixmap(img).scaledToWidth(width//1.5)
        self.label.setPixmap(self.pixmap)


    def valueChange_imageCB(self, index):
        self.currentIndex = index
        img = self.currentSlider.function(self.images[index], 3, (self.currentSlider.sliderLow.value(), self.currentSlider.sliderHigh.value())) * 255
        self.refreshImage(img)



    def execLogic(self):
        img = self.images[0]

        binaryMag = magThresh(img, sobel_kernel=3, thresh=(self.magSlider.sliderLow.value(),self.magSlider.sliderHigh.value()) )
        binaryDir = dirThresh(img, sobel_kernel=3, thresh=(self.dirSlider.sliderLow.value(), self.dirSlider.sliderHigh.value()) )

        binaryGradx = absSobelThresh_X(img, sobel_kernel=3, thresh=(self.sobelxSlider.sliderLow.value(),self.sobelySlider.sliderHigh.value()))
        binaryGrady = absSobelThresh_Y(img, sobel_kernel=3, thresh=(self.sobelxSlider.sliderLow.value(),self.sobelySlider.sliderHigh.value()))

        binaryGrad = np.zeros_like(binaryMag)
   
        codeString = "binaryGrad[%s] = 255" % self.textBox.text()
        exec(codeString)
        self.refreshImage(binaryGrad)




def main():
    app = QApplication(sys.argv)
    sliders = Sliders()
    sliders.show()
    #adjust_threshold()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()





# Extremely slow conversion algorithm
"""
qimage = QImage(image.shape[1], image.shape[0], QImage.Format_RGB888)
for j in range(image.shape[1]):
    for i in range(image.shape[0]):
        qimage.setPixel(j, i, QColor(image[i][j][0],image[i][j][0],image[i][j][0]).rgb())
"""
