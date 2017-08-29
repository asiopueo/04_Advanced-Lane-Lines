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

from hls import HLS_Channel

from functools import partial


HLS_Channel_H = partial(HLS_Channel, channel='h')
HLS_Channel_L = partial(HLS_Channel, channel='l')
HLS_Channel_S = partial(HLS_Channel, channel='s')


class Analyzer(QTabWidget):
    def __init__(self, parent = None):
        super(Analyzer, self).__init__(parent)

        self.tab1 = QWidget()
        self.tab2 = QWidget()

        self.addTab(self.tab1, "Gradients")
        self.addTab(self.tab2, "Colors")



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
        primaryLayout.addLayout(leftLayout)


        self.H_Slider = SliderUnit(HLS_Channel_H, "H")
        self.L_Slider = SliderUnit(HLS_Channel_L, "L")
        self.S_Slider = SliderUnit(HLS_Channel_S, "S")
        
        primaryLayout.addLayout(self.H_Slider.layout)
        primaryLayout.addLayout(self.L_Slider.layout)
        primaryLayout.addLayout(self.S_Slider.layout)

        self.setLayout(primaryLayout)

        self.setWindowTitle("Color Analyzer")
        self.currentIndex = 0
        self.currentSlider = self.H_Slider
        self.currentSlider.button.setChecked(True)
        self.loadImages(testImageList)

        self.H_Slider.button.toggled.connect(lambda x: self.buttonchange(self.H_Slider))
        self.H_Slider.sliderLow.valueChanged.connect(lambda x: self.valuechange(self.H_Slider))
        self.H_Slider.sliderHigh.valueChanged.connect(lambda x: self.valuechange(self.H_Slider))

        self.L_Slider.button.toggled.connect(lambda x: self.buttonchange(self.L_Slider))
        self.L_Slider.sliderLow.valueChanged.connect(lambda x: self.valuechange(self.L_Slider))
        self.L_Slider.sliderHigh.valueChanged.connect(lambda x: self.valuechange(self.L_Slider))

        self.S_Slider.button.toggled.connect(lambda x: self.buttonchange(self.S_Slider))
        self.S_Slider.sliderLow.valueChanged.connect(lambda x: self.valuechange(self.S_Slider))
        self.S_Slider.sliderHigh.valueChanged.connect(lambda x: self.valuechange(self.S_Slider))

        


    def loadImages(self, fileList):
        self.images = []

        for i in range(6):
            self.images.append(cv2.imread("./test_images/" + fileList[i]))

        self.valueChange_imageCB(0)


    def valuechange(self, sl):  
        #self.pixmap.fill(Qt.white)
        sl.button.setChecked(True)
        self.currentSlider = sl
        img = sl.function(self.images[self.currentIndex], (sl.sliderLow.value(), sl.sliderHigh.value())) * 255
        self.refreshImage(img)


    def buttonchange(self, sl):  
        self.currentSlider = sl
        img = sl.function(self.images[self.currentIndex], (sl.sliderLow.value(), sl.sliderHigh.value())) * 255
        self.refreshImage(img)
        




    def refreshImage(self, img):
        width = img.shape[1]
        img = np.dstack((img, img, img))
        img = QImage(img, img.shape[1], img.shape[0], img.shape[1]*3, QImage.Format_RGB888)
        self.pixmap = QPixmap(img).scaledToWidth(width//2)
        self.label.setPixmap(self.pixmap)



    def valueChange_imageCB(self, index):
        self.currentIndex = index
        img = self.currentSlider.function(self.images[index], (self.currentSlider.sliderLow.value(), self.currentSlider.sliderHigh.value())) * 255
        self.refreshImage(img)



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
