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

from gradients import magThresh, dirThresh, absSobelThresh






class Sliders(QWidget):
    def __init__(self, parent = None):
        super(Sliders, self).__init__(parent)

        primaryLayout = QHBoxLayout()
        leftLayout = QVBoxLayout()
        rightLayout = QGridLayout()#QHBoxLayout()

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

    
        #Sliders threshold limits GradientMagn 
        layoutGradMagLow = QHBoxLayout()
        self.rb1 = QRadioButton()
        self.rb1.setObjectName('magButton')
        self.rb1.setChecked(True)
        self.rb1.toggled.connect(self.buttonchange_gradientMag)
        

        self.sl1 = QSlider(Qt.Vertical)
        self.sl1.setMinimum(0)
        self.sl1.setMaximum(255)
        self.sl1.setValue(50)
        self.sl1.setTickPosition(QSlider.TicksBelow)
        self.sl1.setTickInterval(32)
        self.sl1.valueChanged.connect(self.valuechange_gradientMag)
                
        self.sl2 = QSlider(Qt.Vertical)
        self.sl2.setMinimum(0)
        self.sl2.setMaximum(255)
        self.sl2.setValue(200)
        self.sl2.setTickPosition(QSlider.TicksBelow)
        self.sl2.setTickInterval(32)
        self.sl2.valueChanged.connect(self.valuechange_gradientMag)

        layoutGradMagLow.addWidget(self.sl1)
        layoutGradMagLow.addWidget(self.sl2)
        rightLayout.addWidget(self.rb1, 1, 1, Qt.AlignCenter)
        rightLayout.addLayout(layoutGradMagLow, 2, 1)

        #Sliders threshold limits DirectionalMagn 
        # Why is it scaled up to 255?
        layoutGradDirLow = QHBoxLayout()
        self.rb2 = QRadioButton()
        self.rb2.setObjectName('dirButton')
        self.rb2.toggled.connect(self.buttonchange_gradientDir)


        self.sl3 = QSlider(Qt.Vertical)
        self.sl3.setMinimum(0)
        self.sl3.setMaximum(157)
        self.sl3.setValue(30)
        self.sl3.setTickPosition(QSlider.TicksBelow)
        self.sl3.setTickInterval(7)
        self.sl3.valueChanged.connect(self.valuechange_gradientDir)
                
        self.sl4 = QSlider(Qt.Vertical)
        self.sl4.setMinimum(0)
        self.sl4.setMaximum(157)
        self.sl4.setValue(120)
        self.sl4.setTickPosition(QSlider.TicksBelow)
        self.sl4.setTickInterval(7)
        self.sl4.valueChanged.connect(self.valuechange_gradientDir)
        layoutGradDirLow.addWidget(self.sl3)
        layoutGradDirLow.addWidget(self.sl4)

        rightLayout.addWidget(self.rb2, 1, 2, Qt.AlignCenter)
        rightLayout.addLayout(layoutGradDirLow, 2, 2)

        
    
        #Sobelx
        layoutSobelxLow = QHBoxLayout()
        self.rb3 = QRadioButton()
        self.rb3.toggled.connect(self.buttonchange_sobelx)

        self.sl5 = QSlider(Qt.Vertical)
        self.sl5.setMinimum(0)
        self.sl5.setMaximum(255)
        self.sl5.setValue(30)
        self.sl5.setTickPosition(QSlider.TicksBelow)
        self.sl5.setTickInterval(32)
        self.sl5.valueChanged.connect(self.valuechange_sobelx)
        
        self.sl6 = QSlider(Qt.Vertical)
        self.sl6.setMinimum(0)
        self.sl6.setMaximum(255)
        self.sl6.setValue(120)
        self.sl6.setTickPosition(QSlider.TicksBelow)
        self.sl6.setTickInterval(32)
        self.sl6.valueChanged.connect(self.valuechange_sobelx)
        layoutSobelxLow.addWidget(self.sl5)
        layoutSobelxLow.addWidget(self.sl6)
        
        rightLayout.addWidget(self.rb3, 1, 3, Qt.AlignCenter)
        rightLayout.addLayout(layoutSobelxLow, 2, 3)



        #Sobely
        layoutSobelyLow = QHBoxLayout()
        self.rb4 = QRadioButton()
        self.rb4.toggled.connect(self.buttonchange_sobely)

        self.sl7 = QSlider(Qt.Vertical)
        self.sl7.setMinimum(0)
        self.sl7.setMaximum(255)
        self.sl7.setValue(30)
        self.sl7.setTickPosition(QSlider.TicksBelow)
        self.sl7.setTickInterval(32)
        self.sl7.valueChanged.connect(self.valuechange_sobely)
        
        self.sl8 = QSlider(Qt.Vertical)
        self.sl8.setMinimum(0)
        self.sl8.setMaximum(255)
        self.sl8.setValue(120)
        self.sl8.setTickPosition(QSlider.TicksBelow)
        self.sl8.setTickInterval(32)
        self.sl8.valueChanged.connect(self.valuechange_sobely)
        layoutSobelyLow.addWidget(self.sl7)
        layoutSobelyLow.addWidget(self.sl8)

        rightLayout.addWidget(self.rb4, 1, 4, Qt.AlignCenter)
        rightLayout.addLayout(layoutSobelyLow, 2, 4)

        self.label1 = QLabel("Mag")
        self.label2 = QLabel("Dir")
        self.label3 = QLabel("Sobel_x")
        self.label4 = QLabel("Sobel_y")
        rightLayout.addWidget(self.label1, 3, 1, Qt.AlignCenter)
        rightLayout.addWidget(self.label2, 3, 2, Qt.AlignCenter)
        rightLayout.addWidget(self.label3, 3, 3, Qt.AlignCenter)
        rightLayout.addWidget(self.label4, 3, 4, Qt.AlignCenter)



        primaryLayout.addLayout(leftLayout)
        primaryLayout.addLayout(rightLayout)

        self.setLayout(primaryLayout)
        self.setWindowTitle("Gradient Analyzer")
        self.loadImages(testImageList)


        

    def loadImages(self, fileList):
        self.images = []

        for i in range(6):
            self.images.append(cv2.imread("./test_images/" + fileList[i]))

        self.valueChange_imageCB(0)

    """
    def buttonPush(self):
        self.b.setChecked(True)
        img = func(self.images[self.currentIndex], 3, (self.sl1.value(), self.sl2.value())) * 255
        self.refreshImage(img)

    def valuechange(self, func):
        img = func(self.images[self.currentIndex], 3, (self.sl1.value(), self.sl2.value())) * 255
        self.refreshImage(img)
    """

    def valuechange_gradientMag(self):  
        #self.pixmap.fill(Qt.white)
        self.rb1.setChecked(True)
        img = magThresh(self.images[self.currentIndex], 3, (self.sl1.value(), self.sl2.value())) * 255
        self.refreshImage(img)
        

    def valuechange_gradientDir(self):
        self.rb2.setChecked(True)
        img = dirThresh(self.images[self.currentIndex], 15, (self.sl3.value(),self.sl4.value())) * 255
        self.refreshImage(img)

    def valuechange_sobelx(self):  
        self.rb3.setChecked(True)
        img = absSobelThresh(self.images[self.currentIndex], 'x', 3, (self.sl5.value(), self.sl6.value())) * 255
        self.refreshImage(img)


    def valuechange_sobely(self):  
        self.rb4.setChecked(True)
        img = absSobelThresh(self.images[self.currentIndex], 'y', 3, (self.sl7.value(), self.sl8.value())) * 255
        self.refreshImage(img)


    def buttonchange_gradientMag(self):  
        #self.pixmap.fill(Qt.white)
        img = magThresh(self.images[self.currentIndex], 3, (self.sl1.value(), self.sl2.value())) * 255
        self.refreshImage(img)
        

    def buttonchange_gradientDir(self):
        img = dirThresh(self.images[self.currentIndex], 15, (self.sl3.value(),self.sl4.value())) * 255
        self.refreshImage(img)

    def buttonchange_sobelx(self):  
        img = absSobelThresh(self.images[self.currentIndex], 'x', 3, (self.sl5.value(), self.sl6.value())) * 255
        self.refreshImage(img)


    def buttonchange_sobely(self):  
        img = absSobelThresh(self.images[self.currentIndex], 'y', 3, (self.sl7.value(), self.sl8.value())) * 255
        self.refreshImage(img)


    def refreshImage(self, img):
        width = img.shape[1]
        img = np.dstack((img, img, img))
        img = QImage(img, img.shape[1], img.shape[0], img.shape[1]*3, QImage.Format_RGB888)
        self.pixmap = QPixmap(img).scaledToWidth(width//2)
        self.label.setPixmap(self.pixmap)


    # Always jumps back to the values of setting no. 1
    def valueChange_imageCB(self, index):
        self.currentIndex = index
        currentChoice = self.imageCB.itemText(index)
        img = magThresh(self.images[index], 3, (self.sl1.value(), self.sl2.value())) * 255
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
