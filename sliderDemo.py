import sys
import sip
sip.setapi('QVariant', 2)

from PyQt4.QtCore import *
from PyQt4.QtGui import *



class sliderdemo(QWidget):
    def __init__(self, parent = None):
        super(sliderdemo, self).__init__(parent)

        layout = QVBoxLayout()
        self.l1 = QLabel("Hello!")
        self.l1.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.l1)

        self.cb1 = QCheckBox("Make stuff work")
        self.cb1.setChecked(True)
        self.cb1.setText('Hi there!')
        layout.addWidget(self.cb1)

        self.sl = QSlider(Qt.Horizontal)
        self.sl.setMinimum(0)
        self.sl.setMaximum(255)
        self.sl.setValue(20)
        self.sl.setTickPosition(QSlider.TicksBelow)
        self.sl.setTickInterval(5)
        layout.addWidget(self.sl)
        self.sl.valueChanged.connect(self.valuechange)
        self.setLayout(layout)
        self.setWindowTitle("SpinBox demo")

    def valuechange(self):
        if (self.cb1.isChecked()):
            size = self.sl.value()
            self.l1.setFont(QFont("Arial",size))



def main():
    app = QApplication(sys.argv)
    ex = sliderdemo()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()



