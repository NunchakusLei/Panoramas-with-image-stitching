import sys
from PyQt5.QtWidgets import QScrollArea, QGridLayout, QMainWindow, QPushButton, QLabel, QApplication, QWidget, \
    QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QIcon, QPixmap

from stiching import Sticher
import cv2
import numpy as np


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.gridLayout = None
        self.title = 'Panorama Image Stitching'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 640
        self.selectImageBtn = None
        self.generateResultBtn = None
        self.numSelectLabel = None
        self.labelImage = None
        self.imageAreaWidgetContents = None
        self.prevImageBtn = None
        self.nextImageBtn = None

        self.currentIndex = 0
        self.imageFiles = None
        self.results = None
        self.initUI()

    def initUI(self):
        self.statusBar()
        self.numSelectLabel = QLabel(self)
        self.numSelectLabel.setText("Select 0 images")
        self.numSelectLabel.move(30, 10)

        self.selectImageBtn = QPushButton("Select", self)
        self.selectImageBtn.move(200, 10)
        self.selectImageBtn.setEnabled(True)
        self.selectImageBtn.clicked.connect(self.openFileNamesDialog)

        self.generateResultBtn = QPushButton("Panorama", self)
        self.generateResultBtn.move(350, 10)
        self.generateResultBtn.setEnabled(False)
        self.generateResultBtn.clicked.connect(self.generateResult)

        self.prevImageBtn = QPushButton("Prev Image", self)
        self.prevImageBtn.move(200, 40)
        self.prevImageBtn.setEnabled(False)
        self.prevImageBtn.clicked.connect(self.preImage)

        self.nextImageBtn = QPushButton("Next Image", self)
        self.nextImageBtn.move(350, 40)
        self.nextImageBtn.setEnabled(False)
        self.nextImageBtn.clicked.connect(self.nextImage)

        self.labelImage = QLabel(self)
        self.labelImage.move(320, 320)
        self.labelImage.setText("No Result")

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.show()

    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "QFileDialog.getOpenFileNames()", "",
                                                "Images (*.jpg *.jpeg *.png *.bmp)", options=options)
        if files:
            print(files)
            self.currentIndex = 0
            self.imageFiles = files
            self.showImages(self.imageFiles[0])
            self.generateResultBtn.setEnabled(True)
            self.nextImageBtn.setEnabled(True)
            self.prevImageBtn.setEnabled(True)

    def showImages(self, filename):
        self.labelImage.move(0, 80)
        self.labelImage.resize(640, 480)
        pixmap = QPixmap(filename)
        pixmapScaled = pixmap.scaled(640, 480, Qt.KeepAspectRatio)
        self.labelImage.setPixmap(pixmapScaled)

    def nextImage(self):
        self.currentIndex += 1
        if self.currentIndex > len(self.imageFiles) - 1:
            self.currentIndex = 0
        self.showImages(self.imageFiles[self.currentIndex])

    def preImage(self):
        self.currentIndex -= 1
        if self.currentIndex < 0:
            self.currentIndex = len(self.imageFiles) - 1

        self.showImages(self.imageFiles[self.currentIndex])

    def generateResult(self):
        sticher = Sticher(self.imageFiles)
        new_img = sticher.stich_all()
        cv2.destroyAllWindows() ################

        self.currentIndex = 0
        self.imageFiles = None
        self.generateResultBtn.setEnabled(False)
        self.nextImageBtn.setEnabled(False)
        self.prevImageBtn.setEnabled(False)
        # self.showImages(self.imageFiles[0])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
