import sys
from tkinter import Tk

from PyQt5.QtWidgets import QScrollArea, QGridLayout, QMainWindow, QPushButton, QLabel, QApplication, QWidget, \
    QInputDialog, QLineEdit, QFileDialog, QFrame, QComboBox
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QIcon, QPixmap
from timeit import default_timer as timer

from stitching import Stitcher
import cv2
import time
import numpy as np


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.fileNameInput = None
        self.fileNameLabel = None
        self.focalInpput = None
        self.focalLabel = None
        self.compStyle = None
        self.compLabel = None
        self.gridLayout = None
        # self.root = Tk()
        self.title = 'Panorama Image Stitching'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 680
        self.selectImageBtn = None
        self.generateResultBtn = None
        self.numSelectLabel = None
        self.currentIndexLabel = None

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
        self.numSelectLabel.move(30, 40)

        self.selectImageBtn = QPushButton("Select", self)
        self.selectImageBtn.move(120, 40)
        self.selectImageBtn.setEnabled(True)
        self.selectImageBtn.clicked.connect(self.openFileNamesDialog)

        self.generateResultBtn = QPushButton("Generate", self)
        self.generateResultBtn.move(500, 10)
        self.generateResultBtn.setEnabled(False)
        self.generateResultBtn.clicked.connect(self.generateResult)

        # self.line = QFrame(self)
        # self.line.move(280, 60)
        # self.line.setFrameShape(QFrame.HLine)

        self.currentIndexLabel = QLabel(self)
        self.currentIndexLabel.setText("No Image Selected.")
        self.currentIndexLabel.move(280, 580)

        self.prevImageBtn = QPushButton("Prev Image", self)
        self.prevImageBtn.move(200, 620)
        self.prevImageBtn.setEnabled(False)
        self.prevImageBtn.clicked.connect(self.preImage)

        self.nextImageBtn = QPushButton("Next Image", self)
        self.nextImageBtn.move(350, 620)
        self.nextImageBtn.setEnabled(False)
        self.nextImageBtn.clicked.connect(self.nextImage)

        self.compLabel = QLabel(self)
        self.compLabel.setText("Compositing")
        self.compLabel.move(30, 10)

        self.compStyle = QComboBox(self)
        self.compStyle.addItem("Flat")
        self.compStyle.addItem("Cylindrical")
        self.compStyle.addItem("Spherical")
        self.compStyle.move(120, 10)
        self.compStyle.setEnabled(False)
        self.compStyle.activated[str].connect(self.chooseCompStyle)

        self.labelImage = QLabel(self)
        self.labelImage.move(300, 340)
        self.labelImage.setText("No Result")

        self.focalLabel = QLabel(self)
        self.focalLabel.setText("Focal Length")
        self.focalLabel.move(250, 10)

        self.focalInpput = QLineEdit(self)
        self.focalInpput.move(340, 10)
        self.focalInpput.setEnabled(False)
        self.focalInpput.setText("800")

        self.fileNameLabel = QLabel(self)
        self.fileNameLabel.setText("Save As")
        self.fileNameLabel.move(250, 40)

        self.fileNameInput = QLineEdit(self)
        self.fileNameInput.setText("new_img.jpg")
        self.fileNameInput.move(340, 40)
        self.fileNameInput.setEnabled(False)

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.show()

    def openFileNamesDialog(self):
        # https://www.tutorialspoint.com/pyqt/pyqt_qfiledialog_widget.htm
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "QFileDialog.getOpenFileNames()", "",
                                                "Images (*.jpg *.jpeg *.png *.bmp)", options=options)
        if files:
            print(files)
            self.currentIndex = 0
            self.imageFiles = files
            self.numSelectLabel.setText("Select " + str(len(self.imageFiles)) + " images")
            self.currentIndexLabel.setText("Current Index " + str(self.currentIndex))
            self.showImages(self.imageFiles[0])
            self.generateResultBtn.setEnabled(True)
            self.nextImageBtn.setEnabled(True)
            self.prevImageBtn.setEnabled(True)
            if self.compStyle.currentText() != "Flat":
                self.focalInpput.setEnabled(True)
            else:
                self.focalInpput.setEnabled(False)
            self.fileNameInput.setEnabled(True)
            self.compStyle.setEnabled(True)

    def showImages(self, filename):
        self.labelImage.move(0, 100)
        self.labelImage.resize(640, 480)
        pixmap = QPixmap(filename)
        pixmapScaled = pixmap.scaled(640, 480, Qt.KeepAspectRatio)
        self.labelImage.setPixmap(pixmapScaled)

    def nextImage(self):
        self.currentIndex += 1
        if self.currentIndex > len(self.imageFiles) - 1:
            self.currentIndex = 0
        self.showImages(self.imageFiles[self.currentIndex])
        self.currentIndexLabel.setText("Current Index " + str(self.currentIndex))

    def preImage(self):
        self.currentIndex -= 1
        if self.currentIndex < 0:
            self.currentIndex = len(self.imageFiles) - 1

        self.showImages(self.imageFiles[self.currentIndex])
        self.currentIndexLabel.setText("Current Index " + str(self.currentIndex))

    def generateResult(self):
        # self.labelImage.move(300, 80)
        # self.labelImage.setText("Loading......")

        self.labelImage.move(300, 80)
        self.labelImage.setText("Loading......")
        QApplication.processEvents()
        self.stitchImage()
        #
        # callback()
        # # self.update(self)
        # self.root.after_idle(callback)

    def chooseCompStyle(self):
        if self.compStyle.currentText() != "Flat":
            self.focalInpput.setEnabled(True)
        else:
            self.focalInpput.setEnabled(False)

    def stitchImage(self):
        # nonlocal self
        start = timer()
        stitcher = Stitcher(self.imageFiles,f=int(self.focalInpput.text()),mode=self.compStyle.currentText())
        try:
            new_img = stitcher.stitch_all()
            end = timer()
            print("Total Time:",end - start)
            cv2.imwrite(self.fileNameInput.text(), new_img)
            self.showImages(self.fileNameInput.text())
        except Exception:
            print("!!!!!!!!!!!!!")
            self.labelImage.move(150, 80)
            self.labelImage.setText("Failed to stitch the images, try other compositing method please.")
        # cv2.destroyAllWindows()  ################

        # self.currentIndex = 0
        # self.numSelectLabel.setText("Select " + str(len(self.imageFiles)) + " images")
        # self.imageFiles = None
        # self.generateResultBtn.setEnabled(False)
        # self.nextImageBtn.setEnabled(False)
        # self.prevImageBtn.setEnabled(False)
        # self.showImages(self.imageFiles[0])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
