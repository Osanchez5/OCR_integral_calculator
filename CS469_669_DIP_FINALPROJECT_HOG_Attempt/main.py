import sys
import handwriting_recognition
import numpy as np
# import integral_math
import use_model
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


# The code for the image processing should probably go in here
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

       #  self.button_is_checked = True
        self.setWindowTitle("Integral Calculator")
        self.setMinimumSize(QtCore.QSize(1000,750))

        self.select_image = QPushButton("Select Image")
        self.select_image.setFixedSize(QtCore.QSize(75, 25))
        
        # self.select_image.clicked.connect(self.the_button_was_clicked)
        # select_image.clicked.connect(self.the_button_was_toggled)
        self.setCentralWidget(self.select_image)

        self.select_image.clicked.connect(self.load_image)

                
    def load_image(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        feature_descriptor = None
        while True:


            image_path = file_dialog.getOpenFileName(self, 'Select Desired Image', '', 'Image Files (*.png *.jpg)')
            print(image_path[0])
            feature_descriptor = handwriting_recognition.read_in_image(image_path[0])
            if feature_descriptor is None:
                print("The image couldn't be read.")
            else:
                break
        feature_descriptor = np.array(feature_descriptor)
        # print(feature_descriptor.shape)
        # Apply the machine learning model to it
        # digits = use_model.get_handwriting_input(feature_descriptor.reshape(1, -1))

        # print(digits)
        # Run it through integral_math.py
        # math_answer = integral_math.whateverInameit


if __name__ == "__main__":
    # Run the progrma here
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    app.exec()