In order for the tesseract OCR model to work, the handwriting_num.traineddata file has to be put into the tessdata directory for Tesseract-OCR. At least on windows. The program assumes that it is being run on windows.

Test Data from https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols/data

Used to fine tune the OCR model used for the program
https://github.com/tesseract-ocr/tesstrain
The model specifically used is the eng.traineddata model where it is a "best" model.

The final project is not the HOG implementation but rather the OCR implementation.
Where the program is executed by running py main.py. The images have to be manually passed through by editing the main python file.