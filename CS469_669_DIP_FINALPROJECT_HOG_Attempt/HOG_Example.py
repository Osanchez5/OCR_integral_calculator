import os
import cv2 as cv
import joblib
import matplotlib.pyplot as plt
import handwriting_recognition
import numpy as np
from skimage.io import imread
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import hog
# Load dataset
def load_dataset(directory):
    hog_features = []
    class_names = []

    for label in os.listdir(directory):
        folder_path = os.path.join(directory, label)
        
        for img in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img)
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # Resize to 64x64
            img = cv.resize(img, (64, 64), interpolation = cv.INTER_LINEAR)

            feature = hog(img,
                            orientations=9,
                            pixels_per_cell=(4,4),
                            cells_per_block=(1,1),
                            block_norm='L2-Hys',
                            visualize=False)
            
            hog_features.append(feature)
            class_names.append(label)
    return np.array(hog_features), np.array(class_names)




hog_features, class_names = load_dataset("test_images/extracted_images")
print("HOG feature shape: ", hog_features.shape)


# Split data
X_train, X_test, Y_train, Y_test = train_test_split(hog_features, class_names, test_size=0.2, random_state=42)

# Train SVM
model = LinearSVC(max_iter=100000)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

# Evaluate
print("Accuracy: ", accuracy_score(Y_test, Y_pred))
print ("\nClassification Report:\n " , classification_report (Y_test , Y_pred))

# Store the model into a pkl file
# 