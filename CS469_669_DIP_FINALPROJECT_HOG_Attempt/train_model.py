import os
import cv2 as cv
import joblib
import matplotlib.pyplot as plt
import handwriting_recognition
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
# Load dataset
def load_dataset(directory):
    # Creates a dataset from images that can be fed into a SVM model 
    hog_features = []
    class_names = []

    for label in os.listdir(directory):
        folder_path = os.path.join(directory, label)
        
        for img in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img)
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # Resize to 64x128
            img = resize_image_to_desired_size(img)
            feature = handwriting_recognition.HOG_Testing(img)
            
            hog_features.append(feature)
            class_names.append(label)
    return np.array(hog_features), np.array(class_names)

def resize_image_to_desired_size(image_array):
    # Get the original width and height
    original_height, original_width = image_array.shape
    # Rewrite the code to resize to 128x128 instead


    target_height = 64
    target_width = 128

    # Find a new scaling factor
    scale = min(target_width / original_width, target_height / original_height)

    # Determine the new size
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize the image 128x64
    resized_image = cv.resize(image_array, (new_width, new_height), interpolation=cv.INTER_AREA)

    # Create new blank image with target size
    result = np.ones((target_height, target_width), dtype=np.uint8)
    result = result *255

    # Compute top-left corner for centering the resized image
    top = (target_height - new_height) // 2
    left = (target_width - new_width) // 2

    # Paste the resized image onto the center
    result[top:top+new_height, left:left+new_width] = resized_image

    return result



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
joblib.dump((model), "HOG_integral_model.pkl", compress = 3)