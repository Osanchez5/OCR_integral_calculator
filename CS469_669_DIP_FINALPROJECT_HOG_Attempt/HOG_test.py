import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import handwriting_recognition

# Load dataset

digits = datasets.load_digits()
images = digits.images
labels = digits.target


print("Image shape: ", images[0].shape)

# Extract HOG features
hog_features = []
for img in images:
    feature = handwriting_recognition.HOG_Testing(img)
    hog_features.append(feature)

hog_features = np.array(hog_features)
print("HOG feature shape: ", hog_features.shape)

print(hog_features[:5])

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(hog_features, labels, test_size=0.2, random_state=42)

# Train SVM
model = LinearSVC()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

# Evaluate
print("Accuracy: ", accuracy_score(Y_test, Y_pred))
print ("\nClassification Report:\n " , classification_report (Y_test , Y_pred))

# Visualize prediction
fig, axews = plt.subplots(2, 5, figsize = (10, 5))

for i, ax in enumerate(axews.flat):
    ax.imshow(images[i], cmap = 'gray')
    ax.set_title(f"Pred {model.predict([hog_features[i]])[0]}")
    ax.axis('off')
plt.tight_layout()
plt.savefig('digit_predictions.png')
