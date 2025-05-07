import joblib



def get_handwriting_input(hog_features):
    model = joblib.load("HOG_integral_model.pkl")
    digits = []
    prediction = model.predict(hog_features)

    digits.append(str(prediction[0]))

    return digits