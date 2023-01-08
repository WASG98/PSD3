import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC

# Preprocessing function to convert images to grayscale and apply smoothing
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (3,3), 0)
    return smooth

# Pothole detection function using a sliding window approach
def detect_potholes(img, classifier):
    potholes = []
    window_size = (50, 50)
    for y in range(0, img.shape[0]-window_size[1], window_size[1]):
        for x in range(0, img.shape[1]-window_size[0], window_size[0]):
            window = img[y:y+window_size[1], x:x+window_size[0]]
            features = extract_features(window)
            prediction = classifier.predict([features])
            if prediction == 1:
                potholes.append((x, y, window_size[0], window_size[1]))
    return potholes

# Feature extraction function using edge detection and shape analysis
def extract_features(img):
    edges = cv2.Canny(img, 50, 150)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        (x,y), radius = cv2.minEnclosingCircle(c)
        diameter = radius * 2
        return [diameter]
    else:
        return [0]

# Load training data and train classifiers
X_train, y_train = load_training_data()
rf_classifier = RandomForestClassifier(n_estimators=100)
svc_classifier = SVC(kernel='linear', probability=True)
ensemble_classifier = VotingClassifier(estimators=[('rf', rf_classifier), ('svc', svc_classifier)], voting='soft')
ensemble_classifier.fit(X_train, y_train)

# Load test data and apply pothole detection and classification
X_test, y_test = load_test_data()
accuracies = []
for i in range(len(X_test)):
    img = X_test[i]
    true_label = y_test[i]
    preprocessed_img = preprocess(img)
    potholes = detect_potholes(preprocessed_img, ensemble_classifier)
    severity_predictions = []
    for p in potholes:
        x, y, w, h = p
        pothole_img = img[y:y+h, x:x+w]
        depth, width = estimate_magnitude(pothole_img)
        severity_predictions.append(classify_severity(depth, width, 1))
    severity_majority = max(set(severity_predictions), key=severity_predictions.count)
    if severity_majority == true_label:
        accuracies.append(1)
    else:
        accuracies.append(0)
average_accuracy = sum(accuracies) / len(accuracies)
print("Average accuracy:", average_accuracy)

# Estimate the depth and width (area) of the pothole using 3D reconstruction techniques
def estimate_magnitude(img):
    depth, width = 0, 0
    # Use stereo vision or SfM to estimate depth
    depth = estimate_depth(img)
    # Use depth and image to estimate width (area)
    width = estimate_width(img, depth)
    return depth, width

# Classify the severity of the pothole based on depth, width (area), and quantity
# Classify the severity of the pothole based on depth, width (area), and quantity
def classify_severity(depth, width, quantity):
    if depth > 5 and width > 50 and quantity > 5:
        return "Severe"
    elif depth > 2 and width > 25 and quantity > 2:
        return "Moderate"
    else:
        return "Mild"

# Main function to run pothole detection and classification on a dataset of images
def run_pothole_detection(X, y, classifier):
    accuracies = []
    for i in range(len(X)):
        img = X[i]
        true_label = y[i]
        preprocessed_img = preprocess(img)
        potholes = detect_potholes(preprocessed_img, classifier)
        severity_predictions = []
        for p in potholes:
            x, y, w, h = p
            pothole_img = img[y:y+h, x:x+w]
            depth, width = estimate_magnitude(pothole_img)
            severity_predictions.append(classify_severity(depth, width, 1))
        severity_majority = max(set(severity_predictions), key=severity_predictions.count)
        if severity_majority == true_label:
            accuracies.append(1)
        else:
            accuracies.append(0)
    average_accuracy = sum(accuracies) / len(accuracies)
    return average_accuracy

# Load training and test data
X_train, y_train = load_training_data()
X_test, y_test = load_test_data()

# Train and evaluate classifiers
rf_classifier = RandomForestClassifier(n_estimators=100)
svc_classifier = SVC(kernel='linear', probability=True)
ensemble_classifier = VotingClassifier(estimators=[('rf', rf_classifier), ('svc', svc_classifier)], voting='soft')
ensemble_classifier.fit(X_train, y_train)
average_accuracy = run_pothole_detection(X_test, y_test, ensemble_classifier)
print("Average accuracy:", average_accuracy)



           
