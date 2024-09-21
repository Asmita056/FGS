# COLAB FILE

import os
import io
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objs as go
import plotly.io as pio
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


categories = ['Average','Best', 'Worst']
img_size = (128, 128)

def load_images(data_dir, categories, img_size):
    data = []
    labels = []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = tf.keras.preprocessing.image.load_img(os.path.join(path, img), target_size=img_size)
                img_array = tf.keras.preprocessing.image.img_to_array(img_array)
                data.append(img_array)
                labels.append(class_num)
            except Exception as e:
                 print(f"Error loading image {img}: {e}")
    return np.array(data), np.array(labels)

data_dir = "C:\\Users\\akash\\OneDrive\\Desktop\\Fruit Grading System\\FGS\\Dataset FGS"
data, labels = load_images(data_dir, categories, img_size)
data = data / 255.0
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

cnn_model = Sequential([
    # tf.keras.layers.Input(shape=(img_size[0], img_size[1], 3)),
    Conv2D(32, (3, 3), activation='relu',input_shape=(img_size[0], img_size[1], 3) ),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])
# cnn_model = Sequential([
#     tf.keras.layers.Input(shape=(img_size[0], img_size[1], 3)),
#     Conv2D(32, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(3, activation='softmax')
# ])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train, y_train, epochs=10, validation_split=0.2)
cnn_model.summary()

# grad_model = Model(cnn_model.inputs, cnn_model.get_layer('dense_1').output)
grad_model = Model(cnn_model.inputs, cnn_model.layers[-2].output)
features_train = grad_model.predict(X_train)
features_test = grad_model.predict(X_test)

cnn_y_pred = np.argmax(grad_model.predict(X_test), axis=-1)
cnn_y_test = np.argmax(y_test, axis=-1)
print("CNN Classification Report:")
print(classification_report(cnn_y_test, cnn_y_pred))
print("CNN Accuracy:", accuracy_score(cnn_y_test, cnn_y_pred))

y_train_flat = np.argmax(y_train, axis=1)
y_test_flat = np.argmax(y_test, axis=1)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(features_train, y_train_flat)
knn_y_pred = knn.predict(features_test)
print("KNN Classification Report:")
print(classification_report(y_test_flat, knn_y_pred))
print("KNN Accuracy:", accuracy_score(y_test_flat, knn_y_pred))

nb = GaussianNB()
nb.fit(features_train, y_train_flat)
nb_y_pred = nb.predict(features_test)
print("Naive Bayes Classification Report:")
print(classification_report(y_test_flat, nb_y_pred))
print("Naive Bayes Accuracy:", accuracy_score(y_test_flat, nb_y_pred))

rf = RandomForestClassifier(n_estimators=100)
rf.fit(features_train, y_train_flat)
rf_y_pred = rf.predict(features_test)
print("Random Forest Classification Report:")
print(classification_report(y_test_flat, rf_y_pred))
print("Random Forest Accuracy:", accuracy_score(y_test_flat, rf_y_pred))

svm = SVC(kernel='linear')
svm.fit(features_train, y_train_flat)
svm_y_pred = svm.predict(features_test)
print("SVM Classification Report:")
print(classification_report(y_test_flat, svm_y_pred))
print("SVM Accuracy:", accuracy_score(y_test_flat, svm_y_pred))

if y_train_flat.ndim > 1:
    y_train_flat_int = np.argmax(y_train_flat, axis=1)
else:
    y_train_flat_int = y_train_flat

if y_test_flat.ndim > 1:
    y_test_flat_int = np.argmax(y_test_flat, axis=1)
else:
    y_test_flat_int = y_test_flat

log_reg_model = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=1000)
log_reg_model.fit(features_train, y_train_flat_int)
y_pred_log_reg = log_reg_model.predict(features_test)

print("Logistic Regression Classification Report:")
print(classification_report(y_test_flat_int, y_pred_log_reg))

accuracy = accuracy_score(y_test_flat_int, y_pred_log_reg)
print(f"Accuracy: {accuracy:.2f}")

model_names = ['K-Nearest Neighbors', 'Gaussian Naive Bayes', 'Random Forest', 'Support Vector Machine', 'Convolutional Neural Network', 'Logistic Regression']
accuracies = [0.983273596176822, 0.984468339307049,  0.978494623655914, 0.984468339307049, 0.986857825567503, 0.98]

# Create a plotly line plot
fig = go.Figure()

for model_name, accuracy in zip(model_names, accuracies):
    fig.add_trace(go.Scatter(
        x=[model_name], y=[accuracy],
        mode='lines+markers+text',
        name=model_name
    ))

fig.update_layout(
    title='Model Accuracy Comparison',
    xaxis_title='Models',
    yaxis_title='Accuracy',
    showlegend=True
)

fig.show()



def predict_image(img_array):
    features = grad_model.predict(img_array)
    
    cnn_pred = np.argmax(cnn_model.predict(img_array), axis=-1)
    knn_pred = knn.predict(features)
    nb_pred = nb.predict(features)
    rf_pred = rf.predict(features)
    svm_pred = svm.predict(features)
    log_reg_pred = log_reg_model.predict(features)
    
    ## Convert predictions to category names
    # predictions = {
    #     'CNN': categories[cnn_pred[0]],
    #     'KNN': categories[knn_pred[0]],
    #     'Naive Bayes': categories[nb_pred[0]],
    #     'Random Forest': categories[rf_pred[0]],
    #     'SVM': categories[svm_pred[0]],
    #     'Logistic Regression': categories[log_reg_pred[0]]
    # }

    predictions = [
        cnn_pred[0],  
        knn_pred[0],  
        nb_pred[0],
        rf_pred[0],
        svm_pred[0],
        log_reg_pred[0]
    ]

    accuracy_percent = [
        
    ]

    category_predictions = [categories[pred] for pred in predictions]
    prediction_count = Counter(category_predictions)
    most_common_category = prediction_count.most_common(1)[0][0]
    return most_common_category
    

# def predict_image(img_array, true_label):
#     # Extract features for the input image
#     features = grad_model.predict(img_array)
    
#     # Individual model predictions
#     cnn_pred = np.argmax(cnn_model.predict(img_array), axis=-1)
#     knn_pred = knn.predict(features)
#     nb_pred = nb.predict(features)
#     rf_pred = rf.predict(features)
#     svm_pred = svm.predict(features)
#     log_reg_pred = log_reg_model.predict(features)
    
#     # Convert numerical predictions to category names
#     predictions = [
#         cnn_pred[0],  
#         knn_pred[0],  
#         nb_pred[0],
#         rf_pred[0],
#         svm_pred[0],
#         log_reg_pred[0]
#     ]
    
#     # Calculate the accuracy of each model by comparing with the true label
#     model_predictions = [cnn_pred[0], knn_pred[0], nb_pred[0], rf_pred[0], svm_pred[0], log_reg_pred[0]]
#     true_label_encoded = categories.index(true_label)

#     # Dynamic accuracies for each model
#     accuracies = [
#         1 if cnn_pred[0] == true_label_encoded else 0,
#         1 if knn_pred[0] == true_label_encoded else 0,
#         1 if nb_pred[0] == true_label_encoded else 0,
#         1 if rf_pred[0] == true_label_encoded else 0,
#         1 if svm_pred[0] == true_label_encoded else 0,
#         1 if log_reg_pred[0] == true_label_encoded else 0
#     ]
    
#     # Calculate the average accuracy across all models
#     average_accuracy = sum(accuracies) / len(accuracies)
    
#     # Convert numerical predictions to category names
#     category_predictions = [categories[pred] for pred in predictions]
    
#     # Count occurrences of each category to find the most common category
#     prediction_count = Counter(category_predictions)
#     most_common_category = prediction_count.most_common(1)[0][0]
    
#     # Return the most common prediction and average accuracy
#     return {
#         "most_common_category": most_common_category,
#         "average_accuracy": average_accuracy
#     }


    ## GIVE IMAGES MANUALLY THROUGH PATH
#     Print predictions
#     print("Predictions:")
#     for model, prediction in predictions.items():
#         print(f"{model}: {prediction}")

# Example usage
# image_path = 'C:\\Users\\akash\\OneDrive\\Desktop\\Fruit Grading System\\FGS\\test cases\\test5.jpg'  # Replace this with the actual path to your image
# predict_image(image_path)
