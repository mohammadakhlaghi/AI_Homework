# -*- coding: utf-8 -*-
"""
Created on Wed May 24 16:12:14 2023

@author: HP
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def step1_ver1(dataset):
    # Load the dataset
    data = pd.read_csv(dataset)

    # Encode class names with numeric labels
    label_encoder = LabelEncoder()
    data['Class'] = label_encoder.fit_transform(data['Class'])

    # Normalize the features using Min-Max scaling
    scaler = MinMaxScaler()
    data[['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']] = scaler.fit_transform(data[['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']])

    return data

# Preprocess the Iris dataset using step1_ver1
preprocessed_data = step1_ver1('Iris.csv')

# Display the preprocessed dataset
print(preprocessed_data.head())


from sklearn.model_selection import train_test_split

def step2_ver1(preprocessed_data):
    # Separate features and target variable
    X = preprocessed_data[['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']]
    y = preprocessed_data['Class']

    # Split the data into training and test sets, stratified by the target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Concatenate the features and target variable for each set
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    return train_data, test_data,X_train, X_test, y_train, y_test

# Split the preprocessed data using step2_ver1
train_data, test_data,X_train, X_test, y_train, y_test = step2_ver1(preprocessed_data)



from sklearn.neighbors import KNeighborsClassifier

def step3_ver2(train_data, test_data):
    # Separate features and target variable from training data
    X_train = train_data[['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']]
    y_train = train_data['Class']

    # Separate features and target variable from test data
    X_test = test_data[['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']]
    y_test = test_data['Class']

    # Create a KNN classifier with K=5
    knn = KNeighborsClassifier(n_neighbors=5)

    # Train the KNN model
    knn.fit(X_train, y_train)

    # Predict on training and test data
    train_predictions = knn.predict(X_train)
    test_predictions = knn.predict(X_test)

    return knn, train_predictions, test_predictions, y_test

# Train a KNN model using step3_ver2
knn_model, train_predictions, test_predictions, y_test = step3_ver2(train_data, test_data)


from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, jaccard_score

# Predict on the test data
y_pred = knn_model.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate precision, recall, accuracy, and Jaccard similarity score
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
accuracy = accuracy_score(y_test, y_pred)
jaccard = jaccard_score(y_test, y_pred, average='macro')

# Display the confusion matrix and evaluation metrics
print("Confusion Matrix:")
print(cm)
print("\nPrecision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
print("Jaccard Similarity Score:", jaccard)

