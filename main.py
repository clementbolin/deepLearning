#!/usr/local/bin/python3

import sys

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras

# Import data
dataset = pd.read_csv('data/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

# Encode categorical data and scale continuous data
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
preprocess = make_column_transformer(
        (OneHotEncoder(), ['Geography', 'Gender']),
        (StandardScaler(), ['CreditScore', 'Age', 'Tenure', 'Balance',
                            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
                            'EstimatedSalary']))
X = preprocess.fit_transform(X)
X = np.delete(X, [0,3], 1)

# Split in train/test
y = y.values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from keras.models import Sequential
from keras.layers import Dense

# Initiation
classifier = Sequential()

# Create hidden layer (redresseur activation function)
classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform", input_dim=11))
classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))

# Create output layer
classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))

# Compile ANN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train ANN
numberEpoch = 100
if len(sys.argv) != 2:
    numberEpoch = 100
else:
    numberEpoch = int(sys.argv[1]) <= 0 and int(sys.argv[1]) or 100
    print(numberEpoch)
classifier.fit(X_train, y_train, batch_size=10, epochs=numberEpoch)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
precision = (cm[0][0] + cm[1][1]) / (cm[0][1] + cm[0][0] + cm[1][0] + cm[1][1]) * 100
print("precision : " + str(precision) + "%")

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""

Xnew = pd.DataFrame(data={
        'CreditScore': [600], 
        'Geography': ['France'], 
        'Gender': ['Male'],
        'Age': [40],
        'Tenure': [3],
        'Balance': [60000],
        'NumOfProducts': [2],
        'HasCrCard': [1],
        'IsActiveMember': [1],
        'EstimatedSalary': [50000]})
Xnew = preprocess.transform(Xnew)
Xnew = np.delete(Xnew, [0,3], 1)
new_prediction = classifier.predict(Xnew)
new_prediction = (new_prediction > 0.5)
print("""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000""")
if new_prediction == False:
    new_prediction = "he stay in bank"
else:
    new_prediction = "he leave bank"
print("prediction result : " + str(new_prediction))
