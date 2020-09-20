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
