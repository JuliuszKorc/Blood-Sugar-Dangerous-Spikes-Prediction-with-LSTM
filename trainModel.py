import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

# Load data from the file
data = pd.read_csv('cgmTest.txt', sep='|')
data["CGM"] = data["CGM"].apply(pd.to_numeric)
values = data["CGM"]

# A function that divdes the data into all possible sequeneces that predict fifteen minutes of future measurements
# based on the last 40 minutes of glucose level measurements.


def df_to_XY(df, window_size=8, time_to_predict=3):
    NP = df.to_numpy()
    X = []
    Y = []
    for i in range(len(NP) - window_size-time_to_predict):
        row = [[a] for a in NP[i:i+window_size]]
        X.append(row)
        Y.append(NP[i+window_size-1:i+window_size+time_to_predict])
    return np.array(X), np.array(Y)


X, Y = df_to_XY(values)

# distribute the data into train, test and validation sets.
train_size = int(X.shape[0] * 0.8)
validation_size = train_size + int((int(X.shape[0])-train_size) / 2)

X_train, Y_train = X[:train_size], Y[:train_size]
X_val, Y_val = X[train_size:validation_size], Y[train_size: validation_size]

X_test, Y_test = X[validation_size:], Y[validation_size:]

# Build a simple LSTM
model1 = Sequential()
model1.add(InputLayer((8, 1)))
model1.add(LSTM(64))
model1.add(Dense(8, 'relu'))
model1.add(Dense(4))

# Set the checkpoint and saving setings
cp = ModelCheckpoint('modelLSTM/model.keras', save_best_only=True)

# Set the learning rate, optimizer and metrics
model1.compile(loss=MeanSquaredError(), optimizer=Adam(
    learning_rate=0.0001), metrics=[RootMeanSquaredError()])

# Train the model
model1.fit(X_train, Y_train, validation_data=(
    X_val, Y_val), epochs=100, callbacks=[cp])
