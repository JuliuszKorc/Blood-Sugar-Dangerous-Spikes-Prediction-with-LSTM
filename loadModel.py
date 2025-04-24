import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

# Load data from the file
data = pd.read_csv('cgmTest.txt', sep='|')
data["CGM"] = data["CGM"].apply(pd.to_numeric)
values = data["CGM"]

# A function that divdes the data into all possible sequeneces that predict 15 minutes of future measurements
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

# load the model and predict the results.
model = load_model('modelLSTM/model.keras')

# train_predictions = model.predict(X_train)
train_predictions = model.predict(X_test)


# plot some results
train_results1 = pd.DataFrame(
    data={"Train predictions": train_predictions[1950], 'Actuals': Y_test[1950]})

train_results2 = pd.DataFrame(
    data={"Train predictions": train_predictions[2000], 'Actuals': Y_test[2000]})


plt.plot(train_results1['Train predictions'],
         label='prediction', color='magenta')
plt.plot(train_results1['Actuals'], label='actual', color='cyan')

plt.legend()
plt.show()

plt.plot(train_results2['Train predictions'],
         label='prediction', color='magenta')
plt.plot(train_results2['Actuals'], label='actual', color='cyan')
plt.legend()
plt.show()
