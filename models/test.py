import keras
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import numpy as np
import pandas as pd
import datetime as dt
import os
import glob
import time

# Select the model to be tested
out_dir = "trained_models/ResNet_MultiOut/20181230_120722/"
test_epoch = 400
model_name = glob.glob(os.path.join(out_dir, "weights.epoch_" + str(test_epoch) + "*hdf5"))[0]
test_model = keras.models.load_model(model_name) 

# Load the data
print("loading the data...")

file_dir = "../data/"
input_fname = "input.omnHistory_120.onsetDelTCutoff_2.omnDBRes_1.imfNormalize_True.shuffleData_True.npy" 
output_fname = "nBins_1.binTimeRes_30.onsetFillTimeRes_1.onsetDelTCutoff_2.omnHistory_120.omnDBRes_1.shuffleData_True.csv"


input_file = file_dir + input_fname
output_file = file_dir + output_fname
output_df_file = file_dir + "all_data." + output_fname
output_df_test_file = file_dir + "test_data." + output_fname

X = np.load(input_file)
df = pd.read_csv(output_file, index_col=0)
y = df.loc[:, "label"].values.reshape(-1, 1)

batch_size = 32
npoints = X.shape[0]
n_classes = np.unique(y).shape[0]

train_size = 0.75
val_size = 0.15
test_size = 0.1
train_eindex = int(npoints * train_size)
val_eindex = train_eindex + int(npoints * val_size)
x_train = X[:train_eindex, :]
x_val = X[train_eindex:val_eindex, :]
x_test = X[val_eindex:, :]
y_train = y[:train_eindex, :]
y_val = y[train_eindex:val_eindex, :]
y_test = y[val_eindex:, :]

# Encode the labels
enc = OneHotEncoder()
y_train_enc = enc.fit_transform(y_train).toarray()
y_test_enc = enc.fit_transform(y_test).toarray()
y_val_enc = enc.fit_transform(y_val).toarray()
y_enc = enc.fit_transform(y).toarray()

# Evaluate the model on test dataset
print("Evaluating the model...")
y_test_pred_enc = test_model.predict(x_test, batch_size=batch_size)
y_pred_enc = test_model.predict(X, batch_size=batch_size)

# The final activation layer uses softmax
y_test_pred = np.argmax(y_test_pred_enc , axis=1)
y_pred = np.argmax(y_pred_enc , axis=1)
y_test_true = y_test
y_true = y

# Report for all input data
print("Prediction report for all input data.")
print(classification_report(y_true, y_pred))

# Report for test data
print("Prediction report for test data.")
print(classification_report(y_test_true, y_test_pred))

# Save the predicted outputs
df.loc[:, "pred_label"] = y_pred
df_test = df.iloc[val_eindex:, :]
df_test.loc[:, "pred_label"] = y_test_pred
df.to_csv(output_df_file)
df_test.to_csv(output_df_test_file)



