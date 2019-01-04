import keras
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from dnn_classifiers import ResNet
from scipy.io import loadmat
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

#skip_training = True
skip_training = False

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

# create out_dir
out_dir="./trained_models/ResNet/" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# Build a ResNet model
optimizer=keras.optimizers.Adam(lr=0.0001)
batch_size = 64
n_epochs = 400
n_resnet_units = 3
metrics = ["accuracy"]
input_shape = X.shape[1:]

# Define the loss, loss_weights, and class_weights
loss=keras.losses.categorical_crossentropy

#from sklearn import utils
class_weights = None

resnet = ResNet(input_shape, batch_size=batch_size, n_epochs=n_epochs,
                n_classes=n_classes, n_resnet_units=n_resnet_units, loss=loss,
                optimizer=optimizer,
                metrics=metrics, out_dir=out_dir)

# Train the model
if not skip_training:
    print("Training the model...")
    fit_history = resnet.train_model(x_train, y_train_enc, x_val, y_val_enc,
                                     class_weights=class_weights)

# Plot the training 
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
xs = np.arange(n_epochs)
train_loss = fit_history.history["loss"]
val_loss = fit_history.history["val_loss"]
train_acc = fit_history.history["acc"]
val_acc = fit_history.history["val_acc"]
axes[0].plot(xs, train_loss, label="train_loss") 
axes[0].plot(xs, val_loss, label="val_loss") 
axes[1].plot(xs, train_acc, label="train_acc") 
axes[1].plot(xs, val_acc, label="val_acc") 
axes[0].set_title("Training Loss and Accuracy")
axes[0].set_ylabel("Loss")
axes[1].set_ylabel("Accuracy")
axes[1].set_xlabel("Epoch")
axes[0].legend()
axes[1].legend()
fig_path = os.path.join(out_dir, "loss_acc")
fig.savefig(fig_path + ".png", dpi=200, bbox_inches="tight")  

# Evaluate the model on test dataset
print("Evaluating the model...")
#test_epoch = 460
test_epoch = n_epochs
model_name = glob.glob(os.path.join(out_dir, "weights.epoch_" + str(test_epoch) + "*hdf5"))[0]
test_model = keras.models.load_model(model_name) 
y_pred_enc = test_model.predict(X, batch_size=batch_size)
y_test_pred_enc = test_model.predict(x_test, batch_size=batch_size)

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

