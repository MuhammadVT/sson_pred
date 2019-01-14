import keras
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from dnn_classifiers import MLSTM_FCN, train_model
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

nBins = 1
binTimeRes = 60
imfNormalize = True
shuffleData = False 
polarData = True
imageData = True
omnHistory = 180
onsetDelTCutoff = 3
onsetFillTimeRes = 1
omnDBRes = 1

batch_size = 64 * 10
n_epochs = 20
metrics = ["accuracy"]

file_dir = "../data/"
output_fname = "nBins_" + str(nBins) + "." +\
               "binTimeRes_" + str(binTimeRes) + "." +\
               "onsetFillTimeRes_" + str(onsetFillTimeRes) + "." +\
               "onsetDelTCutoff_" + str(onsetDelTCutoff) + "." +\
               "omnHistory_" + str(omnHistory) + "." +\
               "omnDBRes_" + str(omnDBRes) + "." +\
               "shuffleData_" + str(shuffleData) + "." +\
               "polarData_" + str(polarData) + "." +\
               "imageData_" + str(imageData) + "." +\
               "csv"

input_fname = "input." +\
              "nBins_" + str(nBins) + "." +\
              "binTimeRes_" + str(binTimeRes) + "." +\
              "omnHistory_" + str(omnHistory) + "." +\
              "onsetDelTCutoff_" + str(onsetDelTCutoff) + "." +\
              "omnDBRes_" + str(omnDBRes) + "." +\
              "imfNormalize_" + str(imfNormalize) + "." +\
              "shuffleData_" + str(shuffleData) + "." +\
              "polarData_" + str(polarData) + "." +\
              "imageData_" + str(imageData) + "." +\
              "npy"

#out_dir="./trained_models/MLSTM_FCN/20190104_113412/"
out_dir="./trained_models/MLSTM_FCN/" +\
        "nBins_" + str(nBins) + "." +\
        "binTimeRes_" + str(binTimeRes) + "." +\
        "onsetFillTimeRes_" + str(onsetFillTimeRes) + "." +\
        "omnHistory_" + str(omnHistory) + "." +\
        "omnDBRes_" + str(omnDBRes) + "." +\
	dt.datetime.now().strftime("%Y%m%d.%H%M%S")

# create out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

input_file = file_dir + input_fname
output_file = file_dir + output_fname

# Load the data
print("loading the data...")
X = np.load(input_file)
df = pd.read_csv(output_file, index_col=0)
y = df.loc[:, "label"].values.reshape(-1, 1)

# Do 5-min average to the input data
X = np.mean(X[:, :-1, :].reshape(X.shape[0], int((X.shape[1]-1)/5), 5, X.shape[-1]), axis=2)
X = X[::5, :-1, :]
y = y[::5, :]

npoints = X.shape[0]
n_classes = np.unique(y).shape[0]

train_size = 0.75
val_size = 0.15
test_size = 0.10
train_eindex = int(npoints * train_size)
val_eindex = train_eindex + int(npoints * val_size)
x_train = X[:train_eindex, :]
x_val = X[train_eindex:val_eindex, :]
x_test = X[val_eindex:, :]
y_train = y[:train_eindex, :]
y_val = y[train_eindex:val_eindex, :]
y_test = y[val_eindex:, :]

## Shuffle the training data
#idx_train = np.array(range(x_train.shape[0]))
#np.random.shuffle(idx_train)
#x_train = x_train[idx_train, :, :]
#y_train = y_train[idx_train, :]

# Encode the labels
enc = OneHotEncoder()
unique_labels = np.unique(y, axis=0)
enc.fit(unique_labels)
y_train_enc = enc.transform(y_train).toarray()
y_test_enc = enc.transform(y_test).toarray()
y_val_enc = enc.transform(y_val).toarray()
y_enc = enc.transform(y).toarray()

# Build a MLSTM_FCN model
optimizer=keras.optimizers.Adam(lr=0.0001)
input_shape = X.shape[1:]

# Define the loss, loss_weights, and class_weights
loss=keras.losses.categorical_crossentropy

#from sklearn import utils
#class_weights = {0:1, 1:1.3}
class_weights = None

# Train the model
if not skip_training:
    dl_obj = MLSTM_FCN(input_shape, batch_size=batch_size, n_epochs=n_epochs,
                       n_classes=n_classes, loss=loss, optimizer=optimizer,
                       metrics=metrics, out_dir=out_dir)

    print("Training the model...")
    dl_obj.model.summary()
    fit_history = train_model(dl_obj.model, x_train, y_train_enc, x_val, y_val_enc,
                              batch_size=batch_size, n_epochs=n_epochs,
                              callbacks=dl_obj.callbacks, shuffle=True,
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
test_epoch = n_epochs
model_name = glob.glob(os.path.join(out_dir, "weights.epoch_" + str(test_epoch) + "*hdf5"))[0]
test_model = keras.models.load_model(model_name) 
y_train_pred_enc = test_model.predict(x_train, batch_size=batch_size)
y_test_pred_enc = test_model.predict(x_test, batch_size=batch_size)

# The final activation layer uses softmax
y_test_pred = np.argmax(y_test_pred_enc , axis=1)
y_train_pred = np.argmax(y_train_pred_enc , axis=1)
y_test_true = y_test
y_train_true = y_train

# Report for all input data
print("Prediction report for train data.")
print(classification_report(y_train_true, y_train_pred))

# Report for test data
print("Prediction report for test data.")
print(classification_report(y_test_true, y_test_pred))

