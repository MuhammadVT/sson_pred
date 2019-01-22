import keras
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from dnn_regressors import ResNet_Reg, train_model
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
onsetDelTCutoff = 4
onsetFillTimeRes = 5
omnDBRes = 1

batch_size = 32 * 10
n_epochs = 50
n_resnet_units = 30
metrics = ["mae"]

useSML = True
smlDateRange = [dt.datetime(1997,1,1), dt.datetime(2007,12,31)]
smlStrtStr = smlDateRange[0].strftime("%Y%m%d")
smlEndStr = smlDateRange[1].strftime("%Y%m%d")
#omnTrainParams = ["Bz", "Vx", "Np"]
omnTrainParams = ["Bx", "By", "Bz", "Vx", "Np"]
# since we have different omnTrainParams for different datasets
# we'll create seperate folders for them for simplicity
omnDir = "omn_"
for _nom, _npm in enumerate(omnTrainParams):
    omnDir += _npm
    if _nom < len(omnTrainParams)-1:
        omnDir += "_"
    else:
        omnDir += "/"

if useSML:
    print("Using SML data")
    input_file = "../data/" + omnDir + "input." +\
                 "nBins_" + str(nBins) + "." +\
                 "binTimeRes_" + str(binTimeRes) + "." +\
                 "onsetFillTimeRes_" + str(onsetFillTimeRes) + "." +\
                 "onsetDelTCutoff_" + str(onsetDelTCutoff) + "." +\
                 "omnHistory_" + str(omnHistory) + "." +\
                 "omnDBRes_" + str(omnDBRes) + "." +\
                 "imfNormalize_" + str(imfNormalize) + "." +\
                 "shuffleData_" + str(shuffleData) + "." +\
                 "dateRange_" + smlStrtStr + "_" + smlEndStr + "." +\
                 "npy"

    csv_file = "../data/" + omnDir + "sml_" +\
               "nBins_" + str(nBins) + "." +\
               "binTimeRes_" + str(binTimeRes) + "." +\
               "onsetFillTimeRes_" + str(onsetFillTimeRes) + "." +\
               "onsetDelTCutoff_" + str(onsetDelTCutoff) + "." +\
               "omnHistory_" + str(omnHistory) + "." +\
               "omnDBRes_" + str(omnDBRes) + "." +\
               "imfNormalize_" + str(imfNormalize) + "." +\
               "shuffleData_" + str(shuffleData) + "." +\
               "dateRange_" + smlStrtStr + "_" + smlEndStr + "." +\
               "csv"

    out_dir="./trained_models/ResNet_Reg/" + omnDir + \
            "sml.nBins_" + str(nBins) + "." +\
            "binTimeRes_" + str(binTimeRes) + "." +\
            "onsetFillTimeRes_" + str(onsetFillTimeRes) + "." +\
            "omnHistory_" + str(omnHistory) + "." +\
            "omnDBRes_" + str(omnDBRes) + "." +\
            "useSML_" + str(useSML) + "." +\
            dt.datetime.now().strftime("%Y%m%d.%H%M%S")

else:
    input_file = "../data/input." + omnDir +\
                 "nBins_" + str(nBins) + "." +\
                 "binTimeRes_" + str(binTimeRes) + "." +\
                 "onsetFillTimeRes_" + str(onsetFillTimeRes) + "." +\
                 "onsetDelTCutoff_" + str(onsetDelTCutoff) + "." +\
                 "omnHistory_" + str(omnHistory) + "." +\
                 "omnDBRes_" + str(omnDBRes) + "." +\
                 "imfNormalize_" + str(imfNormalize) + "." +\
                 "shuffleData_" + str(shuffleData) + "." +\
                 "polarData_" + str(polarData) + "." +\
                 "imageData_" + str(imageData) + "." +\
                 "npy"

    csv_file = "../data/"  + omnDir +\
               "nBins_" + str(nBins) + "." +\
               "binTimeRes_" + str(binTimeRes) + "." +\
               "onsetFillTimeRes_" + str(onsetFillTimeRes) + "." +\
               "onsetDelTCutoff_" + str(onsetDelTCutoff) + "." +\
               "omnHistory_" + str(omnHistory) + "." +\
               "omnDBRes_" + str(omnDBRes) + "." +\
               "imfNormalize_" + str(imfNormalize) + "." +\
               "shuffleData_" + str(shuffleData) + "." +\
               "polarData_" + str(polarData) + "." +\
               "imageData_" + str(imageData) + "." +\
               "csv"

    out_dir="./trained_models/ResNet_Reg/"  + omnDir +\
            "nBins_" + str(nBins) + "." +\
            "binTimeRes_" + str(binTimeRes) + "." +\
            "onsetFillTimeRes_" + str(onsetFillTimeRes) + "." +\
            "omnHistory_" + str(omnHistory) + "." +\
            "omnDBRes_" + str(omnDBRes) + "." +\
            dt.datetime.now().strftime("%Y%m%d.%H%M%S")

# create out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

# Load the data
print("loading the data...")
X = np.load(input_file)
df = pd.read_csv(csv_file, index_col=0)
y = df.loc[:, "del_minutes"].values.reshape(-1, 1)

# Chop off some part of X
#X = X[:, 120:, :]

# Remove all -1 values
_idx = np.where(y>=0)
_idx = _idx[0]
X = X[_idx, :, :]
y = y[_idx, :]

npoints = X.shape[0]
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

# Build a ResNet_Reg model
optimizer=keras.optimizers.Adam(lr=0.0001)
#optimizer=keras.optimizers.RMSprop(lr=0.0001)
input_shape = X.shape[1:]

# Define the loss, loss_weights, and class_weights
loss=keras.losses.mean_squared_error
#loss=keras.losses.mean_absolute_error

# Train the model
if not skip_training:
    dl_mdl = ResNet_Reg(input_shape, batch_size=batch_size, n_epochs=n_epochs,
                   loss=loss, optimizer=optimizer, n_resnet_units=n_resnet_units,
                   metrics=metrics, out_dir=out_dir)

    print("Training the model...")
    dl_mdl.model.summary()
    fit_history = train_model(dl_mdl.model, x_train, y_train, x_val, y_val,
                              batch_size=batch_size, n_epochs=n_epochs,
                              callbacks=dl_mdl.callbacks, shuffle=False)

    # Plot the training 
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    xs = np.arange(n_epochs)
    train_loss = fit_history.history["loss"]
    val_loss = fit_history.history["val_loss"]
    train_mae = fit_history.history["mean_absolute_error"]
    val_mae = fit_history.history["val_mean_absolute_error"]
    axes[0].plot(xs, train_loss, label="train_loss") 
    axes[0].plot(xs, val_loss, label="val_loss") 
    axes[1].plot(xs, train_mae, label="train_mae") 
    axes[1].plot(xs, val_mae, label="val_mae") 
    axes[0].set_title("Training Loss and MAE")
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("MAE")
    axes[1].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].legend()
    fig_path = os.path.join(out_dir, "loss_mae")
    fig.savefig(fig_path + ".png", dpi=200, bbox_inches="tight")  

# Evaluate the model on test dataset
print("Evaluating the model...")
test_epoch = n_epochs
model_name = glob.glob(os.path.join(out_dir, "weights.epoch_" + str(test_epoch) + "*hdf5"))[0]
test_model = keras.models.load_model(model_name) 
y_train_pred = test_model.predict(x_train, batch_size=batch_size)
y_val_pred = test_model.predict(x_val, batch_size=batch_size)
y_test_pred = test_model.predict(x_test, batch_size=batch_size)

y_train_true = y_train
y_val_true = y_val
y_test_true = y_test

# Report for train data
print("Prediction report for train data.")
print("Mean Absolute Error:", mean_absolute_error(y_train_true, y_train_pred))

# Report for validation data
print("Prediction report for validation data.")
print("Mean Absolute Error:", mean_absolute_error(y_val_true, y_val_pred))

# Report for test data
print("Prediction report for test data.")
print("Mean Absolute Error:", mean_absolute_error(y_test_true, y_test_pred))

# Store the output into a DataFrame
#df_test = df.iloc[val_eindex:, :]
#df_test.loc[:, "del_minutes_pred"] = y_test_pred

