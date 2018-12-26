import keras
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from dnn_classifiers import FCNN
from scipy.io import loadmat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import numpy as np
import datetime as dt
import os
import glob
import time
import sys
sys.path.append("../data_pipeline")
#import batch_utils

# Read the data
###########################
#omn_dbdir = "../data/sqlite3/"
#omn_db_name = "omni_sw_imf.sqlite"
#omn_table_name = "imf_sw"
#omn_train = True
#omn_norm_param_file = omn_dbdir + "omn_mean_std.npy"
#batchObj = batch_utils.DataUtils(omn_dbdir,\
#                    omn_db_name, omn_table_name,\
#                    omn_train, omn_norm_param_file, imfNormalize=True, omnDBRes=1,\
#                    omnTrainParams = [ "By", "Bz", "Bx", "Vx", "Np" ],\
#                    batch_size=64, loadPreComputedOnset=True,\
#                    onsetDataloadFile="../data/binned_data.feather",\
#                    northData=True, southData=False, polarData=True,\
#                    imageData=True, polarFile="../data/polar_data.feather",\
#                    imageFile="../data/image_data.feather", onsetDelTCutoff=2,\
#                    onsetFillTimeRes=1, binTimeRes=30, nBins=3,\
#                    saveBinData=True, onsetSaveFile="../data/binned_data.feather",\
#                    shuffleData=True, omnHistory=120)
#x = time.time()
#onsetData_list = []
#omnData_list = []
#for _bat in batchObj.batchDict.keys():
#    # get the corresponding input (omnData) and output (onsetData)
#    # for this particular batch!
#    onsetData = batchObj.onset_from_batch(batchObj.batchDict[_bat])
#    omnData = batchObj.omn_from_batch(batchObj.batchDict[_bat])
#    onsetData_list.append(onsetData)
#    omnData_list.append(omnData)
#y = time.time()
#print("inOmn calc--->", y-x)
#import pdb
#pdb.set_trace()
###########################
print("loading the data...")
X = np.load("../data/omni.npy")
y = np.load("../data/output.npy")
#y = y[:, 0:2]    # two classes
y = y[:, 0]    # one class
enc = OneHotEncoder()
y = enc.fit_transform(y.reshape(-1,1)).toarray()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=10)

# Build a FCNN model
loss=keras.losses.categorical_crossentropy
optimizer=keras.optimizers.Adam()
batch_size = 32
n_epochs = 10
n_classes = y_train.shape[1] 
input_shape = x_train.shape[1:]
out_dir="./trained_models/FCNN/" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
# create out_dir
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
fcnn = FCNN(input_shape, batch_size=batch_size, n_epochs=n_epochs,
            n_classes=n_classes, loss=loss, optimizer=optimizer,
            metrics=["accuracy"], out_dir=out_dir)

# Train the model
print("Training the model...")
fit_history = fcnn.train_model(x_train, y_train, x_val, y_val, y_test)

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
#fname = "weights.epoch_{epoch}.val_loss_{val_loss}.val_acc_{val_acc}.hdf5"
model_name = glob.glob(os.path.join(out_dir, "weights.epoch_08*hdf5"))[0]
test_model = keras.models.load_model(model_name) 
y_pred = test_model.predict(x_test, batch_size=32)
y_pred = np.argmax(y_pred , axis=1)
y_true = np.argmax(y_test , axis=1)
print(classification_report(y_true, y_pred))

