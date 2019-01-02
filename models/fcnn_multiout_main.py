import keras
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from dnn_classifiers import FCNN_MultiOut
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

#skip_training = True
skip_training = False

# Load the data
print("loading the data...")

#input_file = "../data/input.omnHistory_120.onsetDelTCutoff_2.omnDBRes_1.imfNormalize_True.shuffleData_True.npy" 
#output_file = "../data/output.nBins_3.binTimeRes_20.onsetFillTimeRes_1.shuffleData_True.npy"

input_file = "../data/input.omnHistory_120.onsetDelTCutoff_2.omnDBRes_1.imfNormalize_True.shuffleData_True.npy" 
output_file = "../data/output.nBins_2.binTimeRes_30.onsetFillTimeRes_1.shuffleData_True.npy"

#input_file = "../data/input.omnHistory_120.onsetDelTCutoff_2.omnDBRes_1.imfNormalize_True.shuffleData_True.npy" 
#output_file = "../data/output.nBins_6.binTimeRes_10.onsetFillTimeRes_1.shuffleData_True.npy"

#input_file = "../data/input.omnHistory_120.onsetDelTCutoff_2.omnDBRes_1.imfNormalize_True.shuffleData_True.npy" 
#output_file = "../data/output.nBins_1.binTimeRes_30.onsetFillTimeRes_1.shuffleData_True.npy"

#input_file = "../data/input.omnHistory_120.onsetDelTCutoff_2.omnDBRes_1.imfNormalize_True.shuffleData_False.npy" 
#output_file = "../data/output.nBins_1.binTimeRes_30.onsetFillTimeRes_1.shuffleData_False.npy"

#input_file = "../data/input.omnHistory_120.onsetDelTCutoff_2.omnDBRes_1.imfNormalize_False.shuffleData_False.npy" 
#output_file = "../data/output.nBins_1.binTimeRes_30.onsetFillTimeRes_1.shuffleData_False.npy"

X = np.load(input_file)
y = np.load(output_file)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=10)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=10)

# Build a FCNN_MultiOut model
optimizer=keras.optimizers.Adam(lr=0.0001)
batch_size = 64
n_epochs = 500
n_classes = y_train.shape[1] 
metrics = ["accuracy"]
input_shape = x_train.shape[1:]

# Encode the labels for each output bin
y_train_list = []
y_test_list = []
y_val_list = []
for i in range(y.shape[1]): 
    enc = OneHotEncoder()
    y_train_list.append(enc.fit_transform(y_train[:, i].reshape(-1,1)).toarray())
    y_test_list.append(enc.fit_transform(y_test[:, i].reshape(-1,1)).toarray())
    y_val_list.append(enc.fit_transform(y_val[:, i].reshape(-1,1)).toarray())

# create out_dir
out_dir="./trained_models/FCNN_MultiOut/" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
#out_dir = "trained_models/FCNN_MultiOut/20181228_192830/"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# Define the loss, loss_weights, and class_weights
loss=keras.losses.categorical_crossentropy
#loss_weights = [1. for x in range(n_classes)]
loss_weights = [1., 1.2]
#loss_weights = [1., 1.2, 1.4]

#from sklearn import utils
class_weights = [{0:0.1, 1:0.1*(y_train_list[i].shape[0]-y_train_list[i][:,1].sum())/(y_train_list[i][:,1].sum())} for i in range(n_classes)]
#class_weights = [{0:1, 1:1*(y_train_list[i].shape[0]-y_train_list[i][:,1].sum())/(y_train_list[i][:,1].sum())} for i in range(n_classes)]
#class_weights = [{0:1, 1:10} for i in range(n_classes)]
#class_weights = None

fcnn = FCNN_MultiOut(input_shape, batch_size=batch_size, n_epochs=n_epochs,
            n_classes=n_classes, loss=loss, loss_weights=loss_weights,
            optimizer=optimizer, metrics=metrics, out_dir=out_dir)

# Train the model
if not skip_training:
    print("Training the model...")
    fit_history = fcnn.train_model(x_train, y_train_list, x_val, y_val_list,
                                   class_weights=class_weights)

## Plot the training 
#fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
#xs = np.arange(n_epochs)
#train_loss = fit_history.history["loss"]
#val_loss = fit_history.history["val_loss"]
#train_acc = fit_history.history["acc"]
#val_acc = fit_history.history["val_acc"]
#axes[0].plot(xs, train_loss, label="train_loss") 
#axes[0].plot(xs, val_loss, label="val_loss") 
#axes[1].plot(xs, train_acc, label="train_acc") 
#axes[1].plot(xs, val_acc, label="val_acc") 
#axes[0].set_title("Training Loss and Accuracy")
#axes[0].set_ylabel("Loss")
#axes[1].set_ylabel("Accuracy")
#axes[1].set_xlabel("Epoch")
#axes[0].legend()
#axes[1].legend()
#fig_path = os.path.join(out_dir, "loss_acc")
#fig.savefig(fig_path + ".png", dpi=200, bbox_inches="tight")  

# Evaluate the model on test dataset
print("Evaluating the model...")
#test_epoch = 460
test_epoch = n_epochs
model_name = glob.glob(os.path.join(out_dir, "weights.epoch_" + str(test_epoch) + "*hdf5"))[0]
test_model = keras.models.load_model(model_name) 
y_pred_list = test_model.predict(x_test, batch_size=32)

# The final activation layer uses softmax
y_pred_list = [np.argmax(y_pred , axis=1) for y_pred in y_pred_list]
y_true_list = [np.argmax(y_true , axis=1) for y_true in y_test_list]
for i in range(len(y_pred_list)):
    print(classification_report(y_true_list[i], y_pred_list[i]))


