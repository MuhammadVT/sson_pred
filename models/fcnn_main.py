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

classes_are_mutually_exclusive = False    # Corresponds to sigmoid activation layer
#classes_are_mutually_exclusive = True    # Corresponds to softmax activation layer

# Load the data
print("loading the data...")
#input_file = "../data/input.omnHistory_120.onsetDelTCutoff_2.omnDBRes_1.imfNormalize_True.shuffleData_True.npy" 
#output_file = "../data/output.nBins_6.binTimeRes_10.onsetFillTimeRes_1.shuffleData_True.npy"

input_file = "../data/input.omnHistory_120.onsetDelTCutoff_2.omnDBRes_1.imfNormalize_True.shuffleData_True.npy" 
output_file = "../data/output.nBins_2.binTimeRes_30.onsetFillTimeRes_1.shuffleData_True.npy"

#input_file = "../data/input.omnHistory_120.onsetDelTCutoff_2.omnDBRes_1.imfNormalize_True.shuffleData_True.npy" 
#output_file = "../data/output.nBins_1.binTimeRes_30.onsetFillTimeRes_1.shuffleData_True.npy"

#input_file = "../data/input.omnHistory_120.onsetDelTCutoff_2.omnDBRes_1.imfNormalize_True.shuffleData_False.npy" 
#output_file = "../data/output.nBins_1.binTimeRes_30.onsetFillTimeRes_1.shuffleData_False.npy"

#input_file = "../data/input.omnHistory_120.onsetDelTCutoff_2.omnDBRes_1.imfNormalize_False.shuffleData_False.npy" 
#output_file = "../data/output.nBins_1.binTimeRes_30.onsetFillTimeRes_1.shuffleData_False.npy"

X = np.load(input_file)
y = np.load(output_file)
#y = y[:, 1:]
if classes_are_mutually_exclusive:
    #y = y[:, 0:2]    # two classes
    #y = y[:, 0]    # one class
    enc = OneHotEncoder()
    y = enc.fit_transform(y.reshape(-1,1)).toarray()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=10)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=10)

# Build a FCNN model
optimizer=keras.optimizers.Adam(lr=0.0001)
batch_size = 64
n_epochs = 500
n_classes = y_train.shape[1] 
metrics = ["accuracy"]
input_shape = x_train.shape[1:]

# Set class weight
def binary_crossentropy_weigted(y_true, y_pred):
    import keras.backend as K
    class_weights = K.variable(np.array([5]))
    #class_weights = np.array([5, 1])
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    #loss = K.mean(class_weights*(-y_true * K.log(y_pred) - (1.0 - y_true) * K.log(1.0 - y_pred)),axis=-1)
    loss = K.mean((-y_true * K.log(y_pred) * class_weights - (1.0 - y_true) * K.log(1.0 - y_pred)),axis=-1)
    return loss

#from sklearn import utils
#class_weights = utils.class_weight.compute_class_weight("balanced", np.unique(y_train) 
#class_weights = {i:(1./y_train[:,i].mean()) for i in range(n_classes)}
#class_weights = {0:10, 1:1, 2:1, 3:1, 4:1, 5:1}
class_weights = None

# create out_dir
out_dir="./trained_models/FCNN/" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# Define the loss
if classes_are_mutually_exclusive:
    loss=keras.losses.categorical_crossentropy
else:
    loss=keras.losses.binary_crossentropy
    #loss = binary_crossentropy_weigted


fcnn = FCNN(input_shape, batch_size=batch_size, n_epochs=n_epochs,
            n_classes=n_classes, loss=loss, optimizer=optimizer,
            metrics=metrics, out_dir=out_dir)

# Train the model
print("Training the model...")
fit_history = fcnn.train_model(x_train, y_train, x_val, y_val,
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
#fname = "weights.epoch_{epoch}.val_loss_{val_loss}.val_acc_{val_acc}.hdf5"
test_epoch = n_epochs
model_name = glob.glob(os.path.join(out_dir, "weights.epoch_" + str(test_epoch) + "*hdf5"))[0]
test_model = keras.models.load_model(model_name, custom_objects={'binary_crossentropy_weigted': binary_crossentropy_weigted}) 
y_pred = test_model.predict(x_test, batch_size=32)

if classes_are_mutually_exclusive:
    # The final activation layer uses softmax
    y_pred = np.argmax(y_pred , axis=1)
    y_true = np.argmax(y_test , axis=1)
    print(classification_report(y_true, y_pred))
else:
    # The final activation layer uses sigmoid
    y_pred = (y_pred > 0.5).astype(int)
    y_true = y_test
    for col in range(y_pred.shape[1]):
        #print(classification_report(y_true[:, col], y_pred[:, col], target_names=["bin_"+str(col)]))
        print(classification_report(y_true[:, col], y_pred[:, col]))


