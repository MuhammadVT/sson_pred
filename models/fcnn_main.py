import keras
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report
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

# Read the data

# Build a FCNN model
loss=keras.losses.categorical_crossentropy
optimizer=keras.optimizers.Adam()
batch_size = 32
n_epochs = 10
n_classes = len(np.unique(np.vstack((y_train,y_test)),axis=0))
out_dir="./trained_models/FCNN/" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
# create out_dir
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
fcnn = FCNN(input_shape, batch_size=batch_size, n_epochs=n_epochs,
            n_classes=n_classes, loss=loss, optimizer=optimizer,
            metrics=["accuracy"], out_dir=out_dir)

# Train the model
fit_history = fcnn.train_model(x_train, y_train, x_val, y_val, y_true)

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
#fname = "weights.epoch_{epoch}.val_loss_{val_loss}.val_acc_{val_acc}.hdf5"
model_name = glob.glob(os.path.join(out_dir, "weights.epoch_08*hdf5"))[0]
test_model = keras.models.load_model(model_name) 
y_pred = test_model.predict(x_test, batch_size=32)
y_pred = np.argmax(y_pred , axis=1)
print(classification_report(y_true, y_pred))

