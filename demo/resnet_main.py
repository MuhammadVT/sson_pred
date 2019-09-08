import keras
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from dnn_classifiers import ResNet, train_model
from scipy.io import loadmat
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
import numpy as np
import pandas as pd
import datetime as dt
import os
import glob
import time

skip_training = False
#skip_training = True

transfer_weights = False
transfered_model_epoch = 50
if transfer_weights:
    skip_training = True
    weight_dir = "./trained_models/ResNet/omn_Bx_By_Bz_Vx_Np/" +\
                 "sml.nBins_1.binTimeRes_30.onsetFillTimeRes_5.omnHistory_120.omnDBRes_1.useSML_True.20190122.154340/"

save_pred = True
model_txt = "resnet_"

nBins = 1
binTimeRes = 60
imfNormalize = True
shuffleData = False 
polarData = True
imageData = True
omnHistory = 120
omnHistory_actual = 120
onsetDelTCutoff = 4
onsetFillTimeRes = 30
omnDBRes = 1

batch_size = 16 * 4 * 1
n_epochs = 10
n_resnet_units = 2
metrics = ["accuracy"]

#txt = "deltm."
#txt = "iso."
txt = "interp_15.delay_10."

useSML = True 
smlDateRange = [dt.datetime(1997,1,1), dt.datetime(2018,1,1)]
#smlDateRange = [dt.datetime(1997,1,1), dt.datetime(2007,12,31)]    # Downsampled
#smlDateRange = [dt.datetime(1997,1,1), dt.datetime(2018,1,1)]       # Downsampled
#smlDateRange = [dt.datetime(1997,1,1), dt.datetime(2018,1,3)]       # Downsampled by UT
#smlDateRange = [dt.datetime(1997,1,1), dt.datetime(2017,12,30)]       # Includes SML, SMU. Downsampled by UT
#smlDateRange = [dt.datetime(1997,1,1), dt.datetime(2017,12,29)]       # isolated substorms (sep=120min). Includes SML, SMU. Downsampled by UT

#smlDateRange = [dt.datetime(1997,1,1), dt.datetime(2004,1,1)]
#smlDateRange = [dt.datetime(1997,1,1), dt.datetime(2008,1,1)]

smlStrtStr = smlDateRange[0].strftime("%Y%m%d")
smlEndStr = smlDateRange[1].strftime("%Y%m%d")
omnTrainParams = ["Bx", "By", "Bz", "Vx", "Np"]

omnTrainParams_actual = ["Bx", "By", "Bz", "Vx", "Np"]    # This is the one that goes into the actual training
#omnTrainParams_actual = ["Bx", "By", "Bz", "Vx", "Np", "au", "al"]    # This is the one that goes into the actual training
#omnTrainParams_actual = ["By", "Bz", "Vx", "Np", "al"]    # This is the one that goes into the actual training
#omnTrainParams_actual = ["au", "al"]    # This is the one that goes into the actual training
param_col_dict = {"Bx":0, "By":1, "Bz":2, "Vx":3, "Np":4, "au":5, "al":6}
input_cols = [param_col_dict[x] for x in omnTrainParams_actual]

# since we have different omnTrainParams for different datasets
# we'll create seperate folders for them for simplicity
omnDir = "omn_"
for _nom, _npm in enumerate(omnTrainParams):
    omnDir += _npm
    if _nom < len(omnTrainParams)-1:
        omnDir += "_"
    else:
        omnDir += "/"

omnDir_actual = "omn_"
for _nom, _npm in enumerate(omnTrainParams_actual):
    omnDir_actual += _npm
    if _nom < len(omnTrainParams_actual)-1:
        omnDir_actual += "_"
    else:
        omnDir_actual += "/"

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
                 txt +\
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
               txt +\
               "csv"

    out_dir="../models/trained_models/ResNet/" + omnDir_actual + \
            "sml.nBins_" + str(nBins) + "." +\
            "binTimeRes_" + str(binTimeRes) + "." +\
            "onsetFillTimeRes_" + str(onsetFillTimeRes) + "." +\
            "omnHistory_" + str(omnHistory) + "." +\
            "omnDBRes_" + str(omnDBRes) + "." +\
            "useSML_" + str(useSML) + "." +\
            dt.datetime.now().strftime("%Y%m%d.%H%M%S")

else:  
    input_file = "../data/" + omnDir + "input." +\
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

    out_dir="../models/trained_models/ResNet/"  + omnDir_actual +\
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
df = pd.read_csv(csv_file, index_col=0, parse_dates={"datetime":[0]})
y = df.loc[:, "label"].values.reshape(-1, 1)

# Select certain columns
X = X[:, :, input_cols]

########################
## Limit the UT time range
#ut_range = [5, 12]
#idx_ut = np.where((df.index.hour.values>= 6) & (df.index.hour.values<=12))[0]
#df = df.iloc[idx_ut, :]
#X = X[idx_ut, :, :]
#y = y[idx_ut, :]
########################

########################
## Add UT time features
#dlm = np.array([np.timedelta64(i-omnHistory, "m") for i in range(omnHistory+1)])
#dlms = np.tile(dlm, (X.shape[0], 1))
#dtms = np.tile(df.index.values.reshape((-1, 1)), (1, omnHistory+1))
#dtms = dtms + dlms
#minutes = (dtms - dtms.astype("datetime64[D]")).astype("timedelta64[m]").astype(int)
#minutes = minutes.reshape((minutes.shape[0], omnHistory+1, 1))
## Encode UT minutes using sine and cosine functions
#minutes_sine = np.sin(2*np.pi/(60*24) * minutes)
#minutes_cosine = np.cos(2*np.pi/(60*24) * minutes)
#X = np.concatenate([X, minutes_sine, minutes_cosine], axis=2)
#
## Add Month features
#months = (dtms.astype("datetime64[M]") - dtms.astype("datetime64[Y]")).astype("timedelta64[M]").astype(int) + 1
#months = months.reshape((months.shape[0], omnHistory+1, 1))
## Encode months using sine and cosine functions
#months_sine = np.sin(2*np.pi/12 * months)
#months_cosine = np.cos(2*np.pi/12 * months)
#X = np.concatenate([X, months_sine, months_cosine], axis=2)
#######################

#########################
## Add del_minutes feature
#deltm = df.del_minutes.values
#deltms = np.tile(deltm.reshape(-1, 1, 1), (1, X.shape[1], 1))
#X = np.concatenate([X, deltms], axis=2)
#########################

########################
## Limit the time history
X = X[:, -omnHistory_actual-1:, :]
#import pdb
#pdb.set_trace()
#X = X[:, 120:, :]
########################

## Do x-min average to the input data
#x_min_avg = 30
#X_mean = np.mean(X[:, 1:, :].reshape(X.shape[0], int((X.shape[1]-1)/x_min_avg), x_min_avg, X.shape[-1]), axis=2)
#X_std = np.std(X[:, 1:, :].reshape(X.shape[0], int((X.shape[1]-1)/x_min_avg), x_min_avg, X.shape[-1]), axis=2)
#X_min = np.min(X[:, 1:, :].reshape(X.shape[0], int((X.shape[1]-1)/x_min_avg), x_min_avg, X.shape[-1]), axis=2)
#X_max = np.max(X[:, 1:, :].reshape(X.shape[0], int((X.shape[1]-1)/x_min_avg), x_min_avg, X.shape[-1]), axis=2)
#X = np.concatenate([X_mean, X_std, X_min, X_max], axis=2)

#### Skip every dtm_step
#dtm_step = 1
#X = X[::dtm_step, :, :]
#y = y[::dtm_step, :]

npoints = X.shape[0]
n_classes = np.unique(y).shape[0]

train_size = 0.70
val_size = 0.15
test_size = 0.15
train_eindex = int(npoints * train_size)
val_eindex = train_eindex + int(npoints * val_size)
x_train = X[:train_eindex, :]
x_val = X[train_eindex:val_eindex, :]
x_test = X[val_eindex:, :]
y_train = y[:train_eindex, :]
y_val = y[train_eindex:val_eindex, :]
y_test = y[val_eindex:, :]

df_train = df.iloc[:train_eindex, :]
df_val = df.iloc[train_eindex:val_eindex, :]
df_test = df.iloc[val_eindex:, :]

##################################
# Balance the two classes in train data
ss_idx = np.where(df_train.label.values == 1)[0]
nonss_idx = np.where(df_train.label.values == 0)[0]
np.random.seed(1)
nonss_idx = np.random.choice(nonss_idx, len(ss_idx), replace=False)
event_idx = np.concatenate([ss_idx, nonss_idx])
#np.random.shuffle(event_idx)
# Keep the order of data points the same as before balancing
event_idx.sort()
df_train = df_train.iloc[event_idx, :]
# Select for certain rows and columns
x_train = x_train[event_idx]
y_train = y_train[event_idx]

# Balance the two classes in val data
ss_idx = np.where(df_val.label.values == 1)[0]
nonss_idx = np.where(df_val.label.values == 0)[0]
np.random.seed(1)
nonss_idx = np.random.choice(nonss_idx, len(ss_idx), replace=False)
event_idx = np.concatenate([ss_idx, nonss_idx])
#np.random.shuffle(event_idx)
# Keep the order of data points the same as before balancing
event_idx.sort()
df_val = df_val.iloc[event_idx, :]
# Select for certain rows and columns
x_val = x_val[event_idx]
y_val = y_val[event_idx]

# Balance the two classes in test data
ss_idx = np.where(df_test.label.values == 1)[0]
nonss_idx = np.where(df_test.label.values == 0)[0]
np.random.seed(1)
nonss_idx = np.random.choice(nonss_idx, len(ss_idx), replace=False)
event_idx = np.concatenate([ss_idx, nonss_idx])
#np.random.shuffle(event_idx)
# Keep the order of data points the same as before balancing
event_idx.sort()
df_test = df_test.iloc[event_idx, :]
# Select for certain rows and columns
x_test = x_test[event_idx]
y_test = y_test[event_idx]

df = pd.concat([df_train, df_val, df_test])
X = np.concatenate([x_train, x_val, x_test])
y = np.concatenate([y_train, y_val, y_test])
##################################

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

# Build a ResNet model
optimizer=keras.optimizers.Adam(lr=0.00001)
input_shape = X.shape[1:]

# Define the loss, loss_weights, and class_weights
loss=keras.losses.categorical_crossentropy

#class_weights = {0:0.1, 1:0.9}
class_weights = None

# Train the model
if not skip_training:
    dl_obj = ResNet(input_shape, batch_size=batch_size, n_epochs=n_epochs,
                    n_classes=n_classes, n_resnet_units=n_resnet_units, loss=loss,
                    optimizer=optimizer,
                    metrics=metrics, out_dir=out_dir)

    print("Training the model...")
    dl_obj.model.summary()
    fit_history = train_model(dl_obj.model, x_train, y_train_enc, x_val, y_val_enc,
                              batch_size=batch_size, n_epochs=n_epochs,
                              callbacks=dl_obj.callbacks, shuffle=True,
                              class_weights=class_weights)

if transfer_weights:
    # Load the weight of a pre-trained model
    print("Loading the weights of a pre-trained model...")
    model_name = glob.glob(os.path.join(weight_dir, "weights.epoch_" + str(transfered_model_epoch) + "*hdf5"))[0]
    loaded_model = keras.models.load_model(model_name)

    # Save the model at certain checkpoints
    fname = "weights.epoch_{epoch:02d}.val_loss_{val_loss:.2f}.val_acc_{val_acc:.2f}.hdf5"
    file_path = os.path.join(out_dir, fname)
    model_checkpoint = ModelCheckpoint(file_path, monitor='val_acc', save_best_only=False, period=1)
    callbacks = [model_checkpoint]
    fit_history = train_model(loaded_model, x_train, y_train_enc, x_val, y_val_enc,
                              batch_size=batch_size, n_epochs=n_epochs,
                              callbacks=callbacks, shuffle=True,
                              class_weights=class_weights)

# Plot the loss curve and the prediction accuracy
if transfer_weights or not skip_training:
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
#test_epoch = 50    # The epoch number of the model we want to evaluate
if test_epoch < 10:
    model_name = glob.glob(os.path.join(out_dir, "weights.epoch_0" + str(test_epoch) + "*hdf5"))[0]
else:
    model_name = glob.glob(os.path.join(out_dir, "weights.epoch_" + str(test_epoch) + "*hdf5"))[0]
test_model = keras.models.load_model(model_name) 

# Make predictions
y_train_pred_enc = test_model.predict(x_train, batch_size=batch_size)
y_val_pred_enc = test_model.predict(x_val, batch_size=batch_size)
y_test_pred_enc = test_model.predict(x_test, batch_size=batch_size)
y_pred_enc = test_model.predict(X, batch_size=batch_size)

# The final activation layer uses softmax
y_train_pred = np.argmax(y_train_pred_enc , axis=1)
y_val_pred = np.argmax(y_val_pred_enc , axis=1)
y_test_pred = np.argmax(y_test_pred_enc , axis=1)
y_pred = np.argmax(y_pred_enc , axis=1)
y_train_true = y_train
y_val_true = y_val
y_test_true = y_test

# Report for train data
print("Prediction report for train input data:")
print(classification_report(y_train_true, y_train_pred))

# Report for validation data
print("Prediction report for validation input data:")
print(classification_report(y_val_true, y_val_pred))

# Report for test data
print("Prediction report for test data:")
print(classification_report(y_test_true, y_test_pred))

if save_pred:
    # Save the predicted outputs
    #out_dir = "./trained_models/MLP_iso"
    output_df_file = out_dir + "/" + model_txt + "all_data_pred.csv"
    output_df_train_file = out_dir + "/" + model_txt + "train_data_pred.csv"
    output_df_val_file = out_dir + "/" + model_txt + "val_data_pred.csv"
    output_df_test_file = out_dir + "/" + model_txt + "test_data_pred.csv"

    df.loc[:, "pred_label"] = y_pred
#    df_train = df.iloc[:train_eindex, :]
#    df_val = df.iloc[train_eindex:val_eindex, :]
#    df_test = df.iloc[val_eindex:, :]
    df_train.loc[:, "pred_label"] = y_train_pred
    df_val.loc[:, "pred_label"] = y_val_pred
    df_test.loc[:, "pred_label"] = y_test_pred

    for i in range(y_pred_enc.shape[1]):
        df.loc[:, "prob_"+str(i)] = y_pred_enc[:, i]
        df_train.loc[:, "prob_"+str(i)] = y_train_pred_enc[:, i]
        df_val.loc[:, "prob_"+str(i)] = y_val_pred_enc[:, i]
        df_test.loc[:, "prob_"+str(i)] = y_test_pred_enc[:, i]
    
    df.to_csv(output_df_file)
    df_train.to_csv(output_df_train_file)
    df_val.to_csv(output_df_val_file)
    df_test.to_csv(output_df_test_file)


