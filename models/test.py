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

batch_size = 64 * 10
n_epochs = 10
n_resnet_units = 3
metrics = ["accuracy"]



test_epoch = 10
model_time_str = "20190118.115432"#datetime.datetime().strftime("%Y%m%d.%H%M%S")


file_dir = "../data/"

useSML = True
smlDateRange = [dt.datetime(1997,1,1), dt.datetime(2007,12,31)]
smlStrtStr = smlDateRange[0].strftime("%Y%m%d")
smlEndStr = smlDateRange[1].strftime("%Y%m%d")
omnTrainParams = ["By", "Bz", "Vx", "Np"]
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

    out_dir="./trained_models/ResNet/" + omnDir + \
            "sml.nBins_" + str(nBins) + "." +\
            "binTimeRes_" + str(binTimeRes) + "." +\
            "onsetFillTimeRes_" + str(onsetFillTimeRes) + "." +\
            "omnHistory_" + str(omnHistory) + "." +\
            "omnDBRes_" + str(omnDBRes) + "." +\
            "useSML_" + str(useSML) + "." +\
            model_time_str

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

    out_dir="./trained_models/ResNet/"  + omnDir +\
            "nBins_" + str(nBins) + "." +\
            "binTimeRes_" + str(binTimeRes) + "." +\
            "onsetFillTimeRes_" + str(onsetFillTimeRes) + "." +\
            "omnHistory_" + str(omnHistory) + "." +\
            "omnDBRes_" + str(omnDBRes) + "." +\
            model_time_str



model_name = glob.glob(os.path.join(out_dir, "weights.epoch_" + str(test_epoch) + "*hdf5"))[0]
test_model = keras.models.load_model(model_name) 


# create out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

output_df_file = out_dir + "/all_data.pred.test"
output_df_test_file = out_dir + "/test_data.pred.test"

# Load the data
print("loading the data...")
X = np.load(input_file)
df = pd.read_csv(csv_file, index_col=0)
y = df.loc[:, "label"].values.reshape(-1, 1)

npoints = X.shape[0]
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
unique_labels = np.unique(y, axis=0)
enc.fit(unique_labels)
y_train_enc = enc.transform(y_train).toarray()
y_test_enc = enc.transform(y_test).toarray()
y_val_enc = enc.transform(y_val).toarray()
y_enc = enc.transform(y).toarray()

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
for i in range(y_pred_enc.shape[1]):
    df.loc[:, "prob_"+str(i)] = y_pred_enc[:, i]
    df_test.loc[:, "prob_"+str(i)] = y_test_pred_enc[:, i]
df.to_csv(output_df_file)
df_test.to_csv(output_df_test_file)



