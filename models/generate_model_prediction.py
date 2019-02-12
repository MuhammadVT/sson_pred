import sys
sys.path.append("../data_pipeline")
import omn_utils
import time
import numpy as np
import pandas as pd
import datetime as dt
import glob
import os

def load_omn_data(omnStartDate, omnEndDate, omnDBDir,
	       	  omnDbName, omnTabName,
	       	  omnTrain, omnNormParamFile,
	       	  imf_normalize=True,
	       	  db_time_resolution=1,
	       	  omn_train_params=["By", "Bz", "Bx", "Vx", "Np"]):
    # create the obj and load data
    omnObj = omn_utils.OmnData(omnStartDate, omnEndDate, omnDBDir,
			       omnDbName, omnTabName,
			       omnTrain, omnNormParamFile,
			       imf_normalize=imfNormalize,
			       db_time_resolution=omnDBRes,
			       omn_train_params = omnTrainParams)
    # Set the datetime as index since we are working off of it
    omnObj.omnDF = omnObj.omnDF.set_index(omnObj.omnDF["datetime"])
    omnObj.omnDF = omnObj.omnDF[omnTrainParams]
    return omnObj.omnDF

def create_datapoints(df_omn, dateList, omnHistory=120):
    """
    Given a list of dates from one batch
    get omn data hist for each data point.
    """
    # Note our dateList could be shuffled
    # so we can't simply use a range for
    # accesing data from the index!
    omnBatchMatrix = []
    for _cd in dateList:
        _st = _cd.strftime("%Y-%m-%d %H:%M:%S")
        _et = (_cd - dt.timedelta(\
                minutes=omnHistory) ).strftime(\
                "%Y-%m-%d %H:%M:%S")
        omnBatchMatrix.append(df_omn.loc[ _et : _st ].values)
    return np.array(omnBatchMatrix)

def predict_onset(model_fname, input_datapoints, input_dtms,
		  batch_size=32, nBins=1, binTimeRes=30): 

    import keras

    # Load the model weights
    test_model = keras.models.load_model(model_fname)

    # Set the input data
    X = input_datapoints

    # Make substorm onset prediction
    print("Making substorm onset prediction...")
    y_pred_enc = test_model.predict(X, batch_size=batch_size)
    y_pred = np.argmax(y_pred_enc , axis=1)

    # Save the predicted outputs to a DF
    df = pd.DataFrame(data={"current_datatime":input_dtms})
    df.loc[:, "pred_label"] = y_pred

    return df

if __name__ == "__main__":

    # get the time range from onset data
    omnHistory = 120
    omnStartDate = dt.datetime(1996, 3, 30, 0, 0) - dt.timedelta(minutes=omnHistory)
    omnEndDate = dt.datetime(1996, 5, 1, 0, 0)
    #omnEndDate = dt.datetime(2009, 1, 1, 0, 0)
    datapoint_sdtm = omnStartDate + dt.timedelta(minutes=omnHistory)
    datapoint_edtm = omnEndDate

    omnDBDir = "../data/sqlite3/"
    omnDbName = "omni_sw_imf.sqlite"
    omnTabName = "imf_sw"
    omnTrain = False
    omnNormParamFile = omnDBDir + "omn_mean_std.npy"
    omnTrainParams = [ "By", "Bz", "Bx", "Vx", "Np" ]
    nBins = 1
    binTimeRes = 30
    imfNormalize = True
    onsetFillTimeRes = 1
    omnDBRes = 1
    batch_size = 32

    # Select the model to be tested
    test_epoch = 200
    #out_dir="./trained_models/ResNet/20190104_113412/"    # good one for 1-bin prediction
    #out_dir="./trained_models/ResNet/20190104_155006/"     # good one for 2-bin prediction
    out_dir="./trained_models/ResNet/" +\
            "nBins_" + str(nBins) + "." +\
            "binTimeRes_" + str(binTimeRes) + "." +\
            "onsetFillTimeRes_" + str(onsetFillTimeRes) + "." +\
            "omnHistory_" + str(omnHistory) + "." +\
            "omnDBRes_" + str(omnDBRes) + "." +\
            "20190107.150833"
    model_fname = glob.glob(os.path.join(out_dir, "weights.epoch_" + str(test_epoch) + "*hdf5"))[0]

    # Set file names for datapoints (input) and onset predictions (output)
    file_dir = "../data/"
#    input_fname = "input."+\
#                  datapoint_sdtm.strftime("%Y%m%d.%H%M_") + datapoint_edtm.strftime("%Y%m%d.%H%M")
#                  "nBins_2.binTimeRes_30.omnHistory_120.omnDBRes_1.imfNormalize_True.npy"
    output_fname = "predicted_onset."+\
                   datapoint_sdtm.strftime("%Y%m%d.%H%M_") + datapoint_edtm.strftime("%Y%m%d.%H%M") +\
		   "nBins_" + str(nBins) + "." +\
		   "binTimeRes_" + str(binTimeRes) + "." +\
		   "onsetFillTimeRes_" + str(onsetFillTimeRes) + "." +\
		   "omnHistory_" + str(omnHistory) + "." +\
		   "omnDBRes_" + str(omnDBRes) + "." +\
		   "imfNormalize_" + str(imfNormalize) + "." +\
		   "csv"
#    input_fname = file_dir + input_fname
    output_fname = file_dir + output_fname

    # Load OMNI data
    df_omn = load_omn_data(omnStartDate, omnEndDate, omnDBDir,
			   omnDbName, omnTabName,
			   omnTrain, omnNormParamFile,
			   imf_normalize=imfNormalize,
			   db_time_resolution=omnDBRes,
			   omn_train_params = omnTrainParams)

    # Create data points
    #time_step = nBins * binTimeRes
    time_step = 120 
    input_dtms = pd.date_range(start=datapoint_sdtm, end=datapoint_edtm,
                               freq=str(time_step)+"min").tolist()

    #df_test = pd.read_csv("../data/test_data.nBins_1.binTimeRes_30.onsetFillTimeRes_1.onsetDelTCutoff_2.omnHistory_120.omnDBRes_1.shuffleData_True.csv", index_col=0)
    #input_dtms = [x for x in pd.to_datetime(df_test.index.sort_values()) if ((x >=datapoint_sdtm) and (x <=datapoint_edtm))]

    input_datapoints = create_datapoints(df_omn, input_dtms, omnHistory=omnHistory)

    # Make onset predictions
    df_pred = predict_onset(model_fname, input_datapoints, input_dtms,
		            batch_size=batch_size, nBins=nBins, binTimeRes=binTimeRes)
    df_pred.set_index("current_datatime", inplace=True)
    #df_pred.to_csv(output_fname)


