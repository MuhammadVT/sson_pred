import sys
sys.path.append("../data_pipeline")
import batch_utils
import time
import numpy as np
import pandas as pd
import datetime as dt

omn_dbdir = "../data/sqlite3/"
omn_db_name = "omni_sw_imf.sqlite"
omn_table_name = "imf_sw"
omn_norm_param_file = omn_dbdir + "omn_mean_std.npy"

omnTrainParams = ["By", "Bz", "Vx", "Np"]
imfNormalize = True
omn_train = True
shuffleData = False
polarData=True
imageData=True
omnHistory = 120#120
batch_size = 1
onsetDelTCutoff = 4
onsetFillTimeRes = 5
omnDBRes = 1
binTimeRes = 60#30
nBins = 1
predList=["bin"] 
loadPreComputedOnset = False
saveBinData = False 
onsetSaveFile = "../data/binned_data.feather"

useSML = True
smlDateRange = [ dt.datetime(1997,1,1), dt.datetime(2007,1,1) ]
smlStrtStr = smlDateRange[0].strftime("%Y%m%d")
smlEndStr = smlDateRange[1].strftime("%Y%m%d")

if useSML:
    print("Using SML data")
    input_file = "../data/input." +\
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

    csv_file = "../data/sml_" +\
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
else:  
    input_file = "../data/input." +\
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

    csv_file = "../data/" +\
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

batchObj = batch_utils.DataUtils(omn_dbdir,\
                    omn_db_name, omn_table_name,\
                    omn_train, omn_norm_param_file, useSML=useSML, imfNormalize=imfNormalize, omnDBRes=omnDBRes,\
                    omnTrainParams=omnTrainParams,\
                    batch_size=batch_size, loadPreComputedOnset=loadPreComputedOnset,\
                    onsetDataloadFile="../data/binned_data.feather",\
                    northData=True, southData=False, polarData=polarData,\
                    imageData=imageData, polarFile="../data/polar_data.feather",\
                    imageFile="../data/image_data.feather", onsetDelTCutoff=onsetDelTCutoff,\
                    onsetFillTimeRes=onsetFillTimeRes, binTimeRes=binTimeRes, nBins=nBins,\
                    saveBinData=saveBinData, onsetSaveFile=onsetSaveFile,\
                    shuffleData=shuffleData, omnHistory=omnHistory, smlDateRange=smlDateRange)
x = time.time()
onsetData_list = []
omnData_list = []
dtms = []
for _bat in batchObj.batchDict.keys():
    # get the corresponding input (omnData) and output (onsetData)
    # for this particular batch!
    dtm = batchObj.batchDict[_bat][0]
    onsetData = batchObj.onset_from_batch(batchObj.batchDict[_bat], predList=predList)
    omnData = batchObj.omn_from_batch(batchObj.batchDict[_bat])
    onsetData_list.append(onsetData[0])
    omnData_list.append(omnData)
    dtms.append(dtm)
y = time.time() 
print("inOmn calc--->", y-x)

# Save the output
input_data = np.vstack(omnData_list)
output_data = np.vstack(onsetData_list)
np.save(input_file, input_data)
#np.save(output_file, output_data)

# Save datetimes and output labels
col_dct = {}
lbl = 0
for b in range(nBins):
    col_dct[str(b*binTimeRes) + "_" + str((b+1)*binTimeRes)] = output_data[:, b].astype(int).tolist()
    lbl = lbl + (2**(nBins-1-b)) * output_data[:, b].astype(int)
    
col_dct["label"] = lbl
df = pd.DataFrame(data=col_dct, index=dtms)
df.to_csv(csv_file)

