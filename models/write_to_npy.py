import sys
sys.path.append("../data_pipeline")
import batch_utils
import time
import numpy as np

omn_dbdir = "../data/sqlite3/"
omn_db_name = "omni_sw_imf.sqlite"
omn_table_name = "imf_sw"
omn_norm_param_file = omn_dbdir + "omn_mean_std.npy"

imfNormalize = True
omn_train = True
shuffleData = True 
omnHistory = 120
batch_size = 1
onsetDelTCutoff = 2
onsetFillTimeRes = 1
omnDBRes = 1
binTimeRes = 30
nBins = 2
loadPreComputedOnset = False
saveBinData = True
onsetSaveFile = "../data/binned_data.feather"
input_file = "../data/input." +\
	     "omnHistory_" + str(omnHistory) + "." +\
	     "onsetDelTCutoff_" + str(onsetDelTCutoff) + "." +\
	     "omnDBRes_" + str(omnDBRes) + "." +\
	     "imfNormalize_" + str(imfNormalize) + "." +\
	     "shuffleData_" + str(shuffleData) + "." +\
	     "npy"
output_file = "../data/output." +\
	      "nBins_" + str(nBins) + "." +\
	      "binTimeRes_" + str(binTimeRes) + "." +\
	      "onsetFillTimeRes_" + str(onsetFillTimeRes) + "." +\
	      "shuffleData_" + str(shuffleData) + "." +\
	      "npy"

batchObj = batch_utils.DataUtils(omn_dbdir,\
                    omn_db_name, omn_table_name,\
                    omn_train, omn_norm_param_file, imfNormalize=imfNormalize, omnDBRes=omnDBRes,\
                    omnTrainParams = [ "By", "Bz", "Bx", "Vx", "Np" ],\
                    batch_size=batch_size, loadPreComputedOnset=loadPreComputedOnset,\
                    onsetDataloadFile="../data/binned_data.feather",\
                    northData=True, southData=False, polarData=True,\
                    imageData=True, polarFile="../data/polar_data.feather",\
                    imageFile="../data/image_data.feather", onsetDelTCutoff=onsetDelTCutoff,\
                    onsetFillTimeRes=onsetFillTimeRes, binTimeRes=binTimeRes, nBins=nBins,\
                    saveBinData=saveBinData, onsetSaveFile=onsetSaveFile,\
                    shuffleData=shuffleData, omnHistory=omnHistory)
x = time.time()
onsetData_list = []
omnData_list = []
for _bat in batchObj.batchDict.keys():
    # get the corresponding input (omnData) and output (onsetData)
    # for this particular batch!
    onsetData = batchObj.onset_from_batch(batchObj.batchDict[_bat])
    omnData = batchObj.omn_from_batch(batchObj.batchDict[_bat])
    onsetData_list.append(onsetData[0])
    omnData_list.append(omnData)
y = time.time() 
print("inOmn calc--->", y-x)

# Save the output
input_data = np.vstack(omnData_list)
output_data = np.vstack(onsetData_list)
np.save(input_file, input_data)
np.save(output_file, output_data)


