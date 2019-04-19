import warnings
warnings.filterwarnings('ignore')
import pandas
import datetime
import numpy
from scipy import stats
import sqlite3
import os
import sys
module_path = os.path.abspath(os.path.join('../data_pipeline/'))
if module_path not in sys.path:
    sys.path.append(module_path)
import omn_utils

# set some variables
# omn db
omnDBDir = "../data/sqlite3/"
omnDbName = "omni_sw_imf.sqlite"
omnTabName = "imf_sw"
omnParams = ["By", "Bz", "Bx", "Vx", "Np"]
omnTimeDelay = 10 #minutes
marchTime = 30 # minutes
onsetTimeInterval = 15 #minutes
# onset data
smlFname="../data/filtered-20190103-22-53-substorms.csv"
# some plotting vars
predDateRange = [datetime.datetime(1996,1,1), datetime.datetime(2018,1,1)]#None

# load onset data
ssOnsetDF = pandas.read_csv(smlFname,\
                                parse_dates=["Date_UTC"])
ssOnsetDF.columns = [ "date", "mlat", "mlt" ]
print("loaded onset data")
# get the time range and load OMNI data
if predDateRange is not None:
    omnStartDate = predDateRange[0] - datetime.timedelta(minutes=60)
    omnEndDate = predDateRange[1]
else:
    omnStartDate = ssOnsetDF["date"].min() - datetime.timedelta(minutes=60)
    omnEndDate = ssOnsetDF["date"].max()
# create the obj and load data
omnObj = omn_utils.OmnData(omnStartDate, omnEndDate, omnDBDir,\
                   omnDbName, omnTabName,\
                   True, None,\
                    imf_normalize=False,\
                    db_time_resolution=1,\
                    omn_train_params = omnParams)
# add a 10 minute time delay to the omni dataset
omnObj.omnDF["datetime"] = omnObj.omnDF["datetime"] + datetime.timedelta(minutes=omnTimeDelay)
omnObj.omnDF.set_index(omnObj.omnDF["datetime"], inplace=True)

# loop through and make the predictions
# using the conditions
predDict = {}
predDict["date"] = []
predDict["prediction"] = []
predDict["actual"] = []
predDict["onsetDate"] = []
predDict["triggerTime"] = []
nUnknown = 0
nFalse = 0
nTrue = 0

# Loop through the daterange and get number of onsets/false onsets etc
loopDate = predDateRange[0]
currMonth = loopDate.month
currYear = loopDate.year
print("working through month, year-->",currMonth, currYear)
while loopDate <= predDateRange[1]:
    if (currMonth != loopDate.month) | (currYear != loopDate.year):
        currMonth = loopDate.month
        currYear = loopDate.year
        print("working through month, year-->",currMonth, currYear)
    _sTime = loopDate - datetime.timedelta(minutes=30)
    _eTime = loopDate
    _currOmn = omnObj.omnDF[ _sTime : _eTime ].dropna()# Move to the next date
    loopDate += datetime.timedelta(minutes=marchTime)
    # Rule 1 : in the previous 30 minutes, atleast 22 minutes should have
    #           negative Bz. For this we also need to check if we have 30 
    #           values of Bz (sometimes data is missing)
    # there should be 30 values
    if _currOmn.shape[0] < 30:
        print("insufficient data", _currOmn.shape) 
        predDict["date"].append( loopDate )
        predDict["prediction"].append( "U" )
        predDict["triggerTime"].append( datetime.datetime(1970,1,1) )
        nUnknown += 1
        # check if there is an onset in the next 10 minutes
        _chkOnsets = ssOnsetDF[ (ssOnsetDF["date"] >= loopDate) &\
                            (ssOnsetDF["date"] <= (loopDate+datetime.timedelta(minutes=onsetTimeInterval))) ]
        if _chkOnsets.shape[0] > 0:
            predDict["onsetDate"].append( _chkOnsets["date"].tolist()[0] )
            predDict["actual"].append( "T" )
        else:
            predDict["onsetDate"].append( -1. )
            predDict["actual"].append( "F" )
        continue
    else:
        # count number of negative Bz
        nBz = _currOmn[_currOmn["Bz"] < 0.].shape[0]
        if nBz >= 22:
            # criteria1 succesful!
            predDict["date"].append( loopDate )
            predDict["prediction"].append( "T" )
            predDict["triggerTime"].append( loopDate )
            nTrue += 1
            # check if there is an onset in the next 10 minutes
            _chkOnsets = ssOnsetDF[ (ssOnsetDF["date"] >= loopDate) &\
                                (ssOnsetDF["date"] <= (loopDate+datetime.timedelta(minutes=onsetTimeInterval))) ]
            if _chkOnsets.shape[0] > 0:
                predDict["onsetDate"].append( _chkOnsets["date"].tolist()[0] )
                predDict["actual"].append( "T" )
            else:
                predDict["onsetDate"].append( -1. )
                predDict["actual"].append( "F" )
        else:
            # criteria1 failed! No SS
            predDict["date"].append( loopDate )
            predDict["prediction"].append( "F" )
            predDict["triggerTime"].append( datetime.datetime(1980,1,1) )
            nFalse += 1            
            # check if there is an onset in the next 10 minutes
            _chkOnsets = ssOnsetDF[ (ssOnsetDF["date"] >= loopDate) &\
                                (ssOnsetDF["date"] <= (loopDate+datetime.timedelta(minutes=onsetTimeInterval))) ]
            if _chkOnsets.shape[0] > 0:
                predDict["onsetDate"].append( _chkOnsets["date"].tolist()[0] )
                predDict["actual"].append( "T" )
            else:
                predDict["onsetDate"].append( -1. )
                predDict["actual"].append( "F" )
            continue
    
print("nTrue-->", nTrue)
print("nFalse-->", nFalse)
print("nUnknown-->", nUnknown)
# finally calculate all the TPs/FPs etc
predDF = pandas.DataFrame(predDict)

# save this DF 
predDF.to_csv("../data/lyons_marching_criteria1.csv")

print("##############TP count##############")
print(predDF[ (predDF["prediction"] == "T") & (predDF["actual"] == "T") ].shape)
print("##############TP count##############")

print("##############FP count##############")
print(predDF[ (predDF["prediction"] == "T") & (predDF["actual"] == "F") ].shape)
print("##############FP count##############")

print("##############TN count##############")
print(predDF[ (predDF["prediction"] == "F") & (predDF["actual"] == "F") ].shape)
print("##############TN count##############")

print("##############FN count##############")
print(predDF[ (predDF["prediction"] == "F") & (predDF["actual"] == "T") ].shape)
print("##############FN count##############")