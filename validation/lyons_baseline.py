import warnings
warnings.filterwarnings('ignore')
import pandas
import datetime
import numpy
from scipy import stats
import sqlite3
import os
import sys
import multiprocessing as mp
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
# onset data
smlFname="../data/filtered-20190103-22-53-substorms.csv"
# some plotting vars
predDateRange = [datetime.datetime(1996,1,1), datetime.datetime(1998,1,1)]#None

# load onset data
ssOnsetDF = pandas.read_csv(smlFname,\
                                parse_dates=["Date_UTC"])
ssOnsetDF.columns = [ "date", "mlat", "mlt" ]
print("loaded onset data")
# get the time range and load OMNI data
if predDateRange is not None:
    omnStartDate = predDateRange[0] - datetime.timedelta(minutes=60)
    omnEndDate = predDateRange[1]
    ssOnsetDF = ssOnsetDF.set_index("date").loc[omnStartDate:omnEndDate, :]
    ssOnsetDF.reset_index(inplace=True)
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
predDict["triggerTime"] = []
nUnknown = 0
nFalse = 0
nTrue = 0
# if predDateRange is None:
for _index, _row in ssOnsetDF.iterrows():
    # get the corresponding omni data
    _sTime = _row["date"] - datetime.timedelta(minutes=30)
    _eTime = _row["date"]
    _currOmn = omnObj.omnDF[ _sTime : _eTime ].dropna()
    _criteria1 = False
    _criteria2 = False
    _criteria3 = False
    _criteria4 = False
    _criteria5 = False
    # Rule 1 : in the previous 30 minutes, atleast 22 minutes should have
    #           negative Bz. For this we also need to check if we have 30 
    #           values of Bz (sometimes data is missing)
    # there should be 30 values
    if _currOmn.shape[0] < 30:
        print("insufficient data", _currOmn.shape) 
        predDict["date"].append( _row["date"] )
        predDict["prediction"].append( "U" )
        predDict["triggerTime"].append( 1000. )
        nUnknown += 1
        continue
    else:
        # count number of negative Bz
        nBz = _currOmn[_currOmn["Bz"] < 0.].shape[0]
        if nBz >= 22:
            _criteria1 = True
        else:
            # criteria1 failed! No SS
            predDict["date"].append( _row["date"] )
            predDict["prediction"].append( "F" )
            predDict["triggerTime"].append( -1000. )
            nFalse += 1
            continue

    # Rule 2 : A rapid northward turning must be observed close 
    #           to the onset. This is a bit tricky and we'll give
    #           some room for the criteria as exact onset time is a
    #           a little tricky, instead of taking exact onset time
    #            we'll take times +/- 15 minutes near the onset
    #           and see if we can find any rapid Bz northward turning
    _stOnset = _row["date"] - datetime.timedelta(minutes=15)
    _etOnset = _row["date"] + datetime.timedelta(minutes=15)
    _omniNearOnset = omnObj.omnDF[ _stOnset : _etOnset ].dropna()
    _omniNearOnset["delTOnset"] = (_omniNearOnset["datetime"] - _row["date"]).astype('timedelta64[m]')
    _omniNearOnset["delTAbs"] = numpy.abs( _omniNearOnset["delTOnset"] )    
    _omniNearOnset["diffBz"] = _omniNearOnset["Bz"].diff()
    # get only those times where diffBz is positive!
    _omniNearOnset = _omniNearOnset[ _omniNearOnset["diffBz"] >= 0. ]
    # identify all those locations which satisfy the second criterion!
    # i.e., diffBz > 0.375, if there are multiple such points choose 
    # the one closest to onset time (or largest shift in Bz)
    _rapidNorthOmn = _omniNearOnset[ _omniNearOnset["diffBz"] >= 0.375 ]

    if _rapidNorthOmn.shape[0] == 0:
        predDict["date"].append( _row["date"] )
        predDict["prediction"].append( "F" )
        predDict["triggerTime"].append( -1000. )
        nFalse += 1
        continue
    else:
        if _rapidNorthOmn.shape[0] > 1:
            _rapidNorthOmnClstOnset = _rapidNorthOmn[_rapidNorthOmn["delTAbs"] == _rapidNorthOmn["delTAbs"].min()]
            _rapidNorthOmnHighestChng = _rapidNorthOmn[_rapidNorthOmn["diffBz"] == _rapidNorthOmn["diffBz"].max()]
            # get the exact onset time
            _lyonsTriggerTimeClstOnset = _rapidNorthOmnClstOnset["datetime"].tolist()[0]
            _lyonsTriggerTimeHighestChng = _rapidNorthOmnHighestChng["datetime"].tolist()[0]
    # Rule 3 : Sustained northward turning
    #           There are two sub-rules here:
    #           a) regression of slope of Bz between t0 and t0+10min > 1.75 nT/min
    #           b) Bz(t0 : t0 +3) >= Bz(t0) + 0.15
    #           c) Bz(t0+3:t0+10) >= Bz(t0) + 0.45
    # get the data for the next _lyonsTriggerTime to _lyonsTriggerTime + 10
    _triggerDFClstOnset = omnObj.omnDF[ _lyonsTriggerTimeClstOnset + datetime.timedelta(minutes=1) : (_lyonsTriggerTimeClstOnset + datetime.timedelta(minutes=10)) ].dropna()
    _triggerDFHighestChng = omnObj.omnDF[ _lyonsTriggerTimeHighestChng + datetime.timedelta(minutes=1) : (_lyonsTriggerTimeHighestChng + datetime.timedelta(minutes=10)) ].dropna()
    _slopeClstOnset, _interceptClstOnset, _r_valueClstOnset, _p_valueClstOnset, _std_errClstOnset = stats.linregress( numpy.arange(_triggerDFClstOnset["Bz"].shape[0]), _triggerDFClstOnset["Bz"])
    _slopeClstHighestChng, _interceptHighestChng, _r_valueHighestChng, _p_valueHighestChng, _std_errHighestChng = stats.linregress( numpy.arange(_triggerDFHighestChng["Bz"].shape[0]), _triggerDFHighestChng["Bz"])

    if max(_slopeClstOnset, _slopeClstHighestChng ) < 0.175:
        # criteria1 failed! No SS
        predDict["date"].append( _row["date"] )
        predDict["prediction"].append( "F" )
        predDict["triggerTime"].append( -1000. )
        nFalse += 1
        continue
    else:
        if _slopeClstOnset > _slopeClstHighestChng:
            _selDF = _triggerDFClstOnset
            _bzOnsetVal = _selDF[ _selDF["datetime"] == _lyonsTriggerTimeClstOnset]["Bz"].tolist()[0]
            _trigTime = _lyonsTriggerTimeClstOnset
        else:
            _selDF = _triggerDFHighestChng
            _bzOnsetVal = _selDF[ _selDF["datetime"] == _lyonsTriggerTimeHighestChng]["Bz"].tolist()[0]
            _trigTime = _lyonsTriggerTimeHighestChng
        _selDF["delOnsetBz"] = _selDF["Bz"] - _bzOnsetVal
        import pdb
        pdb.set_trace()
        _sel3MinBz = _selDF[ _selDF.index.min() + datetime.timedelta(minutes=2) : _selDF.index.min() + datetime.timedelta(minutes=3) ]
        _sel3to10MinBz = _selDF[ _selDF.index.min() + datetime.timedelta(minutes=4) : _selDF.index.min() + datetime.timedelta(minutes=10) ]
        if ( (_sel3MinBz["Bz"].min() >= 0.15) & (_sel3to10MinBz["Bz"].min() >= 0.45) ):
            # criteria1 succesful! we get a trigger
            predDict["date"].append( _row["date"] )
            predDict["prediction"].append( "T" )
            predDict["triggerTime"].append( _trigTime )
            nTrue += 1
        else:
            # criteria1 failed! No SS
            predDict["date"].append( _row["date"] )
            predDict["prediction"].append( "F" )
            predDict["triggerTime"].append( -1000. )
            nFalse += 1
print("nTrue-->", nTrue)
print("nFalse-->", nFalse)
print("nUnknown-->", nUnknown)
print("%Pred-->",nTrue*1./(nTrue+nFalse))
print("%failed-->",nFalse*1./(nTrue+nFalse))

