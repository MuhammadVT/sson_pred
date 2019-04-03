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

def load_omn_data(omnStartDate, omnEndDate, omnDBDir,
                  omnDbName, omnTabName,
                  db_time_resolution=1,
                  omn_train_params = ["By", "Bz", "Bx", "Vx", "Np"]):
    """
    Load omni data
    """
    conn = sqlite3.connect(omnDBDir + omnDbName,
                   detect_types = sqlite3.PARSE_DECLTYPES)
    # load data to a dataframe
    command = "SELECT * FROM {tb} WHERE datetime BETWEEN '{stm}' AND '{etm}'"
    command = command.format(tb=omnTabName,\
                             stm=omnStartDate, etm=omnEndDate)
    omnDF = pandas.read_sql(command, conn)
    omnDF = omnDF[ omn_train_params + [ "datetime" ] ]
    omnDF = omnDF.replace(numpy.inf, numpy.nan)
    omnDF = omnDF.set_index("datetime")
    # Add omnStartDate to omnDF in case if it is missing
    if omnStartDate not in omnDF.index:
        omnDF.loc[omnStartDate] = numpy.nan
    # Add omnEndDate to omnDF in case if it is missing
    if omnEndDate not in omnDF.index:
        omnDF.loc[omnEndDate] = numpy.nan
    omnDF.sort_index(inplace=True)
    # Resample the data
    omnDF = omnDF.resample(str(db_time_resolution) + "Min").asfreq().reset_index()

    return omnDF

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
predDateRange = [datetime.datetime(1996,1,1), datetime.datetime(1997,1,1)]#None

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
omnDF = load_omn_data(omnStartDate, omnEndDate, omnDBDir,
                      omnDbName, omnTabName,
                      db_time_resolution=1,
                      omn_train_params = omnParams)
# add a 10 minute time delay to the omni dataset
omnDF["datetime"] = omnDF["datetime"] + datetime.timedelta(minutes=omnTimeDelay)
omnDF.set_index("datetime", inplace=True)

# loop through and make the predictions
# using the conditions
predDict = {}
predDict["date"] = []
predDict["prediction"] = []
predDict["triggerTime"] = []
nUnknown = 0
nFalse = 0
nTrue = 0
timeBeforeOnset = 29
timeAfterOnset = 15
nOmnPnts = timeBeforeOnset + timeAfterOnset + 1
# if predDateRange is None:
for _index, _row in ssOnsetDF.iterrows():
    # get omni data for need for current Onset
    _sTm = _row["date"] - datetime.timedelta(minutes=timeBeforeOnset)
    _eTm = _row["date"] + datetime.timedelta(minutes=timeAfterOnset)
    omnDF_tmp = omnDF.loc[ _sTm : _eTm]

    # Interpolate the omni data if # missing data points are small
    # If not discard this time interval
    if omnDF_tmp.dropna().shape[0] >= nOmnPnts - 5:
        if omnDF_tmp.dropna().shape[0] < nOmnPnts:
            omnDF_tmp.interpolate(method="linear", limit_direction="both", inplace=True)
    else:
        print("insufficient data", omnDF_tmp.shape) 
        predDict["date"].append( _row["date"] )
        predDict["prediction"].append( "U" )
        predDict["triggerTime"].append( 1000. )
        nUnknown += 1
        continue

    # get the corresponding omni data need for Rule 1
    _sTime = _row["date"] - datetime.timedelta(minutes=timeBeforeOnset)
    _eTime = _row["date"]
    _currOmn = omnDF_tmp.loc[ _sTime : _eTime]
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
        import pdb
        pdb.set_trace()
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
    _etOnset = _row["date"] + datetime.timedelta(minutes=timeAfterOnset)
    _omniNearOnset = omnDF[ _stOnset : _etOnset ].dropna()
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
    _triggerDFClstOnset = omnDF[ _lyonsTriggerTimeClstOnset + datetime.timedelta(minutes=1) : (_lyonsTriggerTimeClstOnset + datetime.timedelta(minutes=10)) ].dropna()
    _triggerDFHighestChng = omnDF[ _lyonsTriggerTimeHighestChng + datetime.timedelta(minutes=1) : (_lyonsTriggerTimeHighestChng + datetime.timedelta(minutes=10)) ].dropna()
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




