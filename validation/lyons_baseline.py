import warnings
#warnings.filterwarnings('ignore')
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
omnParams = ["Bz"]
omnTimeDelay = 10 #minutes
# onset data
smlFname="../data/filtered-20190103-22-53-substorms.csv"
# some plotting vars
predDateRange = [datetime.datetime(1996,1,1), datetime.datetime(2006,1,1)]#None

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
timeAfterOnset = 25
nOmnPnts = timeBeforeOnset + timeAfterOnset + 1
# if predDateRange is None:
for _index, _row in ssOnsetDF.iterrows():
    print("Checking Onset at " + _row["date"].strftime("%Y-%m-%d %H:%M:%S"))
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
        print("insufficient data", omnDF_tmp.dropna().shape)
        predDict["date"].append( _row["date"] )
        predDict["prediction"].append( "U" )
        predDict["triggerTime"].append( 1000. )
        nUnknown += 1
        continue

    # get the corresponding omni data need for Rule 1
    _sTime = _row["date"] - datetime.timedelta(minutes=timeBeforeOnset)
    _eTime = _row["date"]
    _currOmn = omnDF_tmp.loc[ _sTime : _eTime]
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
    _stOnset = _row["date"] - datetime.timedelta(minutes=10)
    _etOnset = _row["date"] + datetime.timedelta(minutes=15)
    _omniNearOnset = omnDF_tmp[ _stOnset : _etOnset ].reset_index()
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
        rule_3 = False
        for _indx, _rw in _rapidNorthOmn.iterrows():
            # get the exact onset time
            _lyonsTriggerTimeOnset = _rw["datetime"]
            # Rule 3 : Sustained northward turning
            #           There are two sub-rules here:
            #           a) regression of slope of Bz between t0 and t0+10min > 1.75 nT/min
            #           b) Bz(t0 : t0 +3) >= Bz(t0) + 0.15
            #           c) Bz(t0+3:t0+10) >= Bz(t0) + 0.45
            # get the data for the next _lyonsTriggerTime to _lyonsTriggerTime + 10
            _triggerDFOnset = omnDF_tmp[ _lyonsTriggerTimeOnset : (_lyonsTriggerTimeOnset + datetime.timedelta(minutes=9)) ]
            _slopeOnset, _interceptOnset, _r_valueOnset, _p_valueOnset, _std_errOnset =\
                    stats.linregress( numpy.arange(_triggerDFOnset["Bz"].shape[0]), _triggerDFOnset["Bz"])

            if _slopeOnset >= 0.175:
                _selDF = _triggerDFOnset
                _trigTime = _lyonsTriggerTimeOnset - datetime.timedelta(minutes=1)
                _bzOnsetVal = omnDF_tmp.loc[_trigTime, :].Bz 
                _sel3MinBz = _selDF[ _selDF.index.min() + datetime.timedelta(minutes=1) : _selDF.index.min() + datetime.timedelta(minutes=2) ]
                _sel3to10MinBz = _selDF[ _selDF.index.min() + datetime.timedelta(minutes=3) : _selDF.index.min() + datetime.timedelta(minutes=9) ]
                if ( (_sel3MinBz["Bz"].min() >= _bzOnsetVal + 0.175) &\
                        (_sel3to10MinBz["Bz"].min() >= _bzOnsetVal + 0.45)):
                    # criteria1 succesful! we get a trigger
                    predDict["date"].append( _row["date"] )
                    predDict["prediction"].append( "T" )
                    predDict["triggerTime"].append( _trigTime )
                    nTrue += 1
                    rule_3 = True
                    break
        if not rule_3:
            # criteria failed! No SS
            predDict["date"].append( _row["date"] )
            predDict["prediction"].append( "F" )
            predDict["triggerTime"].append( -1000. )
            nFalse += 1
            continue

print("nTrue-->", nTrue)
print("nFalse-->", nFalse)
print("nUnknown-->", nUnknown)
print("%Pred-->",nTrue*1./(nTrue+nFalse))
print("%failed-->",nFalse*1./(nTrue+nFalse))

