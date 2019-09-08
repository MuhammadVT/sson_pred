import warnings
warnings.filterwarnings('ignore')
import pandas
import datetime
import seaborn as sns
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from matplotlib.ticker import MultipleLocator
plt.style.use("fivethirtyeight")

def pred_bin_out(row, nBins, binTimeRes):
    """
    Given the prediction label, get the actual
    output in bins by converting the label into
    binary representation. For ex, label 2 would
    convert to 10 and 5 to 101 and so on.
    """
    # Note we need the binary format to be consistent
    # with the actual labels, i.e., it depends on the 
    # number of bins. For example, 2 could be 10 or 010.
    binFormtStr = '{0:0' + str(nBins) + 'b}'
    predBinStr = binFormtStr.format(row["pred_label"])
    # Now add these into different pred bins
    for _n, _pb in enumerate(predBinStr):
        row["pbin_" + str(_n)] = int(_pb)
    if row["label"] == 0:
        if row["pred_label"] == 0:
            predType = "TN"
        else:
            predType = "FP"
    if row["label"] == 1:
        if row["pred_label"] == 1:
            predType = "TP"
        else:
            predType = "FN"
    row["pred_type"] = predType
    # Now calculate time from prev onset
    # and time to next onset
    _cpDate = row["datetime"]
    srchStTime = _cpDate.strftime("%Y-%m-%d %H:%M:%S")
    _nextOnsetTime = ssOnDF[ ssOnDF.index > srchStTime ].index.min()
    if not pandas.isna(_nextOnsetTime):
        _nextDelT = (_nextOnsetTime-_cpDate).total_seconds()/60.
    else:
        _nextDelT = -1
    _priorOnsetTime = ssOnDF[ ssOnDF.index < srchStTime ].index.max()
    if not pandas.isna(_priorOnsetTime):
        _priorDelT = (_cpDate-_priorOnsetTime).total_seconds()/60.
    else:
        _priorDelT = -1
    row["next_onset"] = _nextDelT
    row["prior_onset"] = _priorDelT
    # finally get the number of substorms in the bin
    srchETime = (_cpDate + datetime.timedelta(minutes=(binTimeRes+30)*nBins)\
                ).strftime("%Y-%m-%d %H:%M:%S")
    row["num_onsets"] = len( ssOnDF[ srchStTime : srchETime ].index.tolist() )
    return row

def get_sml_vars(row):
    """
    Get mean, median, std, min and max of sml 
    during various substorms over the next interval range.
    """
    delTimeList = [30, 60]#[ 15, 30, 60, 120 ]
    for _dtl in delTimeList:
        _pd = row["datetime"] - datetime.timedelta(minutes=10)
        _cd = row["datetime"] + datetime.timedelta(minutes=1)
        _ed = row["datetime"] + datetime.timedelta(minutes=_dtl)
        _resDF = smlDF[ _cd : _ed ]
        _baselineAl = smlDF[ _pd : _cd ]["al"].median()
        _baselineAe = smlDF[ _pd : _cd ]["ae"].median()
        row["mean_al_" + str(_dtl)] = _resDF["al"].mean()
        row["median_al_" + str(_dtl)] = _resDF["al"].median()
        row["min_al_" + str(_dtl)] = _resDF["al"].min()
        row["max_al_" + str(_dtl)] = _resDF["al"].max()
        # difference between current AL and minimum in the next bin
        # note this is defined to be negative, for easy binning etc
        row["al_dip" + str(_dtl)] = _resDF["al"].min() - _baselineAl
        row["ae_dip" + str(_dtl)] = _resDF["ae"].max() - _baselineAe
    return row

# Load onset data
fName = "../data/filtered-20190103-22-53-substorms.csv"
ssOnDF = pandas.read_csv(fName, parse_dates=["Date_UTC"])
# rename the cols
ssOnDF.columns = [ "datetime", "mlat", "mlt" ]
# convert to hour
ssOnDF.set_index( pandas.to_datetime(\
                        ssOnDF["datetime"]), inplace=True )
nBins = 1
binRes = 60
predDF = pandas.read_csv("../data/pred_files/resnet_test_data_pred.csv",\
                     header=0, parse_dates=["datetime"])
predDF = predDF.apply( pred_bin_out, args=(nBins,binRes,),\
                      axis=1 )

# Load & process SML data
start_date = predDF["datetime"].min() - datetime.timedelta(hours=2)
end_date = predDF["datetime"].max()
print(start_date, end_date)
omn_dbdir = "../data/sqlite3/"
omn_db_name = "smu_sml_sme.sqlite"
omn_table_name = "smusmlsme"
# read omni data
conn = sqlite3.connect(omn_dbdir + omn_db_name,
                       detect_types = sqlite3.PARSE_DECLTYPES)
# load data to a dataframe
command = "SELECT datetime, al, ae, au FROM {tb} WHERE datetime BETWEEN '{stm}' and '{etm}'"
command = command.format(tb=omn_table_name,\
                         stm=start_date, etm=end_date)
smlDF = pandas.read_sql(command, conn)
# drop nan's
smlDF.dropna(inplace=True)
smlDF.set_index(smlDF["datetime"], inplace=True)
smlDF.head()
predDF = predDF.apply( get_sml_vars, axis=1 )

#import pdb
#pdb.set_trace()

# Bin by AL
#alBins = range(-700,100,100)
alBins = [-800, -500, -400, -300, -200, -100, 0]
# get the min al in the next 30 min
oldColNames = predDF.columns.tolist()
predDF = pandas.concat( [ predDF, \
                    pandas.cut( predDF["min_al_30"], \
                               bins=alBins ) ], axis=1 )
predDF.columns = oldColNames + ["min_al_30_bin"]
# get the min al in the next 60 min
oldColNames = predDF.columns.tolist()
predDF = pandas.concat( [ predDF, \
                    pandas.cut( predDF["min_al_60"], \
                               bins=alBins ) ], axis=1 )
predDF.columns = oldColNames + ["min_al_60_bin"]
# get the AL in the next 30 min
oldColNames = predDF.columns.tolist()
predDF = pandas.concat( [ predDF, \
                    pandas.cut( predDF["al_dip30"], \
                               bins=alBins ) ], axis=1 )
predDF.columns = oldColNames + ["al_dip30_bin"]
# get the AL in the next 60 min
oldColNames = predDF.columns.tolist()
predDF = pandas.concat( [ predDF, \
                    pandas.cut( predDF["al_dip60"], \
                               bins=alBins ) ], axis=1 )
predDF.columns = oldColNames + ["al_dip60_bin"]
predDF.dropna(inplace=True)

f, axArr = plt.subplots(2, sharex=True)
sns.countplot(x="al_dip60_bin", hue="pred_type",\
              data=predDF, ax=axArr[0],\
              hue_order=["TP", "FP", "FN", "TN"])
sns.boxplot( x="al_dip60_bin", y="prob_1",\
             hue_order=["TP", "FP", "FN", "TN"],
             hue="pred_type", data=predDF,\
             showfliers=False, ax=axArr[1], linewidth=0.5 )
# log y-scale
# axArr[0].set_yscale("log")
# xlabels
axArr[1].set_xlabel(r'$\Delta_{SML}$ [nT]')
axArr[0].set_xlabel("")
#ylabels
axArr[1].set_ylabel(r'$P_{onset}$')
# legends
axArr[0].legend(loc="center left", prop={'size': 8})
axArr[1].legend().set_visible(False)
# ticks
axArr[0].tick_params(axis='both', which='major', labelsize=10)
axArr[1].tick_params(axis='both', which='major', labelsize=10)
plt.xticks(rotation=45)
axArr[0].annotate("(a)", xy=(0.05, 0.95), xycoords="axes fraction", fontweight="bold")
axArr[1].annotate("(b)", xy=(0.05, 0.95), xycoords="axes fraction", fontweight="bold")
axArr[1].yaxis.set_major_locator(MultipleLocator(0.25))

f.savefig("../plots/paper-figures/fig_5.png", bbox_inches='tight')

