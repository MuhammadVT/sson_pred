import warnings
warnings.filterwarnings('ignore')
import pandas
import numpy
import datetime
import seaborn as sns
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
plt.style.use("fivethirtyeight")

def percentile(n):
    """A wrapper function for calculating the percentile"""
    def percentile_(x):
        return numpy.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def epoch_plot(param="Bz",
               end_deltm=60,
               input_file = "../data/pred_files/omn_cplng_profile_time_delayed_may10.csv"): 
    """ Makes an Epoch plot of the input feature
    Key argumets:
    param -- The parameter of interest to which we perform an epoch analysis
    end_deltm -- the end of the epoch time
    input_file -- The file path to the input file

    """
    # Store data into a pandas DataFrame
    predOmnPrflDF = pandas.read_csv(input_file, index_col=0)
    predOmnPrflDF["newell"] =  -1**(4/3)*numpy.power(numpy.abs(predOmnPrflDF["Vx"]),4./3) *\
                               numpy.power(predOmnPrflDF["B_T"], (2./3)) *\
                               (numpy.sin(predOmnPrflDF["theta_c"] / 2.))**(8./3)
    predOmnPrflDF["ByMagn"] = numpy.abs(predOmnPrflDF["By"])

    f, axArr = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
    # get the min al in the next 60 min
    delTBins = range(-120,end_deltm,10)
    oldColNames = predOmnPrflDF.columns.tolist()
    predDF2 = pandas.concat( [ predOmnPrflDF, \
                                pandas.cut( predOmnPrflDF["delTimeOnset"], \
                                                               bins=delTBins ) ], axis=1 )
    predDF2.columns = oldColNames + ["delT_bin"]
    predDF2 = predDF2[ predDF2["pred_type"].isin([ "TP", "FP", "FN", "TN"]) ]
    selDF = predDF2[ predDF2["delTimeOnset"] <= end_deltm  ]
    # Compute the min, mean and max (could also be other values)
    grouped = selDF.groupby(["pred_type", "delTimeOnset"]\
                           ).agg({param: [percentile(25), numpy.median, percentile(75)]}\
                                ).unstack("pred_type")

    # Getting the color palette used
    palette = sns.color_palette()

    # Initializing an index to get each cluster and each color
    predTypes = [ "TP", "FP", "FN", "TN" ]
    for _ind, _ax in enumerate(axArr):
        # 1st plot
        # boundaries
        _ax.fill_between(grouped.index, grouped.loc[:,(param, 'median', predTypes[_ind*2])], 
                        grouped.loc[:,(param, 'percentile_75', predTypes[_ind*2] )],\
                        alpha=.5, color=palette[_ind*2])
        _ax.fill_between(grouped.index, 
                        grouped.loc[:,(param, 'percentile_25', predTypes[_ind*2])] ,\
                        grouped.loc[:,(param, 'median', predTypes[_ind*2])],\
                        alpha=.5, color=palette[_ind*2])
        # median
        _ax.plot( grouped.index,grouped.loc[:,(param, 'median', predTypes[_ind*2])],\
                        color=palette[_ind*2], label=predTypes[_ind*2]  )
        # 2nd plot
        # boundaries
        _ax.fill_between(grouped.index, grouped.loc[:,(param, 'median', predTypes[_ind*2+1])], 
                        grouped.loc[:,(param, 'percentile_75', predTypes[_ind*2+1] )],\
                        alpha=.5, color=palette[_ind*2+1])
        _ax.fill_between(grouped.index, 
                        grouped.loc[:,(param, 'percentile_25', predTypes[_ind*2+1])] ,\
                        grouped.loc[:,(param, 'median', predTypes[_ind*2+1])],\
                        alpha=.5, color=palette[_ind*2+1])
        # median
        _ax.plot( grouped.index,grouped.loc[:,(param, 'median', predTypes[_ind*2+1])],\
                        color=palette[_ind*2+1], label=predTypes[_ind*2+1]  )
        
        _ax.tick_params(labelsize=10)
        _ax.legend(fontsize="medium")
        _ax.xaxis.set_ticks(numpy.arange(-120, end_deltm+1, 30))
        _ax.set_xlabel(r'$\Delta T_{pred}$ [Minutes]', fontsize="medium")
        _ax.tick_params(axis="both", which="major", labelsize=12)
        _ax.set_ylim([-5.5,5.5])
        
    axArr[0].set_ylabel("IMF " + param + " [nT]", fontsize="medium")    
    axArr[1].set_ylabel("")    
    f.savefig("./plots/epoch_analysis.png", bbox_inches='tight')
    plt.show()

