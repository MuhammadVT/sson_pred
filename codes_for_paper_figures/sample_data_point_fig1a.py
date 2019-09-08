import datetime
import pandas
import numpy
import seaborn as sns
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.dates import DateFormatter
import os
import sys
module_path = os.path.abspath(os.path.join('../data_pipeline/'))
if module_path not in sys.path:
    sys.path.append(module_path)
sys.path.append("../../dnn_substorm_onset/data_pipeline/")
import omn_utils
import sqlite3

class EventSummary(object):
    """
    A class to take an individual SS onset event
    and plot different geophysical parameters within
    a certain time range of the event (Bz, Vx, AE, SML)
    and compare the difference between predicted and 
    actual results. We'll also be able to have a sanity
    check during these events.
    """
    def __init__(self, paramTimeRange, paramDBDir, omnDbName, \
             omnTabName, aulDbName, aulTabName, smlDbName, smlTabName,\
             plotTimeHist=120, plotFutureBins=2,\
             omnParams = ["By", "Bz", "Bx", "Vx", "Np"], \
             smParams=["al"],\
             binTimeRes=60, nBins=1,\
             figDir="../plots/"):
        """
        onsetTime : the time of onset or time under consideration
        plotTimeHist : number of minutes before onset to plot.
        plotFutureBins : number of bins (in addition to nBins) to plot.
                        For example, if onset is at 5 UT
                       and plotTimeHist is 120 minutes and plotFutureBins is 2,
                       nBins=2 and binTimeRes is 30. Then the predictions are made
                       upto 60 minutes in to the future (nBins*binTimeRes) and we'll
                       pad the 2 additional bins into the plot (60 minutes). So the
                       plot's time range would be 3 UT to 7 UT.
        paramTimeRange : Time range to load the parameters (IMF, AU/AL...) into a DF
                        If set to None. The plotTimeRange will be used, else we use this 
                        time range. This is useful when making multiple plots and it is 
                        time consuming to query the database multiple times.
        """
        self.paramTimeRange = paramTimeRange
        self.figDir = figDir
        self.binTimeRes = binTimeRes
        self.nBins = nBins
        self.plotTimeHist = plotTimeHist
        self.plotFutureBins = plotFutureBins
        # db params
        self.paramDBDir = paramDBDir
        # geophysical params
        self.omnParams = omnParams
        self.smParams = smParams
        # Load omni data
        if len(omnParams) > 0:
            self.omnDF = self._load_omn_data(omnDbName, omnTabName)
            print("loaded OMNI data")
        else:
            self.omnDF = None
        # Load sml data
        if len(smParams) > 0:
            self.smDF = self._load_ind_data(smlDbName, smlTabName)
            self.smDF = self.smDF[ ["datetime"] + self.smParams ]
            print("loaded SML data")
        else:
            self.smDF = None
        # set colors for shading regions of True positives, 
        # True Negatives, False positives and negatives.
        self.shadeColDict = {}
        self.shadeColDict["TP"] = "#2d6d66"
        self.shadeColDict["TN"] = "#1a476f"
        self.shadeColDict["FP"] = "#90353b"
        self.shadeColDict["FN"] = "#e30e00"

    def _load_omn_data(self, omnDbName, omnTabName):
        """
        Load omn data
        """
        # get the time range from onset data
        omnStartDate = self.paramTimeRange[0] - datetime.timedelta(\
                        minutes=self.plotTimeHist)
        omnEndDate = self.paramTimeRange[1]
        # create the obj and load data
        omnObj = omn_utils.OmnData(omnStartDate, omnEndDate, self.paramDBDir,\
                           omnDbName, omnTabName,\
                           True, None,\
                            imf_normalize=False,\
                            db_time_resolution=1,\
                            omn_train_params = self.omnParams)

        # Add IMF propagation delay
        imf_delay = 10
        omnObj.omnDF.datetime = omnObj.omnDF.datetime +\
                                datetime.timedelta(minutes=imf_delay)
        return omnObj.omnDF 


    def _load_ind_data(self, dbName, tabName):
        """
        Load AUL data
        """
        conn = sqlite3.connect(self.paramDBDir + dbName,
                       detect_types = sqlite3.PARSE_DECLTYPES)
        # load data to a dataframe
        command = "SELECT * FROM {tb} " +\
                  "WHERE datetime BETWEEN '{stm}' and '{etm}'"
        command = command.format(tb=tabName, stm=self.paramTimeRange[0],\
                                  etm=self.paramTimeRange[1])
        return pandas.read_sql(command, conn)
    
    def generate_onset_plot(self, actualLab, predLab,\
                      onsetTime, predLabProb=None, figType="png"):
        """
        Generate the plot.
        """
        # get the plot details
        # get the time range of the plot
        predStart = onsetTime - datetime.timedelta(minutes=30)
        predEnd = predStart + datetime.timedelta(minutes=60)
        binStart = predStart - datetime.timedelta(minutes=self.plotTimeHist) 
        binEnd = predStart 

        prevTime = binStart
        futTime = predEnd + datetime.timedelta(minutes=60)
        plotTimeRange = [ prevTime, futTime]
        # set plot styling
        plt.style.use("fivethirtyeight")
        f = plt.figure(figsize=(12, 8))
        ax = f.add_subplot(1,1,1)
        # get the number of panels
        nPanels = len(self.omnParams) +\
                    len(self.smParams)
        fig, axes = plt.subplots(nrows=nPanels, ncols=1,\
                                 figsize=(8,8), sharex=True)
        # axis formatting
        dtLabFmt = DateFormatter('%H:%M')
        axCnt = 0
        # plot omni IMF
        for _op in self.omnParams:
            currOmnDF = self.omnDF[ \
                (self.omnDF["datetime"] >= plotTimeRange[0]) &\
                (self.omnDF["datetime"] <= plotTimeRange[1]) ]
            axes[axCnt].plot( currOmnDF["datetime"].values,\
                          currOmnDF[_op].values, linewidth=2 )
            axes[axCnt].set_ylabel(_op, fontsize=14)
            axes[axCnt].xaxis.set_major_formatter(dtLabFmt)
            axCnt += 1
        # plot supermag indices
        for _smp in self.smParams:
            currSmDF = self.smDF[ \
                (self.smDF["datetime"] >= plotTimeRange[0]) &\
                (self.smDF["datetime"] <= plotTimeRange[1]) ]
            axes[axCnt].plot( currSmDF["datetime"].values,\
                          currSmDF[_smp].values, linewidth=3, color="darkorange")
            axes[axCnt].set_ylabel("SML", fontsize=14)
            axes[axCnt].xaxis.set_major_formatter(dtLabFmt)
        # mark the actual onset time and shade the region 
        # based on pred onset.
        # shade the region based on the type (TP/TN/FP/FN)
        for _nax, _ax in enumerate(axes):
            # plot vertical lines to mark the prediction bins
            currCol = "gray" 
            _ax.axvline(x=onsetTime, color='r', linestyle='--',\
                        linewidth=2)
            _ax.axvline(x=binEnd, color='k', linestyle='--',\
                        linewidth=1)
            _ax.axvline(x=predEnd, color='k', linestyle='--',\
                        linewidth=1)
            if _nax < len(axes)-1:
                _ax.axvspan(binStart, binEnd, alpha=0.5,\
                         color=currCol)
            else:
                _ax.axvspan(predStart, predEnd, alpha=0.5,\
                         color="orange")
        # get the figure name
        figName = "sw_imf_input_"+ \
                   onsetTime.strftime("%Y%m%d-%H%M") + "." + figType
        # Labeling and formatting
        plt.xlim([plotTimeRange[0], plotTimeRange[1]])
        plt.xlabel("Time UT")
        plt.tick_params(labelsize=14)
        fig.suptitle(onsetTime.strftime("%Y-%m-%d"))
        fig.savefig(self.figDir + figName, bbox_inches='tight')           
        fig.clf()
        plt.close()

# Run the code
if __name__ == "__main__":
    onsetTime = datetime.datetime(1997, 1, 3, 21, 50)
    paramTimeRange = [onsetTime - datetime.timedelta(minutes=180),
                      onsetTime + datetime.timedelta(minutes=120)]
    actualLab = [0,1]
    predLab = [1,0]
    omnDBDir = "../data/sqlite3/"
    omnDbName = "omni_sw_imf.sqlite"
    omnTabName = "imf_sw"
    aulDbName = "au_al_ae.sqlite"
    aulTabName = "aualae"
    smlDbName = "smu_sml_sme.sqlite"
    smlTabName = "smusmlsme"
    esObj = EventSummary(paramTimeRange, omnDBDir, omnDbName, omnTabName, aulDbName,
                    aulTabName, smlDbName, smlTabName)

    esObj.generate_onset_plot(actualLab, predLab,
                              onsetTime, predLabProb=None, figType="png")

