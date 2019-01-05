import datetime
import pandas
import numpy
import feather
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os
import sys
module_path = os.path.abspath(os.path.join('../data_pipeline/'))
if module_path not in sys.path:
    sys.path.append(module_path)
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
    def __init__(self, eventDate, actualLab, predLab, paramDBDir, omnDbName, \
             omnTabName, aulDbName, aulTabName, smlDbName, smlTabName,\
             modelPredFname, plotTimeHist=120, plotFutureBins=2,\
             omnParams = ["By", "Bz", "Bx", "Vx", "Np"], \
             aulParams = ["au", "al"],smParams=["au", "al"],\
             binTimeRes=30, nBins=2, paramTimeRange=None,\
             figDir="/home/bharat/Documents/data/ss_onset_dataset/plots/"):
        """
        eventDate : the time of onset or time under consideration
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
        self.eventDate = eventDate
        self.actualLab = actualLab
        self.predLab = predLab
        self.figDir = figDir
        # db params
        self.paramDBDir = paramDBDir
        # params
        self.omnParams = omnParams
        self.aulParams = aulParams
        self.smParams = smParams
        # get the time range of the plot
        prevTime = self.eventDate - datetime.timedelta(minutes=plotTimeHist)
        futTime = self.eventDate + datetime.timedelta(\
                                    minutes=(nBins+plotFutureBins)*binTimeRes)
        self.plotTimeRange = [ prevTime, futTime]
        if paramTimeRange is None:
            self.paramTimeRange = self.plotTimeRange
        else:
            self.paramTimeRange = paramTimeRange
        print "plot time -->", self.eventDate
        print "plot time range -->", self.plotTimeRange
        # Load omni data
        if len(omnParams) > 0:
            self.omnDF = self._load_omn_data(omnDbName, omnTabName)
        else:
            self.omnDF = None
        # Load aul data
        if len(aulParams) > 0:
            self.aulDF = self._load_ind_data(aulDbName, aulTabName)
            self.aulDF = self.aulDF[ ["datetime"] + self.aulParams ]
        else:
            self.aulDF = None
        # Load sml data
        if len(smParams) > 0:
            self.smDF = self._load_ind_data(smlDbName, smlTabName)
            self.smDF = self.smDF[ ["datetime"] + self.smParams ]
        else:
            self.smDF = None
        # load the model pred data labels
        self.predDF = self._load_model_pred_data(modelPredFname, nBins)

    def _load_omn_data(self, omnDbName, omnTabName):
        """
        Load omn data
        """
        # get the time range from onset data
        omnStartDate = self.paramTimeRange[0]
        omnEndDate = self.paramTimeRange[1]
        # create the obj and load data
        omnObj = omn_utils.OmnData(omnStartDate, omnEndDate, self.paramDBDir,\
                           omnDbName, omnTabName,\
                           True, None,\
                            imf_normalize=False,\
                            db_time_resolution=1,\
                            omn_train_params = self.omnParams)
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

    def _load_model_pred_data(self, fileName, nBins):
        """
        Load the model predictions into a DF
        """
        # we need to carefully setup the column names
        colNames = ["date"]
        for _nb in range(nBins):
            colNames += [ "bin_" + str(_nb) ]
        colNames += [ "label", "pred_label" ] 
        return pandas.read_csv(fileName, names=colNames,\
                     header=0, parse_dates=["date"])
    
    def generate_plot(self, figType="png"):
        """
        Generate the plot.
        """
        # set plot styling
        plt.style.use("fivethirtyeight")
        f = plt.figure(figsize=(12, 8))
        ax = f.add_subplot(1,1,1)
        # get the number of panels
        nPanels = len(self.omnParams) +\
                    len(self.aulParams)-1 + len(self.smParams)-1
        fig, axes = plt.subplots(nrows=nPanels, ncols=1,\
                                 figsize=(8,8), sharex=True)
        axCnt = 0
        # plot omni IMF
        for _op in self.omnParams:
            currOmnDF = self.omnDF[ \
                (self.omnDF["datetime"] >= self.plotTimeRange[0]) &\
                (self.omnDF["datetime"] <= self.plotTimeRange[1]) ]
            axes[axCnt].plot_date( currOmnDF["datetime"].values,\
                          currOmnDF[_op].values )
            axes[axCnt].set_ylabel(_op, fontsize=14)
            axCnt += 1
        # plot auroral indices
        for _aup in self.aulParams:
            currAulDF = self.aulDF[ \
                (self.aulDF["datetime"] >= self.plotTimeRange[0]) &\
                (self.aulDF["datetime"] <= self.plotTimeRange[1]) ]
            axes[axCnt].plot_date( currAulDF["datetime"].values,\
                          currAulDF[_aup].values )
            axes[axCnt].set_ylabel("AU/AL", fontsize=14)
        axCnt += 1
        # plot supermag indices
        for _smp in self.smParams:
            currSmDF = self.smDF[ \
                (self.smDF["datetime"] >= self.plotTimeRange[0]) &\
                (self.smDF["datetime"] <= self.plotTimeRange[1]) ]
            axes[axCnt].plot_date( currSmDF["datetime"].values,\
                          currSmDF[_smp].values )
            axes[axCnt].set_ylabel("SMU/SML", fontsize=14)
        axCnt += 1
        # get the figure name
        figName = "stack_event_" +\
                self.eventDate.strftime("%Y%m%d-%H%M") + "." + figType
        # Labeling
        plt.xlabel("Time UT")
        plt.tick_params(labelsize=14)
        fig.savefig(self.figDir + figName, bbox_inches='tight')
            

