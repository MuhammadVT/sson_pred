import datetime
import pandas
import numpy
import feather
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
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
    def __init__(self, paramTimeRange, paramDBDir, omnDbName, \
             omnTabName, aulDbName, aulTabName, smlDbName, smlTabName,\
             plotTimeHist=120, plotFutureBins=2,\
             omnParams = ["By", "Bz", "Bx", "Vx", "Np"], \
             aulParams = ["au", "al"],smParams=["au", "al"],\
             binTimeRes=30, nBins=2,\
             figDir="/home/bharat/Documents/data/ss_onset_dataset/onset_plots/"):
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
        self.aulParams = aulParams
        self.smParams = smParams
        # Load omni data
        if len(omnParams) > 0:
            self.omnDF = self._load_omn_data(omnDbName, omnTabName)
            print("loaded OMNI data")
        else:
            self.omnDF = None
        # Load aul data
        if len(aulParams) > 0:
            self.aulDF = self._load_ind_data(aulDbName, aulTabName)
            self.aulDF = self.aulDF[ ["datetime"] + self.aulParams ]
            print("loaded AUL data")
        else:
            self.aulDF = None
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
    
    def generate_bin_plot(self, eventDate, actualLab, predLab,\
                      figType="png"):
        """
        Generate the plot.
        """
        # get the plot details
        # get the time range of the plot
        prevTime = eventDate - datetime.timedelta(minutes=self.plotTimeHist)
        futTime = eventDate + datetime.timedelta(\
                                    minutes=(self.nBins+self.plotFutureBins\
                                    )*self.binTimeRes)
        plotTimeRange = [ prevTime, futTime]
        # set plot styling
        plt.style.use("fivethirtyeight")
        f = plt.figure(figsize=(12, 8))
        ax = f.add_subplot(1,1,1)
        # get the number of panels
        nPanels = len(self.omnParams) +\
                    len(self.aulParams)-1 + len(self.smParams)-1
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
        # plot auroral indices
        for _aup in self.aulParams:
            currAulDF = self.aulDF[ \
                (self.aulDF["datetime"] >= plotTimeRange[0]) &\
                (self.aulDF["datetime"] <= plotTimeRange[1]) ]
            axes[axCnt].plot( currAulDF["datetime"].values,\
                          currAulDF[_aup].values, linewidth=2 )
            axes[axCnt].set_ylabel("AU/AL", fontsize=14)
            axes[axCnt].xaxis.set_major_formatter(dtLabFmt)
        axCnt += 1
        # plot supermag indices
        for _smp in self.smParams:
            currSmDF = self.smDF[ \
                (self.smDF["datetime"] >= plotTimeRange[0]) &\
                (self.smDF["datetime"] <= plotTimeRange[1]) ]
            axes[axCnt].plot( currSmDF["datetime"].values,\
                          currSmDF[_smp].values, linewidth=2 )
            axes[axCnt].set_ylabel("SMU/SML", fontsize=14)
            axes[axCnt].xaxis.set_major_formatter(dtLabFmt)
        axCnt += 1
        # shade the region based on the type (TP/TN/FP/FN)
        for _nax, _ax in enumerate(axes):
            # plot vertical lines to mark the prediction bins
            for _nb in range(self.nBins+1):
                binStart = eventDate + datetime.timedelta(\
                            minutes=_nb*self.binTimeRes)
                _ax.axvline(x=binStart, color='k', linestyle='--',\
                            linewidth=0.5)
                if _nb < self.nBins:
                    binEnd = eventDate + datetime.timedelta(\
                            minutes=(_nb+1)*self.binTimeRes)
                    trueNegative = False
                    if actualLab[_nb] == 0:
                        if predLab[_nb] == 0:
                            currCol = self.shadeColDict["TN"]
                            textOut = "TN"
                            trueNegative = True
                        else:
                            currCol = self.shadeColDict["FP"]
                            textOut = "FP"
                    if actualLab[_nb] == 1:
                        if predLab[_nb] == 1:
                            currCol = self.shadeColDict["TP"]
                            textOut = "TP"
                        else:
                            currCol = self.shadeColDict["FN"]
                            textOut = "FN"
                    if not trueNegative:
                        _ax.axvspan(binStart, binEnd, alpha=0.5, color=currCol)
                    if _nax == 0 :
                        textXLoc = eventDate + datetime.timedelta(\
                                minutes=(_nb+0.5)*self.binTimeRes) 
                        textYLoc = _ax.get_ylim()[1] - (\
                             abs(_ax.get_ylim()[1]) - _ax.get_ylim()[0] )/2
                        _ax.text(textXLoc, textYLoc, textOut)
        # get the figure name
        figName = "bins_binRes_" + str(self.binTimeRes) + \
                    "_nbins_" + str(self.nBins) + "_stack_event_" +\
                    eventDate.strftime("%Y%m%d-%H%M") + "." + figType
        # Labeling and formatting
        plt.xlim([plotTimeRange[0], plotTimeRange[1]])
        plt.xlabel("Time UT")
        plt.tick_params(labelsize=14)
        fig.suptitle(eventDate.strftime("%Y-%m-%d"))
        fig.savefig(self.figDir + figName, bbox_inches='tight')           
        fig.clf()
        plt.close()

    def generate_onset_plot(self, eventDate, actualLab, predLab,\
                      onsetTimeDict, figType="png"):
        """
        Generate the plot.
        """
        # get the plot details
        # get the time range of the plot
        prevTime = eventDate - datetime.timedelta(minutes=self.plotTimeHist)
        futTime = eventDate + datetime.timedelta(\
                                    minutes=(self.nBins+self.plotFutureBins\
                                    )*self.binTimeRes)
        plotTimeRange = [ prevTime, futTime]
        # set plot styling
        plt.style.use("fivethirtyeight")
        f = plt.figure(figsize=(12, 8))
        ax = f.add_subplot(1,1,1)
        # get the number of panels
        nPanels = len(self.omnParams) +\
                    len(self.aulParams)-1 + len(self.smParams)-1
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
        # plot auroral indices
        for _aup in self.aulParams:
            currAulDF = self.aulDF[ \
                (self.aulDF["datetime"] >= plotTimeRange[0]) &\
                (self.aulDF["datetime"] <= plotTimeRange[1]) ]
            axes[axCnt].plot( currAulDF["datetime"].values,\
                          currAulDF[_aup].values, linewidth=2 )
            axes[axCnt].set_ylabel("AU/AL", fontsize=14)
            axes[axCnt].xaxis.set_major_formatter(dtLabFmt)
        axCnt += 1
        # plot supermag indices
        for _smp in self.smParams:
            currSmDF = self.smDF[ \
                (self.smDF["datetime"] >= plotTimeRange[0]) &\
                (self.smDF["datetime"] <= plotTimeRange[1]) ]
            axes[axCnt].plot( currSmDF["datetime"].values,\
                          currSmDF[_smp].values, linewidth=2 )
            axes[axCnt].set_ylabel("SMU/SML", fontsize=14)
            axes[axCnt].xaxis.set_major_formatter(dtLabFmt)
        axCnt += 1
        # mark the actual onset time  and shade the region 
        # based on pred onset.
        # shade the region based on the type (TP/TN/FP/FN)
        for _nax, _ax in enumerate(axes):
            # plot vertical lines to mark the prediction bins
            for _nb in range(self.nBins+1):
                binStart = eventDate + datetime.timedelta(\
                            minutes=_nb*self.binTimeRes)
                _ax.axvline(x=binStart, color='k', linestyle='--',\
                            linewidth=1)
                if _nb < self.nBins:
                    binEnd = eventDate + datetime.timedelta(\
                            minutes=(_nb+1)*self.binTimeRes)
                    trueNegative = False
                    if actualLab[_nb] == 0:
                        if predLab[_nb] == 0:
                            currCol = self.shadeColDict["TN"]
                            textOut = "TN"
                            trueNegative = True
                        else:
                            currCol = self.shadeColDict["FP"]
                            textOut = "FP"
                    if actualLab[_nb] == 1:
                        if predLab[_nb] == 1:
                            currCol = self.shadeColDict["TP"]
                            textOut = "TP"
                        else:
                            currCol = self.shadeColDict["FN"]
                            textOut = "FN"
                    if not trueNegative:
                        _ax.axvspan(binStart, binEnd, alpha=0.5,\
                                 color=currCol)
                    for _ot in onsetTimeDict[_nb]:
                        _ax.axvline(x=_ot, color='r',\
                                 linestyle='--', linewidth=2)
                    if _nax == 0 :
                        textXLoc = eventDate + datetime.timedelta(\
                                minutes=(_nb+0.5)*self.binTimeRes) 
                        textYLoc = _ax.get_ylim()[1] - (\
                             abs(_ax.get_ylim()[1]) - _ax.get_ylim()[0] )/3
                        _ax.text(textXLoc, textYLoc, textOut)
        # get the figure name
        figName = "onset_binRes_" + str(self.binTimeRes) + \
                    "_nbins_" + str(self.nBins) + "_stack_event_" +\
                    eventDate.strftime("%Y%m%d-%H%M") + "." + figType
        # Labeling and formatting
        plt.xlim([plotTimeRange[0], plotTimeRange[1]])
        plt.xlabel("Time UT")
        plt.tick_params(labelsize=14)
        fig.suptitle(eventDate.strftime("%Y-%m-%d"))
        fig.savefig(self.figDir + figName, bbox_inches='tight')           
        fig.clf()
        plt.close()

