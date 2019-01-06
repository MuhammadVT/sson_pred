import datetime
import pandas
import numpy
import event_plot

class PredSumry(object):
    """
    A class to take an individual SS onset event
    and plot different geophysical parameters within
    a certain time range of the event (Bz, Vx, AE, SML)
    and compare the difference between predicted and 
    actual results. We'll also be able to have a sanity
    check during these events.
    """
    def __init__(self, modelPredFname, binTimeRes=30, nBins=2):
        """
        modelPredFname : file name to read the prediction results from.
                        NOTE : we expect the actual labels to be present
                        in this file as well.
        """
        self.nBins = nBins
        self.binTimeRes = binTimeRes
        self.modelPredFname = modelPredFname
        # load the model pred data labels
        self.predDF = self._load_model_pred_data(\
                            modelPredFname, nBins)
        print("loaded model predicted data")

    def _load_model_pred_data(self, fileName, nBins):
        """
        Load the model predictions into a DF
        """
        # we need to carefully setup the column names
        colNames = ["date"]
        for _nb in range(nBins):
            colNames += [ "bin_" + str(_nb) ]
        colNames += [ "label", "pred_label" ] 
        predDF = pandas.read_csv(fileName, names=colNames,\
                     header=0, parse_dates=["date"])
        # get the columns showing the bin predictions
        filterCols = [ col for col in predDF\
                     if col.startswith('bin') ]
        predDF = predDF.apply( self.pred_bin_out,\
                              axis=1 )
        return predDF
    
    def pred_bin_out(self, row):
        """
        Given the prediction label, get the actual
        output in bins by converting the label into
        binary representation. For ex, label 2 would
        convert to 10 and 5 to 101 and so on.
        """
        # Note we need the binary format to be consistent
        # with the actual labels, i.e., it depends on the 
        # number of bins. For example, 2 could be 10 or 010.
        binFormtStr = '{0:0' + str(self.nBins) + 'b}'
        predBinStr = binFormtStr.format(row["pred_label"])
        # Now add these into different pred bins
        for _n, _pb in enumerate(predBinStr):
            row["pbin_" + str(_n)] = int(_pb)
        return row
    
    def create_pred_plots(self, paramDBDir, omnDbName, \
             omnTabName, aulDbName, aulTabName, smlDbName, smlTabName,\
             plotTimeHist=120, plotFutureBins=2,\
             omnParams = ["By", "Bz", "Bx", "Vx", "Np"], \
             aulParams = ["au", "al"],smParams=["au", "al"],\
             binTimeRes=30, nBins=2, paramTimeRange=None,\
             figDir="/home/bharat/Documents/data/ss_onset_dataset/plots/"):
        """
        Loop through each of the events in the prediction
        files and make plots for each of the event dates.
        """
        actBinCols = [ col for col in self.predDF\
                         if col.startswith('bin') ]
        predBinCols = [ col for col in self.predDF\
                                 if col.startswith('pbin') ]
        # get the time range to load the databases and initialize
        # the plot class!
        predTimeRange = [ self.predDF["date"].min(),\
                         self.predDF["date"].max() ]
        esObj = event_plot.EventSummary(predTimeRange,\
                paramDBDir, omnDbName, omnTabName, aulDbName,\
                aulTabName, smlDbName, smlTabName, nBins=2)
        for index, row in self.predDF.iterrows():
            _actBinLabs = row[actBinCols].tolist()
            _prBinLabs = row[predBinCols].tolist()
            _dt = row["date"]
            print "plotting--->", _dt
            esObj.generate_plot(_dt, _actBinLabs, _prBinLabs)
            break
        
        
        