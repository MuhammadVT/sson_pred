import datetime
import pandas
import numpy
import feather
import event_plot
import matplotlib
matplotlib.use('Agg')

class PredSumry(object):
    """
    A class to take an individual SS onset event
    and plot different geophysical parameters within
    a certain time range of the event (Bz, Vx, AE, SML)
    and compare the difference between predicted and 
    actual results. We'll also be able to have a sanity
    check during these events.
    """
    def __init__(self, modelPredFname, binTimeRes=30, nBins=2,\
                    northData=True, southData=False,\
                    polarFile="../data/polar_data.feather",\
                    imageFile="../data/image_data.feather"):
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
        self.ssOnsetDF = self._load_onset_data(northData,\
                                     southData, polarFile, imageFile)
        print("loaded onset data from polar/image")


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

    def _load_onset_data(self, northData, southData, polarFile, imageFile):
        """
        Load actual onset data from POLAR UVI and IMAGE satellites
        """
        polarDF = feather.read_dataframe(polarFile)#pandas.read_feather(polarFile)
        imageDF = feather.read_dataframe(imageFile)#pandas.read_feather(imageFile)
        # hemispheres to use!
        if (not northData) & (not southData):
            print("No hemi chosen! choosing north")
            northData = True
        # POLAR data
        # if only northern hemi is used
        if northData & (not southData):
            polarDF = polarDF[ polarDF["mlat"] >=0\
                             ].reset_index(drop=True)
        # if only southern hemi is used
        if (not northData) & southData:
            polarDF = polarDF[ polarDF["mlat"] <=0\
                             ].reset_index(drop=True)
        # IMAGE data
        # if only northern hemi is used
        if northData & (not southData):
            imageDF = imageDF[ imageDF["mlat"] >=0\
                             ].reset_index(drop=True)
        # if only southern hemi is used
        if (not northData) & southData:
            imageDF = imageDF[ imageDF["mlat"] <=0\
                             ].reset_index(drop=True)
        # Now merge both the dataframes!
        ssOnsetDF = pandas.concat( [ polarDF, imageDF ] )
        return ssOnsetDF
    
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
             binPlotType=False,plotTimeHist=120, plotFutureBins=2,\
             omnParams = ["By", "Bz", "Bx", "Vx", "Np"], \
             aulParams = ["au", "al"],smParams=["au", "al"],\
             binTimeRes=30, nBins=2, paramTimeRange=None,\
             figDir="/home/bharat/Documents/data/ss_onset_dataset/onset_plots/"):
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
        if binPlotType:
            print("generating binned plots with shading")
            for index, row in self.predDF.iterrows():
                _actBinLabs = row[actBinCols].tolist()
                _prBinLabs = row[predBinCols].tolist()
                _dt = row["date"]
                print "plotting date--->", _dt
                esObj.generate_bin_plot(_dt, _actBinLabs, _prBinLabs)
        else:
            # set the date axis as index
            self.ssOnsetDF = self.ssOnsetDF.set_index(\
                        pandas.to_datetime(self.ssOnsetDF["date"]))
            # Loop through the rows of predDF and get onset times
            # and then generate the plots
            for ind, row in self.predDF.iterrows():
                _actBinLabs = row[actBinCols].tolist()
                _prBinLabs = row[predBinCols].tolist()
                _dt = row["date"]
                print "_dt",_dt
                # Now for each bin check if there is a corresponding onset
                onsetTimeDict = {}
                for _n in range(nBins):
                    _cOnsetDts = self.ssOnsetDF.loc[ _dt +\
                        datetime.timedelta(minutes=binTimeRes*(_n)) : _dt +\
                         datetime.timedelta(minutes=binTimeRes*(_n+1))\
                            ].index.tolist()
                    if (_actBinLabs == 0) & (len(_cOnsetDts) >0):
                        print "SOMETHING VERY WRONG, FOUND" +\
                                  " A LABEL FOR NON-SS PERIOD--->",\
                                   _dt, _cOnsetDts, _actBinLabs[_n]
                    onsetTimeDict[_n] = _cOnsetDts
                print "plotting date--->", _dt
                esObj.generate_onset_plot(_dt, _actBinLabs, _prBinLabs, onsetTimeDict)
            # first get the onset times by merging predDF and polar/imageDF
            print("generating prediction bins and actual onsets")