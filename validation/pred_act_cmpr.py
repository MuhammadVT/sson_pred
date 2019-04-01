import warnings
warnings.filterwarnings('ignore')
import datetime
import pandas
import numpy
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
    def __init__(self, modelPredFname, useSML=True, binTimeRes=30, nBins=2,\
                    northData=True, southData=False,\
                    polarFile="../data/polar_data.feather",\
                    imageFile="../data/image_data.feather",\
                    smlFname="../data/filtered-20190103-22-53-substorms.csv",\
                    smlDateRange=None):
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
                            modelPredFname)
        print("loaded model predicted data")
        if not useSML:
            self.ssOnsetDF = self._load_onset_img_pol_data(northData,\
                                         southData, polarFile, imageFile)
            print("loaded onset data from polar/image")
        else:
            self.smlFname = smlFname
            # Read data from the files
            self.ssOnsetDF = pandas.read_csv(self.smlFname,\
                                parse_dates=["Date_UTC"])
            # rename the cols
            self.ssOnsetDF.columns = [ "date", "mlat", "mlt" ]
            # Use the given date range to limit onset predictions
            if smlDateRange is not None:
                self.ssOnsetDF = self.ssOnsetDF[\
                        (self.ssOnsetDF["date"] >= smlDateRange[0]) &\
                        (self.ssOnsetDF["date"] <= smlDateRange[1]) ]
            
            print("loaded onset data from SML")


    def _load_model_pred_data(self, fileName):
        """
        Load the model predictions into a DF
        """
        # we need to carefully setup the column names
        colNames = ["date"]
        for _nb in range(self.nBins):
            colNames += [ "bin_" + str(_nb) ]
        #[ "label", "pred_label" ]
        colNames += ["label", "del_minutes","pred_label"]
        for _nb in range(self.nBins):
            # there are 2 probs for each bin
            # one zero prob and other 1 prob
            colNames += [ "prob_type_0_b_" + str(_nb) ]
            colNames += [ "prob_type_1_b_" + str(_nb) ]
        predDF = pandas.read_csv(fileName, names=colNames,\
                     header=0, parse_dates=["date"])
        # get the columns showing the bin predictions
        filterCols = [ col for col in predDF\
                     if col.startswith('bin') ]
        predDF = predDF.apply( self.pred_bin_out,\
                              axis=1 )
        return predDF

    def _load_onset_img_pol_data(self, northData, southData, polarFile, imageFile):
        """
        Load actual onset data from POLAR UVI and IMAGE satellites
        """
        import feather
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
             paramTimeRange=None,\
             figDir="/home/bharat/Documents/data/ss_onset_dataset/sml_onset_plots/"):
        """
        Loop through each of the events in the prediction
        files and make plots for each of the event dates.
        """
        actBinCols = sorted([ col for col in self.predDF\
                         if col.startswith('bin') ])
        predBinCols = sorted([ col for col in self.predDF\
                                 if col.startswith('pbin') ])
        probPredLabs = sorted([ col for col in self.predDF\
                                 if col.startswith('prob_') ])
        # get the time range to load the databases and initialize
        # the plot class!
        predTimeRange = [ self.predDF["date"].min(),\
                         self.predDF["date"].max() ]
        esObj = event_plot.EventSummary(predTimeRange,\
                paramDBDir, omnDbName, omnTabName, aulDbName,\
                aulTabName, smlDbName, smlTabName, nBins=self.nBins,\
                binTimeRes=self.binTimeRes, figDir=figDir)
        if binPlotType:
            print("generating binned plots with shading")
            for index, row in self.predDF.iterrows():
                _actBinLabs = row[actBinCols].tolist()
                _prBinLabs = row[predBinCols].tolist()
                _dt = row["date"]
                print("plotting date--->", _dt)
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
                _probLabs = row[probPredLabs].to_dict()
                _dt = row["date"]
                # Now for each bin check if there is a corresponding onset
                onsetTimeDict = {}
                for _n in range(self.nBins):
                    _cOnsetDts = self.ssOnsetDF.loc[ _dt +\
                        datetime.timedelta(minutes=self.binTimeRes*(_n)) : _dt +\
                         datetime.timedelta(minutes=self.binTimeRes*(_n+1))\
                            ].index.tolist()
                    if (_actBinLabs == 0) & (len(_cOnsetDts) >0):
                        print("SOMETHING VERY WRONG, FOUND" +\
                                  " A LABEL FOR NON-SS PERIOD--->",\
                                   _dt, _cOnsetDts, _actBinLabs[_n])
                    onsetTimeDict[_n] = _cOnsetDts
                print("plotting date--->", _dt)
                esObj.generate_onset_plot(_dt, _actBinLabs, _prBinLabs,\
                                         onsetTimeDict, predLabProb=_probLabs)
            # first get the onset times by merging predDF and polar/imageDF
            print("generating prediction bins and actual onsets")
