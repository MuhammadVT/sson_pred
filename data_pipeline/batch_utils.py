import omn_utils
import create_onset_data
import pandas
import datetime
import feather

class DataUtils(object):
    
    """
    Instead of caclulating batch dates
    and corresponding data points
    on the fly, we'll pre-calculate them
    and store them in a dict
    """
    
    def __init__(self, omnDBDir, omnDbName, omnTabName, omnTrain, \
            omnNormParamFile, useSML=False, imfNormalize=True, omnDBRes=1,\
             omnTrainParams = [ "By", "Bz", "Bx", "Vx", "Np" ],\
             batch_size=64, loadPreComputedOnset=True,\
             onsetDataloadFile="../data/binned_data.feather",\
             northData=True, southData=False, polarData=True,\
             imageData=True, polarFile="../data/polar_data.feather",\
             imageFile="../data/image_data.feather", onsetDelTCutoff=2,\
             onsetFillTimeRes=1, binTimeRes=30, nBins=2,\
            saveBinData=False, onsetSaveFile="../data/binned_data.feather",\
            shuffleData=True, omnHistory=120,\
            smlDateRange=[datetime.datetime(1997,1,1),datetime.datetime(2000,1,1)]):
        """
        set up the parameters
        NOTE shuffleBatchDates shuffles the data points
        remove it if you dont want to use it!
        omnHistory here is the minutes in history 
        you want to load omn data.
        """
        self.loadPreComputedOnset = loadPreComputedOnset
        self.onsetDataloadFile = onsetDataloadFile
        self.batch_size = batch_size
        self.omnDBDir = omnDBDir
        self.omnDbName = omnDbName
        self.omnTabName = omnTabName
        self.omnTrain = omnTrain
        self.omnNormParamFile = omnNormParamFile
        self.useSML = useSML
        self.smlDateRange = smlDateRange
        self.imfNormalize = imfNormalize
        self.omnDBRes = omnDBRes
        self.omnTrainParams = omnTrainParams
        self.omnHistory = omnHistory
        self.onsetDF = self._load_onset_data( northData, southData,\
              polarData, imageData, polarFile, imageFile, onsetDelTCutoff,\
              onsetFillTimeRes, binTimeRes, nBins, saveBinData, onsetSaveFile)
        self.batchDict = self._get_batchDict(shuffleData)
        self.omnDF = self._load_omn_data()

    def _load_onset_data(self, northData, southData,\
              polarData, imageData, polarFile, imageFile, onsetDelTCutoff,\
              onsetFillTimeRes, binTimeRes, nBins, saveBinData, onsetSaveFile,\
              trnValTestSplitData=True, trnSplit=0.75, valSplit=0.15):
        """
        Load onset datadf, either the precomputed file
        or calculate it on the fly!
        """
        if self.loadPreComputedOnset:
            print("loading precomputed onset data")
            onsetDF = feather.read_dataframe(self.onsetDataloadFile)
            # Note we need to reset the date column as index
            onsetDF = onsetDF.set_index( onsetDF["date"] )
            # drop the date column
            onsetDF.drop(columns=["date"], inplace=True)
        else:
            print("creating onset data")
            dataObj = create_onset_data.OnsetData(useSML=self.useSML,\
                    northData=northData, southData=southData,\
                     polarData=polarData, imageData=imageData,\
                     polarFile=polarFile, imageFile=imageFile,\
                     binTimeRes=binTimeRes, nBins=nBins, \
                     delTCutoff=onsetDelTCutoff, fillTimeRes=onsetFillTimeRes,\
                     trnValTestSplitData=trnValTestSplitData,\
                     trnSplit=trnSplit, valSplit=valSplit)
            if self.useSML:
                onsetDF = dataObj.create_sml_bins(smlDateRange=self.smlDateRange,\
                                    saveBinData=saveBinData, saveFile=onsetSaveFile)
            else:
                onsetDF = dataObj.create_output_bins(\
                            saveBinData=saveBinData, saveFile=onsetSaveFile)
            # drop the date column which is already in index
        return onsetDF

    def _load_omn_data(self):
        """
        Load omn data
        """
        # get the time range from onset data
        omnStartDate = self.onsetDF.index.min() - datetime.timedelta(\
                                                minutes=self.omnHistory)
        omnEndDate = self.onsetDF.index.max()
        # create the obj and load data
        omnObj = omn_utils.OmnData(omnStartDate, omnEndDate, self.omnDBDir,\
                           self.omnDbName, self.omnTabName,\
                           self.omnTrain, self.omnNormParamFile,\
                            imf_normalize=self.imfNormalize,\
                            db_time_resolution=self.omnDBRes,\
                            omn_train_params = self.omnTrainParams)
        # set the datetime as index since we are working off of it
        omnObj.omnDF = omnObj.omnDF.set_index(omnObj.omnDF["datetime"])
        omnObj.omnDF = omnObj.omnDF[self.omnTrainParams]
        return omnObj.omnDF

    def _get_batchDict(self, shuffleData, set_seed=0):
        """
        create a dict with batch dates as keys
        and corresponding datapoint date list as
        values.
        Note we're seeding the values to reproduce
        shuffled results!
        assign the set_seed keyword to None, if you
        dont want to "de-randomize" the shuffling!
        """
        import numpy
        # get the data points/dates
        dataDateList = numpy.array( self.onsetDF.index.tolist() )
        # shuffle if we choose to
        if shuffleData:
            if set_seed is not None:
                numpy.random.seed(set_seed)
            numpy.random.shuffle(dataDateList)
        # divide the dates by batch size and create a dict of batches
        nArrs = round(dataDateList.shape[0]/float(self.batch_size))
        dataDateList = numpy.array_split(dataDateList, nArrs)
        batchDict = {}
        for _nbat, _bat in enumerate(dataDateList):
            batchDict[_nbat] = _bat
        return batchDict
    
    def onset_from_batch(self, dateList, predList=["bin"]):
        """
        Given a list of dates from one batch
        get outputs from the onsetDF
        predList contains the type of outputs we need
        during predictions. we generate multiple params
        and we may not need all of them during training.
        """
        # Note our dateList could be shuffled
        # so we can't simply use a range for 
        # accesing data from the index!
        predCols = []
        for _pr in predList:
            predCols += [ col for col in self.onsetDF\
                         if col.startswith(_pr) ]
        outArr = self.onsetDF[\
                    self.onsetDF.index.isin(dateList)\
                    ][predCols].as_matrix()
        return outArr.reshape( outArr.shape[0], 1, outArr.shape[1] )
    
    def omn_from_batch(self, dateList):
        """
        Given a list of dates from one batch
        get omn data hist for each data point.
        """
        # Note our dateList could be shuffled
        # so we can't simply use a range for 
        # accesing data from the index!
        import numpy
        omnBatchMatrix = []
        for _cd in dateList:
            _st = _cd.strftime("%Y-%m-%d %H:%M:%S")
            _et = (_cd - datetime.timedelta(\
                    minutes=self.omnHistory) ).strftime(\
                    "%Y-%m-%d %H:%M:%S")
            omnBatchMatrix.append(\
                self.omnDF.loc[ _et : _st ].as_matrix())
        return numpy.array(omnBatchMatrix)
