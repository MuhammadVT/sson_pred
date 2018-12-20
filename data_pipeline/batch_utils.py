import omn_utils
import create_onset_data
import pandas
import collections

class DataUtils(object):
    
    """
    Instead of caclulating batch dates
    and corresponding data points
    on the fly, we'll pre-calculate them
    and store them in a dict
    """
    
    def __init__(self, omnDBDir, omnDbName, omnTabName, omnTrain, \
            omnNormParamFile, imfNormalize=True, omnDBRes=1,\
             omnTrainParams = [ "By", "Bz", "Bx", "Vx", "Np" ],\
             batch_size=64, loadPreComputedOnset=True,\
             onsetDataloadFile="../data/binned_data.feather",\
             northData=True, southData=False, polarData=True,\
             imageData=True, polarFile="../data/polar_data.feather",\
             imageFile="../data/image_data.feather", onsetDelTCutoff=2,\
             onsetFillTimeRes=1, binTimeRes=30, nBins=3,\
            saveBinData=True, onsetSaveFile="../data/binned_data.feather",\
            shuffleData=True):
        """
        set up the parameters
        NOTE shuffleBatchDates shuffles the data points
        remove it if you dont want to use it!
        """
        self.loadPreComputedOnset = loadPreComputedOnset
        self.onsetDataloadFile = onsetDataloadFile
        self.batch_size = batch_size
        self.omnDBDir = omnDBDir
        self.omnDbName = omnDbName
        self.omnTabName = omnTabName
        self.omnTrain = omnTrain
        self.omnNormParamFile = omnNormParamFile
        self.imfNormalize = imfNormalize
        self.omnDBRes = omnDBRes
        self.omnTrainParams = omnTrainParams
        self.onsetDF = self._load_onset_data( northData, southData,\
              polarData, imageData, polarFile, imageFile, onsetDelTCutoff,\
              onsetFillTimeRes, binTimeRes, nBins, saveBinData, onsetSaveFile)
        self.batchDict = self._get_batchDict(shuffleData)
        # self.omnDF = self._load_omn_data()

    def _load_onset_data(self, northData, southData,\
              polarData, imageData, polarFile, imageFile, onsetDelTCutoff,\
              onsetFillTimeRes, binTimeRes, nBins, saveBinData, onsetSaveFile):
        """
        Load onset datadf, either the precomputed file
        or calculate it on the fly!
        """
        if self.loadPreComputedOnset:
            print("loading precomputed onset data")
            onsetDF = pandas.read_feather(self.onsetDataloadFile)
            # Note we need to reset the date column as index
            onsetDF = onsetDF.set_index( onsetDF["date"] )
            # drop the date column
            onsetDF.drop(columns=["date"], inplace=True)
        else:
            print("creating onset data")
            dataObj = create_onset_data.OnsetData(\
                        northData=northData, southData=southData,\
                         polarData=polarData, imageData=imageData,\
                         polarFile=polarFile, imageFile=imageFile,\
                         delTCutoff=onsetDelTCutoff,\
                         fillTimeRes=onsetFillTimeRes)
            onsetDF = dataObj.create_output_bins(\
                        binTimeRes=binTimeRes, nBins=nBins,\
                        saveBinData=saveBinData, saveFile=onsetSaveFile)
            # drop the date column which is already in index
            onsetDF.drop(columns=["date"], inplace=True)
        return onsetDF

    def _load_omn_data(self):
        """
        Load omn data
        """
        # get the time range from onset data
        omnStartDate = self.onsetDF.index.min()
        omnEndDate = self.onsetDF.index.max()
        # create the obj and load data
        omnObj = omn_utils.OmnData(omnStartDate, omnEndDate, self.omnDBDir,\
                           self.omnDbName, self.omnTabName,\
                           self.omnTrain, self.omnNormParamFile,\
                            imf_normalize=self.imfNormalize,\
                            db_time_resolution=self.omnDBRes,\
                            omn_train_params = self.omnTrainParams)
        return omnObj.omnDF

    def _get_batchDict(self, shuffleData):
        """
        create a dict with batch dates as keys
        and corresponding datapoint date list as
        values.
        """
        import numpy
        # get the data points/dates
        dataDateList = numpy.array( self.onsetDF.index.tolist() )
        # shuffle if we choose to
        if shuffleData:
            numpy.random.shuffle(dataDateList)
        # divide the dates by batch size and create a dict of batches
        nArrs = round(dataDateList.shape[0]/float(self.batch_size))
        dataDateList = numpy.array_split(dataDateList, nArrs)
        batchDict = {}
        for _nbat, _bat in enumerate(dataDateList):
            batchDict[_nbat] = _bat
        return batchDict

