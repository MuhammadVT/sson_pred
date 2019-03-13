import pandas
import datetime
# import feather
import numpy
from sklearn.utils import resample

class OnsetData(object):
    """
    Load the required data into a DF
    """
    def __init__(self, useSML=False, northData=True, southData=False,\
                 polarData=True, imageData=True, \
                 polarFile="../data/polar_data.feather",\
                 imageFile="../data/image_data.feather",delTCutoff=2,\
                 fillTimeRes=1, binTimeRes=60, nBins=1, \
                 trnValTestSplitData=False, trnSplit=0.75, valSplit=0.15,\
                 smlFname="../data/filtered-20190103-22-53-substorms.csv",smlDateRange=None,\
                 smlDownsample=True, smlUpsample=False, dwnSmplByUT=True):
        """
        setup some vars and load preliminary data.
        Most of the variables are self explanatory.
        delTCutoff is the time range between two consecutive onset
        events that we want to use as cutoff. In other words we'll
        interpolate between these events if time is < self.delTCutoff,
        else we jump to the next event.
        fillTimeRes is the interpolation time resolution
        smlDateRange indicates the date range to create SS bins for,
        if set to None it will create bins for the entire range.
        """
        self.delTCutoff = delTCutoff
        self.fillTimeRes = fillTimeRes
        self.binTimeRes = binTimeRes
        self.nBins = nBins
        # set params for shuffling
        self.trnValTestSplitData = trnValTestSplitData
        self.trnSplit = trnSplit
        self.valSplit = valSplit
        self.dwnSmplByUT = dwnSmplByUT
        if not useSML:
            self.polarDF = None
            self.imageDF = None
            if polarData:
                self.polarDF = feather.read_dataframe(polarFile)#pandas.read_feather(polarFile)
            if imageData:
                self.imageDF = feather.read_dataframe(imageFile)#pandas.read_feather(imageFile)
            # hemispheres to use!
            if (not northData) & (not southData):
                print("No hemi chosen! choosing north")
                northData = True
            self.northData = northData
            self.southData = southData
            # Now filter the data
            self._filter_data()
            # we'll have some arrays to create datelist which is our
            # training/testing dataset
            self.polarDateList = []
            self.imageDateList = []
            # Now populate the data
            self._populate_date_list()
        else:
            self.smlFname = smlFname
            self.smlDateRange = smlDateRange
            self.smlDownsample = smlDownsample
            self.smlUpsample = smlUpsample

    def _filter_data(self):
        """
        Filter and clean the required ss onset data
        """
        # work through polar data
        if self.polarDF is not None:
            # Calculate the time diff between two consecutive SS onsets
            polarDelTime = self.polarDF["date"].diff()
            # convert the difference to hours
            polarDelTime = polarDelTime.apply(\
                            lambda x: x.total_seconds()/3600. )
            self.polarDF["delT"] = polarDelTime
            # if only northern hemi is used
            if self.northData & (not self.southData):
                self.polarDF = self.polarDF[ self.polarDF["mlat"] >=0\
                                 ].reset_index(drop=True)
            # if only southern hemi is used
            if (not self.northData) & self.southData:
                self.polarDF = self.polarDF[ self.polarDF["mlat"] <=0\
                                 ].reset_index(drop=True)
        # work through image data
        if self.imageDF is not None:
            # Calculate the time diff between two consecutive SS onsets
            imageDelTime = self.imageDF["date"].diff()
            imageDelTime = imageDelTime.apply(\
                            lambda x: x.total_seconds()/3600. )
            self.imageDF["delT"] = imageDelTime
            # if only northern hemi is used
            if self.northData & (not self.southData):
                self.imageDF = self.imageDF[ self.imageDF["mlat"] >=0\
                                 ].reset_index(drop=True)
            # if only southern hemi is used
            if (not self.northData) & self.southData:
                self.imageDF = self.imageDF[ self.imageDF["mlat"] <=0\
                                 ].reset_index(drop=True)

    def _populate_date_list(self, delStartTime=30):
        """
        We'll create a list of dates that will be used 
        for training and testing. Basically we'll create
        our core date list here. We'll take the onset times
        and expand that list. If two recurring onsets are 
        less than a threshold value (say 2 hours) then we'll
        fill in the dates between them at a selected time resolution.
        For example, if one onset is at 0400 UT and the other at 0430 UT
        we'll expand our self.polarDateList such that we have multiple datapoints
        between 0400 and 0430 i.e., 0400, 0401,....0430. The outputs
        would be bins based on some value (say 30 min) so that we check
        if there is a SS onset in that bin.
        """
        # work through polar data
        if self.polarDF is not None:
            for row in self.polarDF.iterrows():
                if row[1]["delT"] <= self.delTCutoff:
                    currIndex = row[0]
                    # get start time
                    startTime = datetime.datetime( \
                                self.polarDF.iloc[currIndex-1]["date"].year,\
                                 self.polarDF.iloc[currIndex-1]["date"].month,\
                                 self.polarDF.iloc[currIndex-1]["date"].day,\
                                 self.polarDF.iloc[currIndex-1]["date"].hour,\
                                 self.polarDF.iloc[currIndex-1]["date"].minute)
                    # we can get some additional 1's in the data by 
                    # preceeding the start time. i.e., subtracting a 
                    # few minutes from the start time!
                    startTime = startTime - datetime.timedelta(minutes=delStartTime)
                    # get end time
                    endTime = datetime.datetime( row[1]["date"].year,\
                                                 row[1]["date"].month,\
                                                 row[1]["date"].day,\
                                                 row[1]["date"].hour,\
                                                 row[1]["date"].minute)
                    # Now while creating the bins we need to be careful
                    # about the end times, since we are making predictions
                    # over the next few minutes/hours we need to subtract
                    # certain time from the end time. 
                    if self.nBins > 1:
                        endTime = endTime - datetime.timedelta(\
                                 minutes=(self.nBins-1) * self.binTimeRes )
                    else:
                        endTime = endTime - datetime.timedelta(minutes=self.binTimeRes/2)
                    iterTime = startTime
                    while iterTime <= endTime:
                        self.polarDateList.append(iterTime)
                        iterTime += datetime.timedelta(seconds=self.fillTimeRes*60)
            # there will be some repeat values, discard them
            self.polarDateList = sorted(list(set(self.polarDateList)))
        # work through image data
        if self.imageDF is not None:
            for row in self.imageDF.iterrows():
                if row[1]["delT"] <= self.delTCutoff:
                    currIndex = row[0]
                    # get start time
                    startTime = datetime.datetime( \
                            self.imageDF.iloc[currIndex-1]["date"].year,\
                             self.imageDF.iloc[currIndex-1]["date"].month,\
                             self.imageDF.iloc[currIndex-1]["date"].day,\
                             self.imageDF.iloc[currIndex-1]["date"].hour,\
                             self.imageDF.iloc[currIndex-1]["date"].minute)
                    # we can get some additional 1's in the data by 
                    # preceeding the start time. i.e., subtracting a 
                    # few minutes from the start time!
                    startTime = startTime - datetime.timedelta(minutes=delStartTime)
                    # get end time
                    endTime = datetime.datetime( row[1]["date"].year,\
                                                 row[1]["date"].month,\
                                                 row[1]["date"].day,\
                                                 row[1]["date"].hour,\
                                                 row[1]["date"].minute)
                    # Now while creating the bins we need to be careful
                    # about the end times, since we are making predictions
                    # over the next few minutes/hours we need to subtract
                    # certain time from the end time. 
                    if self.nBins > 1:
                        endTime = endTime - datetime.timedelta(\
                                 minutes=(self.nBins-1) * self.binTimeRes )
                    else:
                        endTime = endTime - datetime.timedelta(minutes=self.binTimeRes/2)
                    iterTime = startTime
                    while iterTime <= endTime:
                        self.imageDateList.append(iterTime)
                        iterTime += datetime.timedelta(seconds=self.fillTimeRes*60)
            # there will be some repeat values, discard them
            self.imageDateList = sorted(list(set(self.imageDateList)))
            
    def onset_binary(self, row, filterColList):
        """
        For the onset DF we have 1/0 for each
        bin. To get an estimate of the counts, we'll
        convert the bins data into a binary out. For 
        example, 000-->0, 100-->4 and so on.
        """
        binOut = ""
        for _fc in filterColList[::-1]:
            binOut += str(int(row[_fc]))
        row["outBinary"] = int( binOut, 2 )
        return row
            
    def get_non_ss_intervals(self, startDate, endDate, aulDBdir, \
                 aulDBName, aulTabName, alSSCutoff = -10, \
                 aeSSCutoff = 50, minDelT = 5, minDiffTime = 180):
        """
        Get the list of dates and times with no substorm activity
        """
        import non_ss_dataset
        # first get the list of dates
        nonSSObj = non_ss_dataset.NonSSData(startDate, endDate, aulDBdir, \
                     aulDBName, aulTabName, alSSCutoff = alSSCutoff, \
                     aeSSCutoff = aeSSCutoff, minDelT = minDelT, \
                     minDiffTime = minDiffTime)
        nonSSDt = nonSSObj.get_non_ss_dates()
        return nonSSDt
    
    def create_non_ss_bins(self, ndd, nonSSDataCnt):
        """
        Given non ss dates and number of rows needed
        get the non ss df with bins
        """
        import random
        nonSSDtList = []
        for _d in ndd:
            _it = _d[0]
            # remember we have bins showing no activity for the next
            # few hours/minutes, so be careful selecting the days!
            while (_d[1] - _it).total_seconds()/(60) >= self.binTimeRes*(self.nBins+1):
                nonSSDtList.append(_it)
                _it += datetime.timedelta(seconds=self.fillTimeRes*60)
        if nonSSDataCnt < len(nonSSDtList):
            print("more data found than required---", nonSSDataCnt, len(nonSSDtList))
            # indices = random.sample(range(len(nonSSDtList)), nonSSDataCnt)
            # nonSSDtList = [ nonSSDtList[_i] for _i in sorted(indices) ]
            nonSSDtList = nonSSDtList[:nonSSDataCnt]
        return pandas.DataFrame(nonSSDtList, columns=["date"])
    
    def create_sml_bins(self, saveBinData=False,randomState=0,\
                 saveFile="../data/sml_binned_data_extra.csv"):
        """
        Create the bins based on SML index instead of
        POLAR UVI and IMAGE datasets. This is an alternative
        dataset.
        """
        # Read data from the files
        smlDF = pandas.read_csv(self.smlFname,\
                            parse_dates=["Date_UTC"])
        # rename the cols
        smlDF.columns = [ "datetime", "mlat", "mlt" ]
        # limit the DF range to
        if self.smlDateRange is None:
            smlDTStart = smlDF["datetime"].min()
            smlDTEnd = smlDF["datetime"].max()
        else:
            smlDTStart = self.smlDateRange[0]
            smlDTEnd = self.smlDateRange[1]
            # Limit SML data to the range
            smlDF = smlDF[ (smlDF["datetime"] >= smlDTStart) &\
                            (smlDF["datetime"] <= smlDTEnd) \
                            ].reset_index(drop=True)
        # for quicker search set datetime as index
        smlDF.set_index( pandas.to_datetime(\
                        smlDF["datetime"]), inplace=True )
        smlDF = smlDF[ ["mlat", "mlt"] ]
        smlBinList = []
        smlMlatList = []
        smlMLTList = []
        smlClstOnsetTime = []
        smlClstDelT = []
        smlDateList = []
        smlMultiSS = []
        _cpDate = smlDTStart
        _printYear = _cpDate.year - 1 # for pritning purposes
        while _cpDate <= smlDTEnd:
            if _printYear < _cpDate.year:
                print("Working through year-->", _cpDate.year)
                _printYear = _cpDate.year
            # we'll start with 0's (no onset) for all the bins
            # then later fill the values based on onset times
            _tmpBinVlas = [ 0 for _tv in range(self.nBins) ]
            _tmpLatVlas = [ -1. for _tv in range(self.nBins) ]
            _tmpMLTVlas = [ -1. for _tv in range(self.nBins) ]
            # get a start time and end time for search
            srchStTime = _cpDate.strftime("%Y-%m-%d %H:%M:%S")
            srchETime = (_cpDate + datetime.timedelta(minutes=self.binTimeRes*self.nBins)\
                ).strftime("%Y-%m-%d %H:%M:%S")
            _cOnsetList = smlDF.loc[ srchStTime : srchETime ].index.tolist()
            # get the difference between current time and the nearest
            # one's found in the DF
            # we'll ignore the time if the difference is less than 1 minute
            # initialize a variable for counting multiple ss onsets
            _multiSS = 0
            for _cto in sorted(_cOnsetList):
                _dt = (_cto - _cpDate).total_seconds()/60.
                for _nb in range(self.nBins):
                    if (_dt >= (_nb*self.binTimeRes) + 1) & (_dt <= (_nb+1)*self.binTimeRes):
                        # we need to find if there are multiple ss in any bin. 
                        # If yes, we'll count them and store in a seperate var
                        if _tmpBinVlas[_nb] == 1:
                            _multiSS += 1
                        _tmpBinVlas[_nb] = 1
                        _tmpLatVlas[_nb] = smlDF.loc[_cto]["mlat"]/90.
                        _tmpMLTVlas[_nb] = smlDF.loc[_cto]["mlt"]/(15*24.)
            smlBinList.append(_tmpBinVlas)
            smlMlatList.append(_tmpLatVlas)
            smlMLTList.append(_tmpMLTVlas)
            smlDateList.append(_cpDate)
            smlMultiSS.append(_multiSS)
            # we'll add another col where we find the shortest time
            # to ss onset
            if len(_cOnsetList) > 0:
                _minDelT = (min(_cOnsetList) - _cpDate).total_seconds()/60.
                # if diff is less than a minute, ignore
                if _minDelT <= 0.:
                    smlClstDelT.append(-1.)
                else:
                    smlClstDelT.append(_minDelT)
            else:
                smlClstDelT.append(-1.)
            # Next bin time
            _cpDate += datetime.timedelta(seconds=self.fillTimeRes*60)
        # convert to dataframe
        smlDataSet = pandas.DataFrame(smlBinList,\
                     columns=["bin_" + str(_i) for _i in range(self.nBins)])
        # add the additional mlat,mlt cols
        smlDataSet[["mlat_" + str(_i) for _i in range(self.nBins)\
                                   ]] = pandas.DataFrame(smlMlatList)
        smlDataSet[["mlt_" + str(_i) for _i in range(self.nBins)\
                                   ]] = pandas.DataFrame(smlMLTList)
        # set the closest time cols
        #         polDataSet["closest_time"] = smlClstOnsetTime
        smlDataSet["del_minutes"] = smlClstDelT
        smlDataSet["data_label"] = "S"
        smlDataSet["multi_ss"] = smlMultiSS

        smlDataSet = smlDataSet.set_index(\
                    pandas.to_datetime(smlDateList))
        # we'll select the cols we need
        binCols = [ col for col in smlDataSet\
                     if col.startswith('bin') ]
        mlatCols = [ col for col in smlDataSet\
                     if col.startswith('mlat') ]
        mltCols = [ col for col in smlDataSet\
                     if col.startswith('mlt') ]
        otrCols = [ "del_minutes", "data_label", "multi_ss" ]
        selCols = binCols + mlatCols + mltCols + otrCols
        smlDataSet = smlDataSet[selCols]
        # sort the index to make sure non-ss intervals
        # are not segregated at the end
        smlDataSet.sort_index(inplace=True)
        print(smlDataSet.head())
        # if both self.smlUpsample and self.smlDownsample
        # are set then choose one self.smlDownsample
        if self.smlUpsample & self.smlDownsample:
            self.smlUpsample = False
        filterCols = [ col for col in smlDataSet\
                         if col.startswith('bin') ]
        smlDataSet = smlDataSet.apply( self.onset_binary, axis=1,\
                         args=(filterCols,) )
        # Downsample the data if the option is set
        print("original DF label counts---->", smlDataSet["outBinary"].value_counts())
        if self.smlDownsample:
            if self.dwnSmplByUT:
                print("Downsampling the data into bins by UT hour")
                smlDataSet['hour'] = pandas.DatetimeIndex(\
                                        smlDataSet.index).hour
                uniqHours = smlDataSet["hour"].unique()
                smlDataSetList = []
                for _uut in uniqHours:
                    currUTSML = smlDataSet[ smlDataSet["hour"] == _uut ]
                    smLabs = currUTSML["outBinary"].value_counts()
                    if self.nBins == 1:
                        # if we just have one bin
                        # downsample the other classes
                        # and make the count equal to the min 
                        dfList = [ currUTSML[ currUTSML["outBinary"] == smLabs.idxmin() ] ]
                        for _ind in smLabs.index:
                            if _ind != smLabs.idxmin():
                                # Downsample majority class
                                dfMaj = currUTSML[ currUTSML["outBinary"] == _ind ]
                                dfList.append( resample(dfMaj, 
                                             replace=False, # sample without replacement
                                             n_samples=smLabs[smLabs.idxmin()], # match min class
                                             random_state=randomState) )# reproducible results
                    else:
                        # if not we'll just downsample the 0's and make them equal 
                        # to anyother bin (which has atleast one 1).
                        nonZeroMax = smLabs[ smLabs.index > 0 ].max()
                        zeroCnt = smLabs[smLabs.index == 0].values[0]
                        if zeroCnt > nonZeroMax:
                            dfList = [ currUTSML[ currUTSML["outBinary"] > 0 ] ]
                            dfZero = currUTSML[ currUTSML["outBinary"] == 0 ]
                            dfList.append( resample(dfZero, 
                                         replace=False, # sample without replacement
                                         n_samples=nonZeroMax, # to match minority class
                                         random_state=randomState) )# reproducible results
                    smlDataSetList.append( pandas.concat(dfList) )
                smlDataSet = pandas.concat( smlDataSetList )
            else:
                print("Downsampling the data without UT bins")
                smLabs = smlDataSet["outBinary"].value_counts()
                if self.nBins == 1:
                    # if we just have one bin
                    # downsample the other classes
                    # and make the count equal to the min 
                    dfList = [ smlDataSet[ smlDataSet["outBinary"] == smLabs.idxmin() ] ]
                    for _ind in smLabs.index:
                        if _ind != smLabs.idxmin():
                            # Downsample majority class
                            dfMaj = smlDataSet[ smlDataSet["outBinary"] == _ind ]
                            dfList.append( resample(dfMaj, 
                                         replace=False, # sample without replacement
                                         n_samples=smLabs[smLabs.idxmin()], # to match minority class
                                         random_state=randomState) )# reproducible results
                else:
                    # if not we'll just downsample the 0's and make them equal 
                    # to anyother bin (which has atleast one 1).
                    nonZeroMax = smLabs[ smLabs.index > 0 ].max()
                    zeroCnt = smLabs[smLabs.index == 0].values[0]
                    if zeroCnt > nonZeroMax:
                        dfList = [ smlDataSet[ smlDataSet["outBinary"] > 0 ] ]
                        dfZero = smlDataSet[ smlDataSet["outBinary"] == 0 ]
                        dfList.append( resample(dfZero, 
                                     replace=False, # sample without replacement
                                     n_samples=nonZeroMax, # to match minority class
                                     random_state=randomState) )# reproducible results
                # Combine minority class with downsampled majority class
                smlDataSet = pandas.concat(dfList)
        if self.smlUpsample:
            print("Upsampling the data")
            smLabs = smlDataSet["outBinary"].value_counts()
            # downsample the other classes
            # and make the count equal to the min 
            dfList = [ smlDataSet[ smlDataSet["outBinary"] == smLabs.idxmax() ] ]
            for _ind in smLabs.index:
                if _ind != smLabs.idxmax():
                    # Downsample majority class
                    dfMin = smlDataSet[ smlDataSet["outBinary"] == _ind ]
                    dfList.append( resample(dfMin, 
                                 replace=False, # sample without replacement
                                 n_samples=smLabs[smLabs.idxmax()], # to match minority class
                                 random_state=randomState) )# reproducible results
            # Combine minority class with downsampled majority class
            smlDataSet = pandas.concat(dfList)
        # Sort the df to shuffle it in a way!
        smlDataSet.sort_index(inplace=True)
        print("new DF label counts---->", smlDataSet["outBinary"].value_counts())
        # save the file to make future calc faster
        if saveBinData:
            # Note feather doesn't support datetime index
            # so we'll reset it and then convert back when
            # we read the file back!
            # sort by dates
            smlDataSet["date"] = smlDataSet.index
            smlDataSet.reset_index(drop=True).to_csv(saveFile)
        return smlDataSet

    def create_output_bins(self,\
                 aulDBdir="../data/sqlite3/", \
                 aulDBName="smu_sml_sme.sqlite",\
                 aulTabName="smusmlsme", alSSCutoff = -25, \
                 aeSSCutoff = 50, minDelT = 5, saveBinData=False,\
                 saveFile="../data/binned_data_extra.csv",\
                 getNonSSInt=False, noSSbinRatio=1):
        """
        For each of the dates in the polar and image lists
        create corresponding output bins!
        There are two parameters here binTimeRes and nBins.
        binTimeRes (in minutes) corresponds to time resolution
        of each bin and nBins is well, number of bins. For example,
        if binTimeRes is 30 min and nBins is 3, then the output bins
        would correspond to next 0-30 min, 30-60 min and 60-90 min.
        if getNonSSInt is True then get non-substorm intervals based on
        AL/AE indices.
        noSSbinRatio indicates what number of datapoints needed for non-SS
        intervals to make the ratio of non-ss intervals equal(or a fraction)
        to the bin with max data. For example, if bin where ss is in next
        30 min has max data then we'll make non-ss intervals equal to this
        count if ratio is 1.
        dropDefaulnonSS : We'll get some data points where all the bins would
        be 0's. But these may not be right as the satellites may not be in the 
        right location to capture SS. So we'll discard these datapoints
        """
        # index the date column of the DFs to make for quicker
        # data selection
        if self.polarDF is not None:
            self.polarDF = self.polarDF.set_index("date")
        if self.imageDF is not None:
            self.imageDF = self.imageDF.set_index("date")
        # Loop through each of the dates and check if
        # there are substorm onset values in the time range
        # to populate our output bins!
        polarBinList = []
        polarMlatList = []
        polarMLTList = []
        polarClstOnsetTime = []
        polarClstDelT = []
        for _cpDate in self.polarDateList:
            # we'll start with 0's (no onset) for all the bins
            # then later fill the values based on onset times
            _tmpBinVlas = [ 0 for _tv in range(self.nBins) ]
            _tmpLatVlas = [ -1. for _tv in range(self.nBins) ]
            _tmpMLTVlas = [ -1. for _tv in range(self.nBins) ]
            # get a start time and end time for search
            srchStTime = _cpDate.strftime("%Y-%m-%d %H:%M:%S")
            srchETime = (_cpDate + datetime.timedelta(minutes=self.binTimeRes*self.nBins)\
                ).strftime("%Y-%m-%d %H:%M:%S")
            _cOnsetList = self.polarDF.loc[ srchStTime : srchETime ].index.tolist()
            # get the difference between current time and the nearest
            # one's found in the DF
            # we'll ignore the time if the difference is less than 1 minute
            for _cto in sorted(_cOnsetList):
                _dt = (_cto - _cpDate).total_seconds()/60.
                for _nb in range(self.nBins):
                    if (_dt >= (_nb*self.binTimeRes) + 1) & (_dt <= (_nb+1)*self.binTimeRes):
                        _tmpBinVlas[_nb] = 1
                        _tmpLatVlas[_nb] = self.polarDF.loc[_cto]["mlat"]/90.
                        _tmpMLTVlas[_nb] = self.polarDF.loc[_cto]["mlt"]/(15*24.)
            polarBinList.append(_tmpBinVlas)
            polarMlatList.append(_tmpLatVlas)
            polarMLTList.append(_tmpMLTVlas)
            # we'll add another col where we find the shortest time
            # to ss onset
            if len(_cOnsetList) > 0:
                _minDelT = (min(_cOnsetList) - _cpDate).total_seconds()/60.
                # if diff is less than a minute, ignore
                if _minDelT <= 0.:
#                     polarClstOnsetTime.append(-1.)
                    polarClstDelT.append(-1.)
                else:
#                     polarClstOnsetTime.append(min(_cOnsetList))
                    polarClstDelT.append(_minDelT)
            else:
#                 polarClstOnsetTime.append(-1.)
                polarClstDelT.append(-1.)
            
        # repeat the same for image data
        imageBinList = []
        imageMlatList = []
        imageMLTList = []
        imageClstOnsetTime = []
        imageClstDelT = []
        for _cpDate in self.imageDateList:
            # we'll start with 0's (no onset) for all the bins
            # then later fill the values based on onset times
            # similarly we'll fill -1's for lats and lons and 
            # then populate them later
            _tmpBinVlas = [ 0 for _tv in range(self.nBins) ]
            _tmpLatVlas = [ -1. for _tv in range(self.nBins) ]
            _tmpMLTVlas = [ -1. for _tv in range(self.nBins) ]
            # get a start time and end time for search
            srchStTime = _cpDate.strftime("%Y-%m-%d %H:%M:%S")
            srchETime = (_cpDate + datetime.timedelta(minutes=self.binTimeRes*self.nBins)\
                ).strftime("%Y-%m-%d %H:%M:%S")
            _cOnsetList = self.imageDF.loc[ srchStTime : srchETime ].index.tolist()
            # get the difference between current time and the nearest
            # one's found in the DF
            # we'll ignore the time if the difference is less than 1 minute
            for _cto in sorted(_cOnsetList):
                _dt = (_cto - _cpDate).total_seconds()/60.
                for _nb in range(self.nBins):
                    if (_dt >= (_nb*self.binTimeRes) + 1) & (_dt <= (_nb+1)*self.binTimeRes):
                        _tmpBinVlas[_nb] = 1
                        _tmpLatVlas[_nb] = self.imageDF.loc[_cto]["mlat"]/90.
                        _tmpMLTVlas[_nb] = self.imageDF.loc[_cto]["mlt"]/24.
            imageBinList.append(_tmpBinVlas)
            imageMlatList.append(_tmpLatVlas)
            imageMLTList.append(_tmpMLTVlas)
            # get the closest time
            if len(_cOnsetList) > 0:
                _minDelT = (min(_cOnsetList) - _cpDate).total_seconds()/60.
                # if diff is less than a minute, ignore
                if _minDelT <= 0.:
#                     imageClstOnsetTime.append(-1.)
                    imageClstDelT.append(-1.)
                else:
#                     imageClstOnsetTime.append(min(_cOnsetList))
                    imageClstDelT.append(_minDelT)
            else:
#                 imageClstOnsetTime.append(-1.)
                imageClstDelT.append(-1.)
            
        # convert the bin lists into dataframes
        allDFList = []
        if len(self.polarDateList) > 0:
            # POLAR
            polDataSet = pandas.DataFrame(polarBinList,\
                         columns=["bin_" + str(_i) for _i in range(self.nBins)])
            # add the additional mlat,mlt cols
            polDataSet[["mlat_" + str(_i) for _i in range(self.nBins)\
                                       ]] = pandas.DataFrame(polarMlatList)
            polDataSet[["mlt_" + str(_i) for _i in range(self.nBins)\
                                       ]] = pandas.DataFrame(polarMLTList)
            # set the closest time cols
    #         polDataSet["closest_time"] = polarClstOnsetTime
            polDataSet["del_minutes"] = polarClstDelT
            polDataSet["data_label"] = "P"
            
            polDataSet = polDataSet.set_index(\
                        pandas.to_datetime(self.polarDateList))
            allDFList.append(polDataSet)
        # IMAGE
        if len(self.imageDateList) > 0:
            imgDataSet = pandas.DataFrame(imageBinList,\
                         columns=["bin_" + str(_i) for _i in range(self.nBins)])
            # add the additional mlat,mlt cols
            imgDataSet[["mlat_" + str(_i) for _i in range(self.nBins)]] = pandas.DataFrame(imageMlatList)
            imgDataSet[["mlt_" + str(_i) for _i in range(self.nBins)]] = pandas.DataFrame(imageMLTList)
            # set the closest time cols
    #         imgDataSet["closest_time"] = imageClstOnsetTime
            imgDataSet["del_minutes"] = imageClstDelT
            imgDataSet["data_label"] = "I"

            imgDataSet = imgDataSet.set_index(\
                            pandas.to_datetime(self.imageDateList))
            allDFList.append(imgDataSet)
        # Now merge both the dataframes!
        ssBinDF = pandas.concat( allDFList )
        # sort by index(dates here)
        ssBinDF.sort_index(inplace=True)
        # Now we'll get the non-ss times
        if getNonSSInt:
            _sDt = ssBinDF.index.min()
            _eDt = ssBinDF.index.max()
            # get binary rep of the bins
            filterCols = [ col for col in ssBinDF\
                         if col.startswith('bin') ]
            ssBinDF = ssBinDF.apply( self.onset_binary, axis=1,\
                             args=(filterCols,) )
            # Now get value counts for the binary 
            valCntDict = ssBinDF["outBinary"].value_counts().to_dict()
            print("data distribution---->", valCntDict)
            maxNonZeroCnt = int(max({_k: _v for _k, _v in\
                          valCntDict.items() if _k > 0}.values()))
            zeroCnt = valCntDict[0]
            # get the number of data points needed for non-ss intervals
            nonSSDataCnt = int( (maxNonZeroCnt-zeroCnt) * noSSbinRatio)
            # Now nonSSDataCnt could some times be higher!
            # in such a case skip the rest of hte process
            if nonSSDataCnt > 0:
                # we'll set minDiffTime to be slightly higher than
                # bin time resolution * number of bins. This way
                # we make sure there is a good set of 0's
                minDiffTime = self.binTimeRes*(self.nBins+1)
                ndd = self.get_non_ss_intervals(_sDt, _eDt, aulDBdir, \
                     aulDBName, aulTabName, alSSCutoff = alSSCutoff, \
                     aeSSCutoff = aeSSCutoff, minDelT = minDelT,\
                     minDiffTime = minDiffTime)
                # get the nonSSDF from the dates
                nonSSDF = self.create_non_ss_bins(ndd, nonSSDataCnt)
                # add 0's to the bins
                for _cc in ssBinDF.columns:
                    if "bin" in _cc:
                        nonSSDF[_cc] = 0
                    else:
                        nonSSDF[_cc] = -1.
                # make both the DFs similar
                nonSSDF["data_label"] = "N"
                nonSSDF = nonSSDF.set_index(\
                                pandas.to_datetime(nonSSDF["date"].tolist()))
                # we'll select the cols we need
                binCols = [ col for col in ssBinDF\
                             if col.startswith('bin') ]
                mlatCols = [ col for col in ssBinDF\
                             if col.startswith('mlat') ]
                mltCols = [ col for col in ssBinDF\
                             if col.startswith('mlt') ]
                otrCols = [ "del_minutes", "data_label" ]
                selCols = binCols + mlatCols + mltCols + otrCols
                nonSSDF = nonSSDF[selCols]
                ssBinDF = ssBinDF[selCols]
                # there could be some common dates between SS
                # and non-SS DFs. For now we'll drop all the dates
                # which are present in SS from the non SS DF. But we'll
                # work on a better method in the future
                ssInds = set(ssBinDF.index.tolist())
                nssInds = set(nonSSDF.index.tolist())
                interInds = list(ssInds.intersection(nssInds))
                if len(interInds) > 0:
                    print("There are common dates found in both DFs",\
                            len(interInds))
                    # drop the common dates from non SS DF
                    nonSSDF.drop(interInds, inplace=True)
                    print("dropped common rows")
                else:
                    print("no common rows found between ss and non-ss data")
                # Merge Both the DFs
                origSize = ssBinDF.shape[0]
                ssBinDF = pandas.concat( [ ssBinDF, nonSSDF ] ) 
                newSize = ssBinDF.shape[0]
                print("expanded with new non-SS data--->", origSize, newSize)
            else:
                # we'll select the cols we need
                binCols = [ col for col in ssBinDF\
                             if col.startswith('bin') ]
                mlatCols = [ col for col in ssBinDF\
                             if col.startswith('mlat') ]
                mltCols = [ col for col in ssBinDF\
                             if col.startswith('mlt') ]
                otrCols = [ "del_minutes", "data_label" ]
                selCols = binCols + mlatCols + mltCols + otrCols
                ssBinDF = ssBinDF[selCols]
        # Now we might want to rearrange the DF based on data_labels
        # so that its easier to split it later
        if self.trnValTestSplitData:
            # sort the index before splitting
            ssBinDF.sort_index(inplace=True)
            print("Splitting the data into train, validation and test")
            # get the counts in different labels
            splitBins = ssBinDF["data_label"].value_counts()
            indsTrain = numpy.empty([0], dtype=ssBinDF.index.dtype)
            indsVal = numpy.empty([0], dtype=ssBinDF.index.dtype)
            indsTest = numpy.empty([0], dtype=ssBinDF.index.dtype)
            for _dl in splitBins.index:
                _currInds = ssBinDF[ ssBinDF["data_label"] == _dl\
                            ].index.get_values()
                _split = numpy.split( _currInds, [int(self.trnSplit*_currInds.size),\
                                    int((self.trnSplit+self.valSplit)*_currInds.size)] )
                indsTrain = numpy.concatenate( [indsTrain, _split[0]] )
                indsVal = numpy.concatenate( [indsVal, _split[1]] )
                indsTest = numpy.concatenate( [indsTest, _split[2]] )
            # re-order the dataframe based on new splits
            ssBinDF = pandas.concat( [ ssBinDF.loc[numpy.sort(indsTrain)],\
                        ssBinDF.loc[numpy.sort(indsVal)],\
                         ssBinDF.loc[numpy.sort(indsTest)] ] )
        else:
            # sort the index to make sure non-ss intervals
            # are not segregated at the end
            ssBinDF.sort_index(inplace=True)
        print(ssBinDF.head())
        # save the file to make future calc faster
        if saveBinData:
            # Note feather doesn't support datetime index
            # so we'll reset it and then convert back when
            # we read the file back!
            # sort by dates
            ssBinDF["date"] = ssBinDF.index
            ssBinDF.reset_index(drop=True).to_csv(saveFile)
        return ssBinDF

