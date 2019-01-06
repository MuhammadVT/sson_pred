import pandas
import datetime
import feather

class OnsetData(object):
    """
    Load the required data into a DF
    """
    def __init__(self, northData=True, southData=False, polarData=True,\
                 imageData=True, polarFile="../data/polar_data.feather",\
                imageFile="../data/image_data.feather", delTCutoff=2,\
                 fillTimeRes=1):
        """
        setup some vars and load preliminary data.
        Most of the variables are self explanatory.
        self.delTCutoff is the time range between two consecutive onset
        events that we want to use as cutoff. In other words we'll
        interpolate between these events if time is < self.delTCutoff,
        else we jump to the next event.
        self.fillTimeRes is the interpolation time resolutino
        """
        self.delTCutoff = delTCutoff
        self.fillTimeRes = fillTimeRes
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

    def _populate_date_list(self):
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
                    # get end time
                    endTime = datetime.datetime( row[1]["date"].year,\
                                                 row[1]["date"].month,\
                                                 row[1]["date"].day,\
                                                 row[1]["date"].hour,\
                                                 row[1]["date"].minute)
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
                    # get end time
                    endTime = datetime.datetime( row[1]["date"].year,\
                                                 row[1]["date"].month,\
                                                 row[1]["date"].day,\
                                                 row[1]["date"].hour,\
                                                 row[1]["date"].minute)
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
    
    def create_non_ss_bins(self, ndd, nonSSDataCnt,\
                          binTimeRes=30, nBins=3):
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
            while (_d[1] - _it).total_seconds()/(60) >= binTimeRes*nBins:
                nonSSDtList.append(_it)
                _it += datetime.timedelta(seconds=self.fillTimeRes*60)
        if nonSSDataCnt < len(nonSSDtList):
            print("more data found than required---", nonSSDataCnt, len(nonSSDtList))
            indices = random.sample(range(len(nonSSDtList)), nonSSDataCnt)
            nonSSDtList = [ nonSSDtList[_i] for _i in sorted(indices) ]
        return pandas.DataFrame(nonSSDtList, columns=["date"])

    def create_output_bins(self,\
                 aulDBdir="../data/sqlite3/", \
                 aulDBName="smu_sml_sme.sqlite",\
                 aulTabName="smusmlsme", alSSCutoff = -25, \
                 aeSSCutoff = 50, minDelT = 5,\
                 binTimeRes=30, nBins=3, saveBinData=True,\
                 saveFile="../data/binned_data_extra.feather",\
                 getNonSSInt=True, noSSbinRatio=1, dropDefaulnonSS=True):
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
            self.polarDF = self.polarDF.set_index(self.polarDF["date"])
        if self.imageDF is not None:
            self.imageDF = self.imageDF.set_index(self.imageDF["date"])
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
            _tmpBinVlas = [ 0 for _tv in range(nBins) ]
            _tmpLatVlas = [ -1. for _tv in range(nBins) ]
            _tmpMLTVlas = [ -1. for _tv in range(nBins) ]
            # get a start time and end time for search
            srchStTime = _cpDate.strftime("%Y-%m-%d %H:%M:%S")
            srchETime = (_cpDate + datetime.timedelta(minutes=binTimeRes*nBins)\
                ).strftime("%Y-%m-%d %H:%M:%S")
            _cOnsetList = self.polarDF.loc[ srchStTime : srchETime ].index.tolist()
            # get the difference between current time and the nearest
            # one's found in the DF
            # we'll ignore the time if the difference is less than 1 minute
            for _cto in sorted(_cOnsetList):
                _dt = (_cto - _cpDate).total_seconds()/60.
                for _nb in range(nBins):
                    if (_dt >= (_nb*binTimeRes) + 1) & (_dt <= (_nb+1)*binTimeRes):
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
                if _minDelT <= 1.:
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
            _tmpBinVlas = [ 0 for _tv in range(nBins) ]
            _tmpLatVlas = [ -1. for _tv in range(nBins) ]
            _tmpMLTVlas = [ -1. for _tv in range(nBins) ]
            # get a start time and end time for search
            srchStTime = _cpDate.strftime("%Y-%m-%d %H:%M:%S")
            srchETime = (_cpDate + datetime.timedelta(minutes=binTimeRes*nBins)\
                ).strftime("%Y-%m-%d %H:%M:%S")
            _cOnsetList = self.imageDF.loc[ srchStTime : srchETime ].index.tolist()
            # get the difference between current time and the nearest
            # one's found in the DF
            # we'll ignore the time if the difference is less than 1 minute
            for _cto in sorted(_cOnsetList):
                _dt = (_cto - _cpDate).total_seconds()/60.
                for _nb in range(nBins):
                    if (_dt >= (_nb*binTimeRes) + 1) & (_dt <= (_nb+1)*binTimeRes):
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
                if _minDelT <= 1.:
#                     imageClstOnsetTime.append(-1.)
                    imageClstDelT.append(-1.)
                else:
#                     imageClstOnsetTime.append(min(_cOnsetList))
                    imageClstDelT.append(_minDelT)
            else:
#                 imageClstOnsetTime.append(-1.)
                imageClstDelT.append(-1.)
            
        # convert the bin lists into dataframes
        # POLAR
        polDataSet = pandas.DataFrame(polarBinList,\
                     columns=["bin_" + str(_i) for _i in range(nBins)])
        # add the additional mlat,mlt cols
        polDataSet[["mlat_" + str(_i) for _i in range(nBins)\
                                   ]] = pandas.DataFrame(polarMlatList)
        polDataSet[["mlt_" + str(_i) for _i in range(nBins)\
                                   ]] = pandas.DataFrame(polarMLTList)
        # set the closest time cols
#         polDataSet["closest_time"] = polarClstOnsetTime
        polDataSet["del_minutes"] = polarClstDelT
        
        polDataSet = polDataSet.set_index(\
                    pandas.to_datetime(self.polarDateList))
        # IMAGE
        imgDataSet = pandas.DataFrame(imageBinList,\
                     columns=["bin_" + str(_i) for _i in range(nBins)])
        # add the additional mlat,mlt cols
        imgDataSet[["mlat_" + str(_i) for _i in range(nBins)]] = pandas.DataFrame(imageMlatList)
        imgDataSet[["mlt_" + str(_i) for _i in range(nBins)]] = pandas.DataFrame(imageMLTList)
        # set the closest time cols
#         imgDataSet["closest_time"] = imageClstOnsetTime
        imgDataSet["del_minutes"] = imageClstDelT
        imgDataSet = imgDataSet.set_index(\
                        pandas.to_datetime(self.imageDateList))
        # Now merge both the dataframes!
        ssBinDF = pandas.concat( [ polDataSet, imgDataSet ] )
        # sort by index(dates here)
        ssBinDF.sort_index(inplace=True)
        # Now we'll get the non-ss times
        if getNonSSInt:
            _sDt = ssBinDF.index.min()
            _eDt = ssBinDF.index.max()
            # get binary rep of the bins
            filterCols = [ col for col in ssBinDF\
                         if col.startswith('bin') ]
            ssBinDF = ssBinDF.apply( self.onset_binary, axis=1, args=(filterCols,) )
            if dropDefaulnonSS:
                ssBinDF = ssBinDF[ ssBinDF["outBinary"] > 0. ] 
            # Now get value counts for the binary 
            valCntDict = ssBinDF["outBinary"].value_counts().to_dict()
            # get the number of data points needed for non-ss intervals
            nonSSDataCnt = int( max(valCntDict.values()) * noSSbinRatio)
            # we'll set minDiffTime to be slightly higher than
            # bin time resolution * number of bins. This way
            # we make sure there is a good set of 0's
            minDiffTime = binTimeRes*(nBins+1)
            ndd = self.get_non_ss_intervals(_sDt, _eDt, aulDBdir, \
                 aulDBName, aulTabName, alSSCutoff = alSSCutoff, \
                 aeSSCutoff = aeSSCutoff, minDelT = minDelT,\
                 minDiffTime = minDiffTime)
            # get the nonSSDF from the dates
            nonSSDF = self.create_non_ss_bins(ndd, nonSSDataCnt,\
                                binTimeRes=binTimeRes, nBins=nBins)
            # add 0's to the bins
            for _cc in ssBinDF.columns:
                if "bin" in _cc:
                    nonSSDF[_cc] = 0
                else:
                    nonSSDF[_cc] = -1.
            # make both the DFs similar
            nonSSDF = nonSSDF.set_index(\
                            pandas.to_datetime(nonSSDF["date"].tolist()))
            # we'll select the cols we need
            binCols = [ col for col in ssBinDF\
                         if col.startswith('bin') ]
            mlatCols = [ col for col in ssBinDF\
                         if col.startswith('mlat') ]
            mltCols = [ col for col in ssBinDF\
                         if col.startswith('mlt') ]
            otrCols = [ "del_minutes" ]
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
        print(ssBinDF.head())
        # save the file to make future calc faster
        if saveBinData:
            # Note feather doesn't support datetime index
            # so we'll reset it and then convert back when
            # we read the file back!
            ssBinDF["date"] = ssBinDF.index
            ssBinDF.reset_index(drop=True).to_feather(saveFile)
        return ssBinDF

