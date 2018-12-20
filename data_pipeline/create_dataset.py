import pandas
import datetime


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
        Most of the variables are self explanatory
        delTCutoff is the time range between two consecutive onset
        events that we want to use as cutoff. In other words we'll
        interpolate between these events if time is < delTCutoff,
        else we jump to the next event.
        """
        self.polarDF = None
        self.imageDF = None
        if polarData:
            self.polarDF = pandas.read_feather(polarFile)
        if imageData:
            self.imageDF = pandas.read_feather(imageFile)
        # hemispheres to use!
        if (not northData) & (not southData):
            print("No hemi chosen! choosing north")
            northData = True
        self.northData = northData
        self.southData = southData

    def filter_data(self):
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



