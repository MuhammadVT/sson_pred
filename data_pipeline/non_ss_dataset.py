import pandas
import datetime
import sqlite3

class NonSSData(object):
    """
    Load the required data into a DF
    """
    def __init__(self, startDate, endDate, aulDBdir, aulDBName,\
                 aulTabName, alSSCutoff = -10, aeSSCutoff = 50,\
                 minDelT = 5, minDiffTime = 180):
        """
        setup some vars
        alSSCutoff and aeSSCutoff are cutoffs for al and ae
        to identify non-ss periods.
        minDelT is the minimum time gap where al/ae can exceed
        cutoffs.
        minDiffTime is the continuous set of time range
        which is considered a non-substorm
        period. In other words, if non-SS
        conditions are satisfied for this
        interval of time (say 3 hours), then
        we consider this period to be a non-ss
        period.
        """
        self.startDate = startDate
        self.endDate = endDate
        self.aulDBdir = aulDBdir
        self.aulDBName = aulDBName
        self.aulTabName = aulTabName
        self.alSSCutoff = alSSCutoff
        self.aeSSCutoff = aeSSCutoff
        self.minDelT = minDelT
        self.minDiffTime = minDiffTime
        self.aulDF = self._load_aul_data()

    def _load_aul_data(self):
        """
        Load AUL data
        """
        conn = sqlite3.connect(self.aulDBdir + self.aulDBName,
                       detect_types = sqlite3.PARSE_DECLTYPES)
        # load data to a dataframe
        command = "SELECT * FROM {tb} " +\
                  "WHERE datetime BETWEEN '{stm}' and '{etm}'"
        command = command.format(tb=self.aulTabName,\
                                 stm=self.startDate, etm=self.endDate)
        return pandas.read_sql(command, conn)

    def get_non_ss_dates(self):
        """
        Get a list of dates where non-ss intervals
        were identified!
        """
        self.aulDF = self.aulDF[ (self.aulDF["al"] >= self.alSSCutoff) &\
             (self.aulDF["ae"] <= self.aeSSCutoff)\
             ].reset_index(drop=True)
        # Calculate the time diff between two consecutive timesteps
        self.aulDF["delT"] = self.aulDF["datetime"].diff()
        # convert the difference to minutes
        self.aulDF["delT"] = self.aulDF["delT"].apply(\
                    lambda x: x.total_seconds()/60. )
        # get contiguous set of measurements where
        # there is no ss
        # here we're getting a diff on the series to get
        # the most continuous dataset
        brkInds = self.aulDF[ (self.aulDF["delT"] > self.minDelT)\
                       ].index.to_frame().diff().reset_index()
        brkInds.columns = [ "inds", "diffs" ]
        # now we also need index value from prev row
        # to get the range of time
        shftdRows = brkInds["inds"].shift(1)
        brkInds["prevRowInds"] = shftdRows
        # store the dates in a list and return them!
        nonSSDtList = []
        for row in brkInds[ brkInds["diffs"] > self.minDiffTime ].iterrows():
            dd = self.aulDF.iloc[ \
                int(row[1]["prevRowInds"]): int(row[1]["inds"]-1)]["datetime"]
            nonSSDtList.append( (dd.min(), dd.max()) )
        return nonSSDtList




