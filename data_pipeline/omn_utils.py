import pandas
import datetime
import numpy
import sqlite3


class OmnData(object):
    """
    Utils to work with omni data
    """
    def __init__(self, start_date, end_date, omn_dbdir, omn_db_name,\
                 omn_table_name, omn_train, omn_norm_param_file,\
                sml_train=True, sml_norm_file=None, imf_normalize=True,\
                smlDbName=None, sml_normalize=True, smlTabName=None, db_time_resolution=1, \
                include_omn=True, omn_train_params = [ "By", "Bz", "Bx", "Vx", "Np" ],\
                include_sml=False, sml_train_params = [ "au", "al" ] ):
        """
        setup some vars
        """
        self.start_date = start_date
        self.end_date = end_date
        self.omn_dbdir = omn_dbdir
        self.omn_db_name = omn_db_name
        self.omn_table_name = omn_table_name
        self.db_time_resolution = db_time_resolution
        self.omn_train_params = omn_train_params
        self.omn_train = omn_train
        self.sml_train = sml_train
        self.imf_normalize = imf_normalize
        self.omn_norm_param_file = omn_norm_param_file
        self.sml_train_params = sml_train_params
        self.paramDBDir = omn_dbdir
        self.sml_normalize = sml_normalize
        self.sml_norm_file = sml_norm_file
        # if both omn and sml are set to false 
        # set include omn to true! you need some input
        # right!
        if (not include_sml) & (not include_omn):
            include_omn = True
        if include_omn:
            self.omnDF = self._load_omn_data()
            print("loaded OMNI data")
        if include_sml:
            if len(sml_train_params) > 0:
                if smlDbName is None:
                    print("Enter a valid sml db name")
                if smlTabName is None:
                    print("Enter a valid sml tab name")
                self.smDF = self._load_ind_data(smlDbName, smlTabName)
                self.smDF = self.smDF[ ["datetime"] + self.sml_train_params ]
                print("loaded SML data")
        # Merge the data
        if include_omn & include_sml:
            self.omnDF = pandas.merge(self.omnDF, self.smDF,\
                             how="inner",on="datetime")
        if include_sml & (not include_omn):
            self.omnDF=self.smDF

    def _load_ind_data(self, dbName, tabName):
        """
        Load AUL data
        """
        conn = sqlite3.connect(self.paramDBDir + dbName,
                       detect_types = sqlite3.PARSE_DECLTYPES)
        # load data to a dataframe
        command = "SELECT * FROM {tb} " +\
                  "WHERE datetime BETWEEN '{stm}' and '{etm}'"
        command = command.format(tb=tabName, stm=self.start_date,\
                                  etm=self.end_date)
        indDF = pandas.read_sql(command, conn)
        indDF = indDF.replace(numpy.inf, numpy.nan)
        indDF = indDF.set_index("datetime")
        # Add self.start_date to indDF in case if it is missing
        if self.start_date not in indDF.index:
            indDF.loc[self.start_date] = numpy.nan
        if self.end_date not in indDF.index:
            indDF.loc[self.end_date] = numpy.nan
        indDF.sort_index(inplace=True)
        indDF = indDF.resample( str(self.db_time_resolution) + "Min" ).ffill().reset_index()
        
        # Replace nan's with preceding value (forward filling)
        indDF = indDF.fillna(method='ffill').fillna(method='bfill')
        if (self.sml_normalize == True):
            print ('normalizing the IMF data ...')
            if(self.sml_train == True):
                #Once the data is loaded, we normalize the columns based on its respective mean and std (z-score)
                
                #Storing the current mean and std which will be used to normalize the test data (in get_prediction file)

                mean_std_values = (indDF[self.sml_train_params].mean(), indDF[self.sml_train_params].std())
                print("mean and std values...")
                print(mean_std_values)    
                numpy.save(self.sml_norm_file, mean_std_values)

                #This operation does the column wise normalization only on the selected columns             
                indDF[self.sml_train_params] = indDF[self.sml_train_params].apply(lambda x: (x - x.mean()) / x.std())
            else:
                
                #this part will be called when get_prediction file is run for getting the predicted values
                print ("Using the mean_std.npy file for IMF normalization ...")
                mean_std_values = numpy.load(self.sml_norm_file)
                col_means, col_stds = mean_std_values
                print ("mean is:", col_means)
                print ("std is:", col_stds)
                
                i = 0
                for imf in self.sml_train_params:
                    indDF[imf] = (indDF[imf] - col_means[i]) / col_stds[i]   
                    i += 1  
        return indDF

    def _load_omn_data(self):
        """
        Load all omni data
        """
        conn = sqlite3.connect(self.omn_dbdir + self.omn_db_name,
                       detect_types = sqlite3.PARSE_DECLTYPES)
        # load data to a dataframe
        command = "SELECT * FROM {tb} WHERE datetime BETWEEN '{stm}' AND '{etm}'"
        command = command.format(tb=self.omn_table_name,\
                                 stm=self.start_date, etm=self.end_date)
        omnDF = pandas.read_sql(command, conn)
        # print omnDF.head()
        # We'll do some processing to 
        # fill missing values in IMF
        # Now we need to find missing dates
        # get a list of dates we have and reindex
        # new_omn_index_arr = []
        # curr_time = self.start_date
        # while curr_time <= self.end_date:
        #     new_omn_index_arr.append( curr_time )
        #     curr_time += datetime.timedelta(minutes=self.db_time_resolution)
        
        omnDF = omnDF[ self.omn_train_params + [ "datetime" ] ]
        omnDF = omnDF.replace(numpy.inf, numpy.nan)
        omnDF = omnDF.set_index("datetime")
        # Add self.start_date to omnDF in case if it is missing
        if self.start_date not in omnDF.index:
            omnDF.loc[self.start_date] = numpy.nan
        if self.end_date not in omnDF.index:
            omnDF.loc[self.end_date] = numpy.nan
        omnDF.sort_index(inplace=True)
        omnDF = omnDF.resample( str(self.db_time_resolution) + "Min" ).ffill().reset_index()
        
        # Replace nan's with preceding value (forward filling)
        # omnDF = omnDF.fillna(method='ffill').fillna(method='bfill')
        
        if (self.imf_normalize == True):
            print ('normalizing the IMF data ...')
            if(self.omn_train == True):
                #Once the data is loaded, we normalize the columns based on its respective mean and std (z-score)
                #Storing the current mean and std which will be used to normalize the test data (in get_prediction file)
                mean_std_values = (omnDF[self.omn_train_params].mean(), omnDF[self.omn_train_params].std())
                print("mean and std values...")
                print(mean_std_values)    
                numpy.save(self.omn_norm_param_file, mean_std_values)

                #This operation does the column wise normalization only on the selected columns             
                omnDF[self.omn_train_params] = omnDF[self.omn_train_params].apply(lambda x: (x - x.mean()) / x.std())
            else:
                
                #this part will be called when get_prediction file is run for getting the predicted values
                print ("Using the mean_std.npy file for IMF normalization ...")
                mean_std_values = numpy.load(self.omn_norm_param_file)
                col_means, col_stds = mean_std_values
                print ("mean is:", col_means)
                print ("std is:", col_stds)
                
                i = 0
                for imf in self.omn_train_params:
                    omnDF[imf] = (omnDF[imf] - col_means[i]) / col_stds[i]   
                    i += 1  
        return omnDF
