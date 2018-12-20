import pandas
import datetime
import numpy
import sqlite3


class OmnData(object):
    """
    Utils to work with omni data
    """
    def __init__(self, start_date, end_date, omn_dbdir, \
                omn_db_name, omn_table_name, omn_train, imf_normalize, model_path, db_time_resolution=5,\
                omn_train_params = [ "By", "Bz", "Vx", "Np" ]):
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
        self.imf_normalize = imf_normalize
        self.model_path = model_path
        self.omnDF = self._load_omn_data()

    def _load_omn_data(self):
        """
        Load all omni data
        """
        conn = sqlite3.connect(self.omn_dbdir + self.omn_db_name,
                       detect_types = sqlite3.PARSE_DECLTYPES)
        # load data to a dataframe
        command = "SELECT * FROM {tb} " +\
                  "WHERE datetime BETWEEN '{stm}' and '{etm}'"
        command = command.format(tb=self.omn_table_name,\
                                 stm=self.start_date, etm=self.end_date)
        omnDF = pandas.read_sql(command, conn)
        # We'll do some processing to 
        # fill missing values in IMF
        # Now we need to find missing dates
        # get a list of dates we have and reindex
        new_omn_index_arr = []
        curr_time = self.start_date
        while curr_time <= self.end_date:
            new_omn_index_arr.append( curr_time )
            curr_time += datetime.timedelta(minutes=self.db_time_resolution)
        omnDF = omnDF.replace(numpy.inf, numpy.nan)
        omnDF = omnDF.set_index("datetime").reindex(new_omn_index_arr).reset_index()
        # Replace nan's with preceding value (forward filling)
        omnDF = omnDF.fillna(method='ffill')
        
        if (self.imf_normalize == True):
            print ('normalizing the IMF data ...')
            if(self.omn_train == True):
                #Once the data is loaded, we normalize the columns based on its respective mean and std (z-score)
                
                #Storing the current mean and std which will be used to normalize the test data (in get_prediction file)

                mean_std_values = (omnDF[self.omn_train_params].mean(), omnDF[self.omn_train_params].std())
                print ("mean and std values...")
                print (mean_std_values)    
                numpy.save(self.model_path+'/mean_std', mean_std_values)

                #This operation does the column wise normalization only on the selected columns             
                omnDF[self.omn_train_params] = omnDF[self.omn_train_params].apply(lambda x: (x - x.mean()) / x.std())
            else:
                
                #this part will be called when get_prediction file is run for getting the predicted values
                print ("Using the mean_std.npy file for IMF normalization ...")
                mean_std_values = numpy.load(self.model_path+'/mean_std.npy')
                col_means, col_stds = mean_std_values
                print ("mean is:", col_means)
                print ("std is:", col_stds)
                
                i = 0
                for imf in self.omn_train_params:
                    omnDF[imf] = (omnDF[imf] - col_means[i]) / col_stds[i]   
                    i += 1
                      
return omnDF