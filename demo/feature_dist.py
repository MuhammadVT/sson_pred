import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
plt.style.use("fivethirtyeight")
import numpy as np
import pandas as pd

def plot_feature_hist(omnHistory = 120,
                      omnTrainParams_actual = ["Bx", "By", "Bz", "Vx", "Np"],
                      add_UT = False,
                      use_stat_features = False,
                      file_dir = "../../DAAS_project/data/substorms/"):
    """ Plots the histograms of the input features for two classes. 
    Returns None.

    keyword arguments:
    omnHistory -- the time history of the input data
    omnTrainParams_actual -- Parameters that go into the actual training,
                             default to ["Bx", "By", "Bz", "Vx", "Np"] 
    add_UT -- whether to add UT time as input feature
    use_stat_features -- whether to use the statistical features as inputs
    file_dir -- file directory for the input and output datasets
    """

    # Select parameters of interest
    param_col_dict = {"Bx":0, "By":1, "Bz":2, "Vx":3, "Np":4, "UT_sine":5, "UT_cosine":6}
    input_cols = [param_col_dict[x] for x in omnTrainParams_actual]

    # Set input file names
    input_file = file_dir +\
                 "input.nBins_1.binTimeRes_60.onsetFillTimeRes_5.onsetDelTCutoff_4."+\
                 "omnHistory_120.omnDBRes_1.imfNormalize_True.shuffleData_False." +\
                 "dateRange_19970101_20180101.iso.npy"
    csv_file = file_dir +\
               "sml_nBins_1.binTimeRes_60.onsetFillTimeRes_5.onsetDelTCutoff_4."+\
               "omnHistory_120.omnDBRes_1.imfNormalize_True.shuffleData_False."+\
               "dateRange_19970101_20180101.iso.csv"

    #input_file = file_dir +\
    #             "input.nBins_1.binTimeRes_60.onsetFillTimeRes_30.onsetDelTCutoff_4."+\
    #             "omnHistory_120.omnDBRes_1.imfNormalize_True.shuffleData_False." +\
    #             "dateRange_19970101_20180101.no_marching.npy"
    #csv_file = file_dir +\
    #           "sml_nBins_1.binTimeRes_60.onsetFillTimeRes_30.onsetDelTCutoff_4."+\
    #           "omnHistory_120.omnDBRes_1.imfNormalize_True.shuffleData_False."+\
    #           "dateRange_19970101_20180101.no_marching.csv"

    # Load the data
    print("Loading the data...")
    X = np.load(input_file)
    X = X[:, :, input_cols]
    df = pd.read_csv(csv_file, index_col=0, parse_dates={"datetime":[0]})
    y = df.loc[:, "label"].values.reshape(-1, 1)

    # Balance the two classes
    ss_idx = np.where(df.label.values == 1)[0]
    nonss_idx = np.where(df.label.values == 0)[0]
    np.random.seed(1)
    nonss_idx = np.random.choice(nonss_idx, len(ss_idx), replace=False)
    event_idx = np.concatenate([ss_idx, nonss_idx])
    #np.random.shuffle(event_idx)
    event_idx.sort()
    df = df.iloc[event_idx, :]

    # Select for certain rows and columns
    X = X[event_idx]
    y = y[event_idx]

    # Select for omnHistory
    X = X[:, -omnHistory-1:, :]

    ## Limit the number of data points
    #npoints = 1000
    #X = X[:npoints, :, :]
    #y = y[:npoints, :]
    #df = df.iloc[:npoints,:]

    idx_0 = np.where(y==0)[0]
    idx_1 = np.where(y==1)[0]
    # Create features
    if use_stat_features:
        X_mean = X.mean(axis=1)
        X_std = X.std(axis=1)
        X_min = X.min(axis=1)
        X_max = X.max(axis=1)
        X = np.concatenate([X_mean, X_std, X_min, X_max], axis=1)

    if add_UT:
        # Add UT time features
        print("Adding the UT time features...")
        dlm = np.array([np.timedelta64(i-omnHistory, "m") for i in range(omnHistory+1)])
        dlms = np.tile(dlm, (X.shape[0], 1))
        dtms = np.tile(df.index.values.reshape((-1, 1)), (1, omnHistory+1))
        dtms = dtms + dlms
        minutes = (dtms - dtms.astype("datetime64[D]")).astype("timedelta64[m]").astype(int)
        minutes = minutes.reshape((minutes.shape[0], omnHistory+1, 1))
        # Encode UT minutes using sine and cosine functions
        minutes_sine = np.sin(2*np.pi/(60*24) * minutes)
        minutes_cosine = np.cos(2*np.pi/(60*24) * minutes)
        #months = (dtms - dtms.astype("datetime64[D]")).astype("timedelta64[m]").astype(int)
        X = np.concatenate([X, minutes_sine, minutes_cosine], axis=2)

    if not use_stat_features:
        # Plot feature hist
        fig, axes = plt.subplots(nrows=len(omnTrainParams_actual), ncols=1,
                                 sharex=True, figsize=(6, 8))
        fig.subplots_adjust(hspace=0.5)
        bins = 100
        dat_range = (-5, 5)
        histtype = "step"
        print("Plotting hist of each feature...")
        for i, ax in enumerate(axes):
            ax.hist(X[idx_0, :, i].flatten(), bins=bins, range=dat_range, color="blue",
                    histtype=histtype, density=True, alpha=1.0, linewidth=1.5, label="Non-SS")
            ax.hist(X[idx_1, :, i].flatten(), bins=bins, range=dat_range, color="red",
                    histtype=histtype, density=True, alpha=1.0, linewidth=1.5, label="SS")
            ax.set_title("Normalized " + omnTrainParams_actual[i])
            ax.set_ylabel("Prob.") 
            ax.set_xlim([-4, 4])
            ax.legend(fontsize="small")
            if i < (len(axes)-1):
                ax.set_ylim([0, 0.6])
            else:
                ax.set_ylim([0, 1.0])
            
        #axes[0].set_xlabel("Normalized " + omnTrainParams_actual[i])
        #fig.suptitle(str(df.loc[df.label==1, :].shape[0]) + " Substorms, " +\
        #             str(df.loc[df.label==1, :].shape[0]) + " Non-Substorms")

        #fname = "feature_hist_iso_events" + "_omnHistory" + str(omnHistory) + ".png"
        fname = "feature_hist" + \
                "_omnHistory" + str(omnHistory) + "_noMarching" + ".png"
        fig.savefig("./plots/" + fname, dpi=200, bbox_inches="tight")

    else:
        # Plot feature hist
        stat_features = ["avg", "std", "min", "max"]
        for k, param in enumerate(stat_features):
            fig, axes = plt.subplots(nrows=len(omnTrainParams_actual), ncols=1,
                                     sharex=True, figsize=(6, 8))
            fig.subplots_adjust(hspace=0.5)
            bins = 100
            dat_range = (-5, 5)
            histtype = "step"
            print("Plotting hist of each feature...")
            for i, ax in enumerate(axes):
                ax.hist(X[idx_0, k*len(omnTrainParams_actual) + i].flatten(), bins=bins, range=dat_range, color="blue",
                        histtype=histtype, density=True, alpha=1.0, linewidth=1.5, label="Non-SS")
                ax.hist(X[idx_1, k*len(omnTrainParams_actual) + i].flatten(), bins=bins, range=dat_range, color="red",
                        histtype=histtype, density=True, alpha=1.0, linewidth=1.5, label="SS")
                ax.set_xlabel("Normalized " + omnTrainParams_actual[i])
                ax.set_ylabel("Prob.") 
                ax.set_xlim([-4, 4])
                ax.legend(fontsize="small")

            fname = "stat_feature_" + param + "_hist_iso_events" + "_omnHistory" + str(omnHistory) + ".png"
            fig.savefig("./plots/" + fname, dpi=200, bbox_inches="tight")


