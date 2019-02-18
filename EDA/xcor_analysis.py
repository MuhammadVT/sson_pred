import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
plt.style.use("fivethirtyeight")
import numpy as np
import pandas as pd

add_UT = True
corr_between_datapoints = False
corr_between_features = True

omnHistory = 120
# Select parameters of interest
omnTrainParams_actual = ["Bx", "By", "Bz", "Vx", "Np"]    # This is the one that goes into the actual training
param_col_dict = {"Bx":0, "By":1, "Bz":2, "Vx":3, "Np":4, "UT_sine":5, "UT_cosine":6}
input_cols = [param_col_dict[x] for x in omnTrainParams_actual]

# Set input file names
file_dir = "../data/omn_Bx_By_Bz_Vx_Np/"
input_file = file_dir +\
	     "input.nBins_1.binTimeRes_30.onsetFillTimeRes_5.onsetDelTCutoff_4."+\
	     "omnHistory_120.omnDBRes_1.imfNormalize_True.shuffleData_False." +\
	     "dateRange_19970101_20040101.npy"
csv_file = file_dir +\
	   "sml_nBins_1.binTimeRes_30.onsetFillTimeRes_5.onsetDelTCutoff_4."+\
	   "omnHistory_120.omnDBRes_1.imfNormalize_True.shuffleData_False."+\
	   "dateRange_19970101_20040101.csv"


# Load the data
print("Loading the data...")
X = np.load(input_file)
df = pd.read_csv(csv_file, index_col=0, parse_dates={"datetime":[0]})
y = df.loc[:, "label"].values.reshape(-1, 1)

# Select certain columns
X = X[:, :, input_cols]

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


#################################################
if corr_between_datapoints:

    # Limit the number of data points
    npoints = 10000
    X = X[:npoints, :, :]
    y = y[:npoints, :]
    df = df.iloc[:npoints,:]

    # Flatten X
    X = X.reshape((X.shape[0], -1))

    # Plot the correlation between data points
    print("Calculating corr coeff...")
    time_window = 240       # Minutes
    time_res = 5
    ncols = int(time_window/time_res) + 1
    xcor_arr = np.ones((X.shape[0]-ncols, ncols))
    for i in range(xcor_arr.shape[0]):
        for j in range(ncols):
            xcor_arr[i, j] = np.corrcoef(X[i, :], X[i+j, :])[0, 1]

    fig, ax = plt.subplots()
    print("Plotting the data...")
    for j in range(ncols):
        ax.scatter([time_res*j]*xcor_arr.shape[0], xcor_arr[:, j],
                   marker=".", s=0.2, c="gray")

    # Plot the mean and interquartile of the corr coeff
    ax.plot(range(0, time_window+time_res, time_res), xcor_arr.mean(axis=0), "r-o", linewidth=1., markersize=2)
    ax.plot(range(0, time_window+time_res, time_res), np.percentile(xcor_arr, 25, axis=0), "b-o", linewidth=1., markersize=2)
    ax.plot(range(0, time_window+time_res, time_res), np.percentile(xcor_arr, 75, axis=0), "b-o", linewidth=1., markersize=2)

    ax.set_ylim([-0.55, 1.05])
    ax.set_xlabel("Time [Minutes]")
    ax.set_ylabel("Correlation Coefficient")
    ax.xaxis.set_major_locator(MultipleLocator(60))
    ax.xaxis.set_minor_locator(MultipleLocator(time_res))

    if add_UT:
        fig.savefig("./corr_vs_time_10000points_withUT.png", dpi=200, bbox_inches="tight")
    else:
        fig.savefig("./corr_vs_time_10000points_withoutUT.png", dpi=200, bbox_inches="tight")
#################################################

#################################################
if corr_between_features:

    import seaborn as sns

    # Limit the number of data points
    #npoints = 10000
    npoints = X.shape[0]
    X = X[:npoints, :, :]
    y = y[:npoints, :]
    df = df.iloc[:npoints,:]

    if add_UT:
        col_names = ["Bx", "By", "Bz", "Vx", "Np", "UT_sine", "UT_cosine"] 
    else:
        col_names = ["Bx", "By", "Bz", "Vx", "Np"]    
    features = {ky: X[:, -1, param_col_dict[ky]].flatten() for ky in col_names}
    dfn = pd.DataFrame(data=features, index=df.index.values)

    # Plot the correlation between features
    fig, ax = plt.subplots()
    sns.heatmap(dfn.corr(), ax=ax, annot=True, fmt=".2f")

    fig.savefig("./features_corr.png", dpi=200, bbox_inches="tight")

#    import pdb
#    pdb.set_trace()

