import matplotlib
matplotlib.use("Agg")
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
plt.style.use("fivethirtyeight")
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import sqlite3

sml_onset_file =  "../data/" +\
                  "20190103-22-53-substorms_onset_sep_120min.csv"

# Read substorm onsets
df_onset = pd.read_csv(sml_onset_file, parse_dates={"onset_time":[0]})
df_onset.set_index("onset_time", inplace=True)

add_UT = False
#use_stat_features = True
use_stat_features = False

omnHistory = 120
# Select parameters of interest
omnTrainParams_actual = ["Bx", "By", "Bz", "Vx", "Np"]    # This is the one that goes into the actual training
param_col_dict = {"Bx":0, "By":1, "Bz":2, "Vx":3, "Np":4, "UT_sine":5, "UT_cosine":6}
input_cols = [param_col_dict[x] for x in omnTrainParams_actual]

# Set input file names
file_dir = "../data/omn_Bx_By_Bz_Vx_Np/"
#input_file = file_dir +\
#	     "input.nBins_1.binTimeRes_30.onsetFillTimeRes_5.onsetDelTCutoff_2."+\
#	     "omnHistory_120.omnDBRes_1.imfNormalize_True.shuffleData_False." +\
#	     "dateRange_19970101_20180101.npy"
#csv_file = file_dir +\
#	   "sml_nBins_1.binTimeRes_30.onsetFillTimeRes_5.onsetDelTCutoff_2."+\
#	   "omnHistory_120.omnDBRes_1.imfNormalize_True.shuffleData_False."+\
#	   "dateRange_19970101_20180101.csv"

input_file = file_dir +\
             "input.nBins_1.binTimeRes_60.onsetFillTimeRes_5.onsetDelTCutoff_4."+\
             "omnHistory_120.omnDBRes_1.imfNormalize_True.shuffleData_False." +\
             "dateRange_19970101_20180101.iso.npy"
csv_file = file_dir +\
           "sml_nBins_1.binTimeRes_60.onsetFillTimeRes_5.onsetDelTCutoff_4."+\
           "omnHistory_120.omnDBRes_1.imfNormalize_True.shuffleData_False."+\
           "dateRange_19970101_20180101.iso.csv"

# Load the data
print("Loading the data...")
X = np.load(input_file)
df = pd.read_csv(csv_file, index_col=0, parse_dates={"datetime":[0]})
y = df.loc[:, "label"].values.reshape(-1, 1)

# Select certain columns
X = X[:, :, input_cols]

# Limit the number of data points
#npoints = 10000
#X = X[:npoints, :, :]
#y = y[:npoints, :]
#df = df.iloc[:npoints,:]

# Create features
if use_stat_features:
    X_mean = X.mean(axis=1)
    X_std = X.std(axis=1)
    X_min = X.min(axis=1)
    X_max = X.max(axis=1)
    X = np.concatenate([X_mean, X_std, X_min, X_max], axis=1)
else:
    # Flatten X
    Xr = X.reshape((X.shape[0], -1), order="F")

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
    Xr = np.concatenate([Xr, minutes_sine, minutes_cosine], axis=2)

# Do PCA
n_components = 0.95
print("Doing PCA ...")
pca = PCA(n_components=n_components)
Xn = pca.fit_transform(Xr)
#print("explained_variance_:", pca.explained_variance_)
print("explained_variance_ratio:", pca.explained_variance_ratio_)
print("singular_values:", pca.singular_values_)

idx_p2_pos = np.where((Xn[:, 1] >= 0) * (y[:, 0]==1))[0]
idx_p2_neg = np.where((Xn[:, 1] < 0) * (y[:, 0]==1))[0]

#idx_p2_pos = np.where((Xn[:, 1] >= 0) * (y[:, 0]==0))[0]
#idx_p2_neg = np.where((Xn[:, 1] < 0) * (y[:, 0]==0))[0]

#############################################
# Plot hist of each features
fig, axes = plt.subplots(nrows=len(omnTrainParams_actual), ncols=1,
                         sharex=True, figsize=(6, 8))
fig.subplots_adjust(hspace=0.5)
bins = 100
dat_range = (-5, 5)
histtype = "step"
print("Plotting hist of each feature...")
for i, ax in enumerate(axes):
    ax.hist(X[idx_p2_pos, :, i].flatten(), bins=bins, range=dat_range, color="blue",
            histtype=histtype, density=True, alpha=1.0, linewidth=1.5, label="Non-SS-P2+")
    ax.hist(X[idx_p2_neg, :, i].flatten(), bins=bins, range=dat_range, color="red",
            histtype=histtype, density=True, alpha=1.0, linewidth=1.5, label="Non-SS-P2-")
    ax.set_xlabel("Normalized " + omnTrainParams_actual[i])
    ax.set_ylabel("Prob.")
    ax.set_xlim([-4, 4])
    ax.legend(fontsize="small")

#fname = "non_SS_cluster_feature_hist.png"
fname = "cluster_feature_hist.png"
fig.savefig("./plots/" + fname, dpi=200, bbox_inches="tight")
#############################################

##############################################
# Plotting MLAT, MLT histograms
fig, axes = plt.subplots(nrows=2, ncols=1,
                         figsize=(8, 8))
fig.subplots_adjust(hspace=0.5)
print("Plotting MLAT, MLT histograms...")
df_p2_pos = df_onset.loc[df.iloc[idx_p2_pos, :].index, :]
df_p2_neg = df_onset.loc[df.iloc[idx_p2_neg, :].index, :]
mlt_p2_pos = [x if x <=12 else x-24 for x in df_p2_pos.MLT.values]
mlt_p2_neg = [x if x <=12 else x-24 for x in df_p2_neg.MLT.values]
bins = 50
axes[0].hist(df_p2_pos.MLAT.values, bins=bins, range=None, color="blue",
        histtype=histtype, density=True, alpha=1.0, linewidth=1.5, label="SS-P2+")
axes[0].hist(df_p2_neg.MLAT.values, bins=bins, range=None, color="red",
        histtype=histtype, density=True, alpha=1.0, linewidth=1.5, label="SS-P2-")
axes[1].hist(mlt_p2_pos, bins=bins, range=None, color="blue",
        histtype=histtype, density=True, alpha=1.0, linewidth=1.5, label="SS-P2+")
axes[1].hist(mlt_p2_neg, bins=bins, range=None, color="red",
        histtype=histtype, density=True, alpha=1.0, linewidth=1.5, label="SS-P2-")

axes[0].set_xlabel("MLAT")
axes[1].set_xlabel("MLT")
axes[0].set_ylabel("Prob. Density")
axes[1].set_ylabel("Prob. Density")
axes[0].legend(fontsize="small")
axes[1].legend(fontsize="small")

fname = "cluster_MLAT_MLT_hist.png"
fig.savefig("./plots/" + fname, dpi=200, bbox_inches="tight")
#############################################


