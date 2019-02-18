import matplotlib
matplotlib.use("Agg")
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
plt.style.use("fivethirtyeight")
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

add_UT = False
use_stat_features = True
#use_stat_features = False

omnHistory = 180
# Select parameters of interest
omnTrainParams_actual = ["Bx", "By", "Bz", "Vx", "Np"]    # This is the one that goes into the actual training
param_col_dict = {"Bx":0, "By":1, "Bz":2, "Vx":3, "Np":4, "UT_sine":5, "UT_cosine":6}
input_cols = [param_col_dict[x] for x in omnTrainParams_actual]

# Set input file names
file_dir = "../data/omn_Bx_By_Bz_Vx_Np/"
input_file = file_dir +\
	     "input.nBins_1.binTimeRes_60.onsetFillTimeRes_30.onsetDelTCutoff_4."+\
	     "omnHistory_180.omnDBRes_1.imfNormalize_True.shuffleData_False." +\
	     "dateRange_19970101_20171229.npy"
csv_file = file_dir +\
	   "sml_nBins_1.binTimeRes_60.onsetFillTimeRes_30.onsetDelTCutoff_4."+\
	   "omnHistory_180.omnDBRes_1.imfNormalize_True.shuffleData_False."+\
	   "dateRange_19970101_20171229.csv"

# Load the data
print("Loading the data...")
X = np.load(input_file)
df = pd.read_csv(csv_file, index_col=0, parse_dates={"datetime":[0]})
y = df.loc[:, "label"].values.reshape(-1, 1)

# Select certain columns
X = X[:, :, input_cols]

## Limit the number of data points
#npoints = 1000
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
    X = X.reshape((X.shape[0], -1))


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

# Do PCA
n_components = 0.95
print("Doing PCA ...")
pca = PCA(n_components=n_components)
Xn = pca.fit_transform(X)
#print("explained_variance_:", pca.explained_variance_)
print("explained_variance_ratio:", pca.explained_variance_ratio_)
print("singular_values:", pca.singular_values_)

#############################################
# Plot the % explained variance
fig, ax = plt.subplots()
ax.plot(range(1, len(pca.explained_variance_ratio_)+1), np.cumsum(pca.explained_variance_ratio_), marker="o")
ax.set_xlabel('Number of Components')
ax.set_ylabel('Cumulative Explained Variance');
#ax.set_xlim([0, len(pca.explained_variance_ratio_)])
ax.set_ylim([0,1.])
#ax.xaxis.set_major_locator(MultipleLocator(base=1))
if use_stat_features:
    fname = "explained_variances_stat_features.png"
else:
    fname = "explained_variances_raw_features.png"

fig.savefig("./plots/" + fname, dpi=200, bbox_inches="tight")


#############################################
# Plot the data in transformed coords
fig, ax = plt.subplots()
idx_0 = np.where(y==0)
idx_1 = np.where(y==1)
ax.scatter(Xn[idx_0, 0], Xn[idx_0, 1], s=1., alpha=1.0, label="Non-Substorm")
ax.scatter(Xn[idx_1, 0], Xn[idx_1, 1], s=1., alpha=0.3, label="Substorm")
ax.set_xlabel("P1")
ax.set_ylabel("P2")
ax.legend(markerscale=5)
if use_stat_features:
    ax.set_xlim([-8,8])
    ax.set_ylim([-8,8])
    fname = "pca_stat_features.png"
else:
    ax.set_xlim([-60,60])
    ax.set_ylim([-60,60])
    fname = "pca_raw_features.png"

fig.savefig("./plots/" + fname, dpi=200, bbox_inches="tight")
#############################################

