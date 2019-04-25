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

threeD_plot = False 
if threeD_plot:
    threeD_txt = "_3D"
else:
    threeD_txt = ""

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

#input_file = file_dir +\
#             "input.nBins_1.binTimeRes_60.onsetFillTimeRes_5.onsetDelTCutoff_4."+\
#             "omnHistory_120.omnDBRes_1.imfNormalize_True.shuffleData_False." +\
#             "dateRange_19970101_20180101.iso.npy"
#csv_file = file_dir +\
#           "sml_nBins_1.binTimeRes_60.onsetFillTimeRes_5.onsetDelTCutoff_4."+\
#           "omnHistory_120.omnDBRes_1.imfNormalize_True.shuffleData_False."+\
#           "dateRange_19970101_20180101.iso.csv"

#input_file = file_dir +\
#             "input.nBins_1.binTimeRes_60.onsetFillTimeRes_30.onsetDelTCutoff_4."+\
#             "omnHistory_120.omnDBRes_1.imfNormalize_True.shuffleData_False." +\
#             "dateRange_19970101_20180101.no_marching.npy"
#csv_file = file_dir +\
#           "sml_nBins_1.binTimeRes_60.onsetFillTimeRes_30.onsetDelTCutoff_4."+\
#           "omnHistory_120.omnDBRes_1.imfNormalize_True.shuffleData_False."+\
#           "dateRange_19970101_20180101.no_marching.csv"

input_file = file_dir +\
             "input.nBins_1.binTimeRes_60.onsetFillTimeRes_30.onsetDelTCutoff_4."+\
             "omnHistory_120.omnDBRes_1.imfNormalize_True.shuffleData_False." +\
             "dateRange_19970101_20180101.interp_20.delay_10.npy"
csv_file = file_dir +\
           "sml_nBins_1.binTimeRes_60.onsetFillTimeRes_30.onsetDelTCutoff_4."+\
           "omnHistory_120.omnDBRes_1.imfNormalize_True.shuffleData_False."+\
           "dateRange_19970101_20180101.interp_20.delay_10.csv"


# Load the data
print("Loading the data...")
X = np.load(input_file)
X = X[:, :, input_cols]
df = pd.read_csv(csv_file, index_col=0, parse_dates={"datetime":[0]})
y = df.loc[:, "label"].values.reshape(-1, 1)

################################
## Select onset by delSML
#delSML_threshold = -200
#idx_delSML = np.where(~ ((df.del_sml.values > delSML_threshold) & (df.label.values==1)))[0]
##idx_delSML = np.where(~ ((df.del_sml.values < delSML_threshold) & (df.label.values==1)))[0]
#df = df.iloc[idx_delSML, :]
#X = X[idx_delSML, :, :]
#y = y[idx_delSML, :]

###############################
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
###############################

# Select for omnHistory
X = X[:, -omnHistory-1:, :]

# Limit the number of data points
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
    X = X.reshape((X.shape[0], -1), order="F")

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
#n_components = 0.95
n_components = 8 
print("Doing PCA ...")
pca = PCA(n_components=n_components)
Xn = pca.fit_transform(X)
#print("explained_variance_:", pca.explained_variance_)
print("explained_variance_ratio:", pca.explained_variance_ratio_)
print("singular_values:", pca.singular_values_)

#############################################
# Plot the % explained variance
#fig, ax = plt.subplots()
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
panel_labels = ["(a)", "(b)", "(c)", "(d)"]
ax = axes[0,0]
ax.plot(range(1, len(pca.explained_variance_ratio_)+1), np.cumsum(pca.explained_variance_ratio_), marker="o")
ax.set_xlabel('# Components')
ax.set_ylabel('Explained Variance');
#ax.set_xlim([0, len(pca.explained_variance_ratio_)])
ax.set_ylim([0,1.])
ax.xaxis.set_major_locator(MultipleLocator(base=1))
ax.yaxis.set_major_locator(MultipleLocator(base=0.2))
ax.annotate(panel_labels[0], xy=(0.05, 0.9), xycoords="axes fraction", fontweight="bold")

#############################################
# Plot the data in transformed coords
idx_0 = np.where(y==0)[0]
idx_1 = np.where(y==1)[0]
if threeD_plot:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d") 
    ax.scatter(Xn[idx_0, 0], Xn[idx_0, 1], Xn[idx_0, 2], marker="o", s=1., alpha=1.0, label="NSS")
    ax.scatter(Xn[idx_1, 0], Xn[idx_1, 1], Xn[idx_0, 2], marker="*",  s=1., alpha=0.3, label="SS")
else:
    #fig, ax = plt.subplots()
    for k, ij in enumerate([(0,1), (0,2), (1,2)]):
        ax = axes.flatten()[k+1]
        ax.scatter(Xn[idx_0, ij[0]], Xn[idx_0, ij[1]], marker="o", s=1.5, alpha=1.0, color="blue", label="NSS")
        ax.scatter(Xn[idx_1, ij[0]], Xn[idx_1, ij[1]], marker="o", s=1.5, alpha=0.1, color="red", label="SS")
        ax.set_xlabel("P"+str(ij[0]+1))
        ax.set_ylabel("P"+str(ij[1]+1))
        ax.legend(markerscale=4, fontsize=7, loc="upper right")
        ax.xaxis.set_major_locator(MultipleLocator(base=50))
        ax.yaxis.set_major_locator(MultipleLocator(base=50))
        ax.annotate(panel_labels[k+1], xy=(0.05, 0.9), xycoords="axes fraction", fontweight="bold")

if use_stat_features:
    ax.set_xlim([-8,8])
    ax.set_ylim([-8,8])
    if threeD_plot:
        ax.set_zlim([-8,8])
    fname = "pca_stat_features" + threeD_txt + "_iso_events" + "_omnHistory" + str(omnHistory) + ".png"
else:
    for i in range(1,4):
        ax = axes.flatten()[i]
        ax.set_xlim([-100,100])
        ax.set_ylim([-100,100])
        if threeD_plot:
            ax.set_zlim([-8,8])
    #fname = "pca_raw_features" + threeD_txt + "_iso_events" + "_omnHistory" + str(omnHistory) + ".png"
    fname = "fig_7a-d" + ".png"

# Place panel numbers
#for i in range(axes.flatten())
#x.annotate("(a)", xy=(0.05, 0.9), xycoords="axes fraction", fontweight='bold')

#fig.suptitle(str(df.loc[df.label==1, :].shape[0]) + " Onsets")
fig.savefig("../plots/paper-figures/" + fname, dpi=200, bbox_inches="tight")
#############################################

