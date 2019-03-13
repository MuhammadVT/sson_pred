import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
plt.style.use("fivethirtyeight")
import pandas as pd
import numpy as np
import datetime as dt
import sqlite3

sml_onset_file =  "../data/" +\
                  "20190103-22-53-substorms.csv"

before_onset = 1 * 60     # minutes
after_onset = 1 * 60     # minutes
onset_gap_minlim = 20    # minutes

sdate = dt.datetime(1997, 1, 1)
edate = dt.datetime(2017, 12, 30)
#edate = dt.datetime(1999, 1, 1)

# Read substorm onsets
df_onset = pd.read_csv(sml_onset_file, parse_dates={"onset_time":[0]})
df_onset.set_index("onset_time", inplace=True)
df_onset = df_onset.loc[sdate:edate, :]

# Loop through each onset
onset_dtms = df_onset.index.to_pydatetime()
onset_idx_subset = []
for i, dtm in enumerate(onset_dtms[:-1]):
    if i == 0:
        continue
    dff_before = (dtm - onset_dtms[i-1]).total_seconds()/60.
    dff_after = (onset_dtms[i+1] - dtm).total_seconds()/60.
    if (dff_before >= onset_gap_minlim) and (dff_after >= onset_gap_minlim):
        onset_idx_subset.append(i)
dfn = df_onset.iloc[onset_idx_subset, :]

#############################################################
# Plot histograms of MLAT and MLT
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8,8))
fig.subplots_adjust(wspace=0.5)
bins = 50
histtype = "bar"
mlt_c0 = [x if x <= 12 else x-24 for x in dfn.MLT.values]
axes[0].hist(dfn.MLAT, bins=bins, histtype=histtype)
axes[1].hist(mlt_c0, bins=bins, histtype=histtype)

axes[0].set_ylabel("Count")
axes[1].set_ylabel("Count")
axes[0].set_xlabel("MLAT")
axes[1].set_xlabel("MLT")
#############################################################

#idxs = [ALs, AUs]
#idx_labels = ["SML", "SMU"]
#for i, ax in enumerate(axes):
#    for j in range(len(ALs)):
#        ax.plot(rel_dtms[j], idxs[i][j], linewidth=0.3, color="gray")
#
#    # Plot the average
#    ax.plot(range(-before_onset, after_onset+1,1), np.array(idxs[i]).mean(axis=0), linewidth=1.0, color="k")
#
#    ax.axvline(x=0, linewidth=1.0, color="r")
#    ax.set_ylabel(idx_labels[i] + " [nT]")
#
#axes[0].set_title("Superposed Substorm Onsets (" + str(len(ALs)) + " Events)", )
#axes[-1].set_xlabel("Relative Time [Minutes]")
#
#axes[1].set_ylim([-50, 1000])
#if onset_gap_minlim <= 30: 
#    axes[0].set_ylim([-3000, 50])
#else:
#    axes[0].set_ylim([-2000, 50])

fig_name = "substorm_loc_hist.onset_sep_" + str(onset_gap_minlim) + "." +\
           "dtm_" + sdate.strftime("%Y%m%d") + "_" + edate.strftime("%Y%m%d") 
fig.savefig("./plots/" + fig_name + ".png", dpi=200, bbox_inches="tight")

