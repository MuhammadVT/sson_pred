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
onset_gap_minlim = 120    # minutes

sdate = dt.datetime(1997, 1, 1)
edate = dt.datetime(2017, 12, 30)
#edate = dt.datetime(1999, 1, 1)

# Read substorm onsets
df_onset = pd.read_csv(sml_onset_file, parse_dates={"onset_time":[0]})
df_onset.set_index("onset_time", inplace=True)
df_onset = df_onset.loc[sdate:edate, :]

# Load AUL data
dbName = "au_al_ae.sqlite"
DBDir = "../data/sqlite3/"
tabName = "aualae"
conn = sqlite3.connect(DBDir + dbName, detect_types = sqlite3.PARSE_DECLTYPES)
# load data to a dataframe
command = "SELECT * FROM {tb} " +\
          "WHERE datetime BETWEEN '{stm}' and '{etm}'"
command = command.format(tb=tabName, stm=sdate, etm=edate)
df_al = pd.read_sql(command, conn)
df_al.set_index("datetime", inplace=True)

# Resample at 1 min and then fill the missing data
df_al.resample("1Min" ).ffill().bfill()

# Loop through each onset
rel_dtms = []
ALs = []
AUs = []
onset_dtms = df_onset.index.to_pydatetime()
for i, dtm in enumerate(onset_dtms[:-1]):
    if i == 0:
        continue
    dff_before = (dtm - onset_dtms[i-1]).total_seconds()/60.
    dff_after = (onset_dtms[i+1] - dtm).total_seconds()/60.
    if (dff_before >= onset_gap_minlim) and (dff_after >= onset_gap_minlim):
        stm = dtm - dt.timedelta(minutes=before_onset)
        etm = dtm + dt.timedelta(minutes=after_onset)
        df_tmp = df_al.loc[stm:etm, :]
        df_tmp.sort_index(inplace=True)
        if (not df_tmp.empty) and (df_tmp.shape[0] == (before_onset+after_onset+1)):
            cdtms = df_tmp.index.to_pydatetime().tolist()
            rel_dtms.append([(x-dtm).total_seconds()/60. for x in cdtms])
            ALs.append(df_tmp.al.values)
            AUs.append(df_tmp.au.values)

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8,8))
fig.subplots_adjust(wspace=0.4)

idxs = [ALs, AUs]
idx_labels = ["SML", "SMU"]
for i, ax in enumerate(axes):
    for j in range(len(ALs)):
        ax.plot(rel_dtms[j], idxs[i][j], linewidth=0.3, color="gray")

    # Plot the average
    ax.plot(range(-before_onset, after_onset+1,1), np.array(idxs[i]).mean(axis=0), linewidth=1.0, color="k")

    ax.axvline(x=0, linewidth=1.0, color="r")
    ax.set_ylabel(idx_labels[i] + " [nT]")

axes[0].set_title("Superposed Substorm Onsets (" + str(len(ALs)) + " Events)", )
axes[-1].set_xlabel("Relative Time [Minutes]")

axes[0].set_ylim([-2000, 50])
axes[1].set_ylim([-50, 1000])

fig_name = "superposed_SMUL.onset_sep_" + str(onset_gap_minlim) + "." +\
           sdate.strftime("%Y%m%d") + "_" + edate.strftime("%Y%m%d") 
fig.savefig("./" + fig_name + ".png", dpi=200, bbox_inches="tight")




