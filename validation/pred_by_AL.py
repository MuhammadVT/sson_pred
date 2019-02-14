import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.metrics import precision_score, recall_score 
plt.style.use("fivethirtyeight")
import pandas as pd
import numpy as np
import datetime as dt
import sqlite3

#pred_file = "../models/trained_models/ResNet/omn_Bx_By_Bz_Vx_Np/"+\
#	    "sml.nBins_1.binTimeRes_30.onsetFillTimeRes_5.omnHistory_120.omnDBRes_1.useSML_True.20190124.151816/"+\
#	    "test_data_pred.csv"
#	    #"all_data_pred.csv"
#	    #"test_data_pred_nosampling.csv"

pred_file = "../models/trained_models/RF/"+\
	    "test_data_pred.csv"
	    #"all_data_pred.csv"


binTimeRes = 30
onsetFillTimeRes = 5
#AL_bins = [[-50, 10000], [-150, -50], [-10000, -150]]
#albin_txt = ["AL>-50", "-150<AL<-50", "AL<-150"]
AL_bins = [[-30, 10000], [-100, -30], [-200, -100], [-300, -200], [-10000, -300]]
albin_txt = ["AL>-30", "-100<AL<-30", "-200<AL<-100", "-300<AL<-200", "AL<-300"]


# Read predicted labels
df_pred = pd.read_csv(pred_file, index_col=0, parse_dates={"onset_time":[0]})
df_pred.index.rename("current_time", inplace=True)

# Load AUL data
dbName = "au_al_ae.sqlite"
DBDir = "../data/sqlite3/"
tabName = "aualae"
stm = df_pred.index.min()
etm = df_pred.index.max()
conn = sqlite3.connect(DBDir + dbName, detect_types = sqlite3.PARSE_DECLTYPES)
# load data to a dataframe
command = "SELECT * FROM {tb} " +\
          "WHERE datetime BETWEEN '{stm}' and '{etm}'"
command = command.format(tb=tabName, stm=stm, etm=etm)
df_al = pd.read_sql(command, conn)
df_al.set_index("datetime", inplace=True)
# Do 5-min average
df_al5 = df_al.resample("5Min").mean()

# Join df_pred with df_al
df = df_pred.join(df_al5) 

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))

P1s = []
R1s = []
P0s = []
R0s = []
for bn in AL_bins:
    df_bn = df.loc[(df.al >= bn[0]) & (df.al < bn[1])]
    print(bn, df_bn.shape)
    P1s.append(precision_score(df_bn.label.values, df_bn.pred_label.values,
                               pos_label=1))
    R1s.append(recall_score(df_bn.label.values, df_bn.pred_label.values,
                            pos_label=1))
    P0s.append(precision_score(df_bn.label.values, df_bn.pred_label.values,
                               pos_label=0))
    R0s.append(recall_score(df_bn.label.values, df_bn.pred_label.values,
                            pos_label=0))

#x = list(range(1, len(AL_bins)+1))
x = list(range(len(AL_bins)))
labels = ["P1", "R1", "P0", "R0"]
colors = ["red", "orange", "blue", "green"] 
for i, y in enumerate([P1s, R1s, P0s, R0s]):
    ax.plot(x, y, linewidth=2, label=labels[i], marker="o", color=colors[i])
    ax.set_ylabel("Precision/Recall")
    ax.set_ylim([-0.05, 1.05])
    ax.legend()

ax.xaxis.set_major_locator(MultipleLocator(1))
xlabels = [x.get_text() for x in ax.xaxis.get_majorticklabels()]
for i, lb in enumerate(albin_txt):
    xlabels[i+1] = lb
ax.set_xticklabels(xlabels)
ax.set_xlabel("Geomagentic Disturbance Level")
plt.tick_params(axis='x', which='major', labelsize=12)

fig.savefig("./RF_test_al.png", dpi=200, bbox_inches="tight")
#fig.savefig("./tmp_al.png", dpi=200, bbox_inches="tight")

