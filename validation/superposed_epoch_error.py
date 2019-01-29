import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
plt.style.use("fivethirtyeight")
import pandas as pd
import numpy as np
import datetime as dt

sml_onset_file =  "../data/" +\
                  "20190103-22-53-substorms.csv"

pred_file = "../models/trained_models/ResNet/omn_Bx_By_Bz_Vx_Np/"+\
	    "sml.nBins_1.binTimeRes_30.onsetFillTimeRes_5.omnHistory_120.omnDBRes_1.useSML_True.20190124.151816/"+\
	    "test_data_pred.csv"
	    #"all_data_pred.csv"
	    #"test_data_pred_nosampling.csv"

binTimeRes = 30
onsetFillTimeRes = 5
before_onset = 1 * 60     # minutes
after_onset = 1 * 5     # minutes
onset_gap_minlim = 1 * 19    # minutes

# Read predicted labels
df_pred = pd.read_csv(pred_file, index_col=0, parse_dates={"onset_time":[0]})
df_pred.index.rename("current_time", inplace=True)

# Read substorm onsets
df_onset = pd.read_csv(sml_onset_file, parse_dates={"onset_time":[0]})
df_onset.set_index("onset_time", inplace=True)

# Select onsets within the time interval of interest
df_onset = df_onset.loc[df_pred.index.min():, :]

# Loop through each onset
errs = []
accs = []
tps = []
tns = []
fps = []
fns = []
del_dtms = []
onset_dtms = df_onset.index.to_pydatetime()
for i, dtm in enumerate(onset_dtms[:-1]):
    if i == 0:
        continue
    dff_before = (dtm - onset_dtms[i-1]).total_seconds()/60.
    dff_after = (onset_dtms[i+1] - dtm).total_seconds()/60.
    if (dff_before > onset_gap_minlim) and (dff_after > onset_gap_minlim):
        stm = dtm - dt.timedelta(minutes=before_onset)
        etm = dtm + dt.timedelta(minutes=after_onset)
        df_tmp = df_pred.loc[stm:etm, :]
        df_tmp.sort_index(inplace=True)
        if not df_tmp.empty:
            cdtms = df_tmp.index.to_pydatetime().tolist()
            true_labels = df_tmp.loc[:, "label"].values
            pred_labels = df_tmp.loc[:, "pred_label"].values
            del_dtms.extend([(x-dtm).total_seconds()/60. for x in cdtms])
            errs.extend(0 + (true_labels!=pred_labels))
            accs.extend(0 + (true_labels==pred_labels))
            tps.extend(0 + ((true_labels*pred_labels) == 1))
            tns.extend(0 + ((true_labels+pred_labels) == 0))
            fps.extend(0 + ((true_labels-pred_labels) == -1))
            fns.extend(0 + ((true_labels-pred_labels) == 1))

df = pd.DataFrame(data={"errs":errs, "accs":accs, "tps":tps, "tns":tns,\
                        "fps":fps, "fns":fns}, index=del_dtms)
dfg = df.groupby(by=df.index).sum()
tm = dfg.index.values
acc_num = dfg.accs.values
err_num = dfg.errs.values
tp_num = dfg.tps.values
tn_num = dfg.tns.values
fp_num = dfg.fps.values
fn_num = dfg.fns.values

acc = np.divide(acc_num, (acc_num+err_num))
err = np.divide(err_num, (acc_num+err_num))
#precision_zero = 1.* tn_num / (tn_num + fn_num)
#recall_zero = 1.* tn_num / (tn_num + fp_num)
#precision_one = 1.* tp_num / (tp_num + fp_num)
#recall_one = 1.* tp_num / (tp_num + fn_num)

#fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True)
#fig.subplots_adjust(hspace=0.3, wspace=0.5)

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8,8))
fig.subplots_adjust(wspace=0.4)


#axes[0, 0].plot(tm, precision_zero, linewidth=2)
#axes[0, 0].set_ylabel("precision_0")
#axes[0, 0].set_ylim([-0.05, 1.05])
#axes[0, 1].plot(tm, recall_zero, linewidth=2)
#axes[0, 1].set_ylabel("recall_0")
#axes[0, 1].set_ylim([-0.05, 1.05])
#
#axes[1, 0].plot(tm, precision_one, linewidth=2)
#axes[1, 0].set_ylabel("precision_1")
#axes[1, 0].set_ylim([-0.05, 1.05])
#axes[1, 1].plot(tm, recall_one, linewidth=2)
#axes[1, 1].set_ylabel("recall_1")
#axes[1, 1].set_ylim([-0.05, 1.05])
#
#axes[2, 0].plot(tm, acc, linewidth=2)
#axes[2, 0].set_ylabel("Accuracy")
#axes[2, 0].set_ylim([-0.05, 1.05])
#axes[2, 1].plot(tm, err, linewidth=2)
#axes[2, 1].set_ylabel("Error")
#axes[2, 1].set_ylim([-0.05, 1.05])

axes[0].plot(tm, acc, linewidth=2)
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim([0.45, 1.05])
#axes[0].set_ylim([0.0, 1.05])
axes[1].plot(tm, err, linewidth=2)
axes[1].set_ylabel("Error")
axes[1].set_xlabel("Relative Time [Minutes]")
axes[1].set_ylim([-0.05, 0.55])
#axes[1].set_ylim([0.0, 1.05])

for ax in axes.flatten():
    #ax.axvline(x=binTimeRes, color="r", linestyle="--", linewidth=1.5)
    ax.axvline(x=-binTimeRes, color="r", linestyle="--", linewidth=1.5)

fig.savefig("./tmp_test.png", dpi=200, bbox_inches="tight")
#fig.savefig("./tmp.png", dpi=200, bbox_inches="tight")

