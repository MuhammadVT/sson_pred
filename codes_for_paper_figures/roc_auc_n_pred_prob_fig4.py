import pandas
import datetime
import seaborn as sns
from sklearn.utils import resample
import matplotlib.pyplot as plt
import numpy

from sklearn.metrics import roc_curve, auc

predDF = pandas.read_csv("../data/pred_files/resnet_test_data_pred.csv",\
                     header=0, parse_dates=["datetime"])
predDF['hour'] = pandas.DatetimeIndex(predDF['datetime']).hour
# to calculate roc we'll need two bins for two classes
predDF["label_0"] = [ 1 if x==0 else 0 for x in predDF["label"] ]
predDF["label_1"] = [ 1 if x==1 else 0 for x in predDF["label"] ]
actBinLabArr = predDF[["label_0", "label_1"]].values
predProbArr = predDF[["prob_0", "prob_1"]].values

binLab=1
predDF.head()

plt.style.use("fivethirtyeight")
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 6))
plt.subplots_adjust(wspace=0.3)

######### ROC_AUC ##########
ax = axes[0]
# get the roc and auc
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
thresholds = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], thresholds[i] = roc_curve(actBinLabArr[:, i], predProbArr[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])

print(roc_auc)

ax.plot([0, 1], [0, 1], color="k", linestyle='--', linewidth=2.0)
ax.plot(fpr[binLab], tpr[binLab], color="k", linewidth=3.0, label='ROC curve (area = %0.2f)' % roc_auc[binLab])

# Mark (FPR, TPR) that corresponds to 0.5 
idx_05 = (numpy.abs(thresholds[binLab] - 0.5)).argmin()
ax.plot([fpr[binLab][idx_05], fpr[binLab][idx_05]], [0, tpr[binLab][idx_05]], color="k", linestyle="--", linewidth=1.0)
ax.plot([0, fpr[binLab][idx_05]], [tpr[binLab][idx_05], tpr[binLab][idx_05]], color="k", linestyle="--", linewidth=1.0)
ax.scatter(fpr[binLab][idx_05], tpr[binLab][idx_05], color="k", marker="x", s=100, linewidth=3.0)

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver operating characteristic curve', fontsize=14)
ax.legend(loc="lower right")

ax.annotate("(a)", xy=(0.05, 0.9), xycoords="axes fraction", fontweight='bold')

######### Pred_Prob ##########
def pred_bin_out(row, nBins):
    """
    Given the prediction label, get the actual
    output in bins by converting the label into
    binary representation. For ex, label 2 would
    convert to 10 and 5 to 101 and so on.
    """
    # Note we need the binary format to be consistent
    # with the actual labels, i.e., it depends on the 
    # number of bins. For example, 2 could be 10 or 010.
    binFormtStr = '{0:0' + str(nBins) + 'b}'
    predBinStr = binFormtStr.format(row["pred_label"])
    # Now add these into different pred bins
    for _n, _pb in enumerate(predBinStr):
        row["pbin_" + str(_n)] = int(_pb)
    if row["label"] == 0:
        if row["pred_label"] == 0:
            predType = "TN"
        else:
            predType = "FP"
    if row["label"] == 1:
        if row["pred_label"] == 1:
            predType = "TP"
        else:
            predType = "FN"
    row["pred_type"] = predType
    return row
ax = axes[1]
nBins = 1
# get the columns showing the bin predictions
filterCols = [ col for col in predDF\
             if col.startswith('bin') ]
predDF = predDF.apply( pred_bin_out, args=(nBins,),\
                      axis=1 )
predDF.head()

# setup the bins
bins = numpy.arange(-0.1,1.1, 0.1)[:-1]
# plot label 0
pltDF = predDF[ predDF["label"] == 0]
hist, binEdges = numpy.histogram( pltDF["prob_1"].values,bins=bins )
ax.plot( binEdges[1:], hist, linewidth=3.0, color="#339966")
# plot label 1
pltDF = predDF[ predDF["label"] == 1]
hist, binEdges = numpy.histogram( pltDF["prob_1"].values,bins=bins )
ax.plot( binEdges[1:], hist , linewidth=3.0, color="#e60073")
# plot the 0.5 mark to seperate TPs, FPs, TNs and FNs
ax.axvline(x=0.5, color="k", linestyle="--", linewidth=2.)
ax.annotate("(b)", xy=(0.9, 0.9), xycoords="axes fraction", fontweight='bold')

ax.set_xlabel("Probability")
ax.set_ylabel('Counts')

#predDF[ (predDF["pred_type"] == "FP") & (predDF["prob_1"] >= 0.6) ].head()

fig.savefig("../plots/paper-figures/fig_4.png", bbox_inches="tight")

