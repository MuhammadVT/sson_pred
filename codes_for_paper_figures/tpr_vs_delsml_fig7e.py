import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
plt.style.use("fivethirtyeight")
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.metrics import precision_score, recall_score
import sqlite3

def load_SML_to_df(stm, etm):
    # Load SM-AUL data
    dbName = "smu_sml_sme.sqlite"
    DBDir = "../../dnn_substorm_onset/data/sqlite3/"
    tabName = "smusmlsme"
    conn = sqlite3.connect(DBDir + dbName, detect_types = sqlite3.PARSE_DECLTYPES)
    # load data to a dataframe
    command = "SELECT * FROM {tb} " +\
              "WHERE datetime BETWEEN '{stm}' and '{etm}'"
    command = command.format(tb=tabName, stm=stm, etm=etm)
    df_sml = pd.read_sql(command, conn)
    df_sml.set_index("datetime", inplace=True)

    return df_sml


def plot_hist(param, ax, color="darkorange", label="logistic", dat_range = (0, 1),
              bins=20, density=False, histtype="step"):

    hist, bin_edges = np.histogram(param, bins=bins, range=dat_range, density=density)
    bin_centers = bin_edges[:-1] + (bin_edges[1]-bin_edges[0])/2.
    ax.plot(bin_centers, hist, "-", color=color, alpha=1.0, linewidth=1.5, label=label)
    #ax.plot(hst, bins=bins, range=dat_range, color=color,
    #        histtype=histtype, density=False, alpha=1.0, linewidth=1.5, label=label)

    return


def plot_prob1_hist(pred_files, colors, labels, fname="prob_dist.png"):

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))
    fig.subplots_adjust(hspace=0.4)
    for i, fl in enumerate(pred_files):
        df = pd.read_csv(fl, index_col=0, parse_dates=["datetime"])
        acc = df.loc[df.label == df.pred_label, :].prob_1.values
        plot_hist(acc, axes[0], color=colors[i], label=labels[i])
        axes[0].set_xlabel("Prob(1) for Correct Prediction")

        err = df.loc[df.label != df.pred_label, :].prob_1.values
        plot_hist(err, axes[1], color=colors[i], label=labels[i])
        axes[1].set_xlabel("Prob(1) for Incorrect Prediction")

        axes[0].axvline(x=0.5, color="red", linestyle="--", linewidth=1., zorder=0)
        axes[1].axvline(x=0.5, color="red", linestyle="--", linewidth=1., zorder=0)
        axes[0].set_ylim([0, 350])
        axes[1].set_ylim([0, 350])
        axes[0].set_ylabel("Count")
        axes[1].set_ylabel("Count")
        axes[0].legend(fontsize="small", loc="upper center")
        axes[1].legend(fontsize="small", loc="upper center")
    fig.savefig("../plots/" + fname, dpi=200, bbox_inches="tight")

    return

def plot_roc(pred_files, colors, labels, fname="roc_curves.png"):

    from sklearn.metrics import roc_curve, auc
    # Overlay ROC curves
    fig, ax = plt.subplots()
    for i, fl in enumerate(pred_files):
        label = labels[i]
        color = colors[i]
        df = pd.read_csv(fl, index_col=0, parse_dates=["datetime"])
        fpr, tpr, thresholds = roc_curve(df.label.values, df.prob_1, pos_label=1)
        roc_auc = auc(fpr, tpr)
        label = label + ", area=%0.2f"%roc_auc
        ax.plot(fpr, tpr, color=color, label=label, linewidth=1.)

        # Mark (FPR, TPR) that corresponds to 0.5 
        idx_05 = (np.abs(thresholds - 0.5)).argmin()
        ax.scatter(fpr[idx_05], tpr[idx_05], color=color, marker="*", s=50)

    ax.set_xlabel("False Positive Rate")
    #ax.set_ylabel("True Positive Rate")
    ax.set_ylabel("Recall")
    ax.legend(fontsize="small", loc="lower right")
    fig.savefig("../plots/" + fname, dpi=200, bbox_inches="tight")

    return


def plot_SML_at_event_time_hist(pred_files, df_sml, colors, labels,
                                fname1="ss_nonss_onset_SML_dist.png", 
                                fname2 = "events_onset_SML_dist.png"):

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))
    fig.subplots_adjust(hspace=0.4)
    dat_range = [-1000, 200]
    for i, fl in enumerate(pred_files):
        df = pd.read_csv(fl, index_col=0, parse_dates=["datetime"])
        df = df.join(df_sml)
        SML_acc = df.loc[df.label == df.pred_label, :].al.values
        plot_hist(SML_acc, axes[0], color=colors[i], label=model_labels[i], dat_range=dat_range)
        axes[0].set_xlabel("SML for Correct Prediction")

        SML_err = df.loc[df.label != df.pred_label, :].al.values
        plot_hist(SML_err, axes[1], color=colors[i], label=model_labels[i], dat_range=dat_range)
        axes[1].set_xlabel("SML for Incorrect Prediction")

        axes[0].set_ylim([0, 1400])
        axes[1].set_ylim([0, 1400])
        axes[0].set_ylabel("Count")
        axes[1].set_ylabel("Count")
        axes[0].legend(fontsize="small", loc="best")
        axes[1].legend(fontsize="small", loc="best")
    
    #axes[0].set_title("SS+Non-SS (" + str(df.shape[0]) + " Events)")
    fig.savefig("../plots/" + fname1, dpi=200, bbox_inches="tight")

    # Plot histograms of SML at onset time of all events
    fig, ax = plt.subplots()
    plot_hist(df.al.values, ax, dat_range=dat_range)
    ax.set_xlabel("SML for Both Classes")
    ax.set_ylabel("Count")
    ax.set_ylim([0, 1400])
    ax.set_title("SS+Non-SS (" + str(df.shape[0]) + " Events)")

    fig.savefig("../plots/" + fname2, dpi=200, bbox_inches="tight")

    return

def calc_delSML_delSMLratio(pred_files, df_sml):
    # Calculated delSML and delSML ration and returns a dataframe 
    # Which also includes the onset datetime

    df_fl0 = pd.read_csv(pred_files[0], index_col=0, parse_dates=["datetime"])
    df_onset = df_fl0.loc[df_fl0.label==1, ["label"]]

    # Loop through each event and calculate delSML
    deltime_after_onset = 30
    SML = []
    delSML = []
    delSMLratio = []
    event_dtms = df_onset.index.to_pydatetime()
    for i, dtm in enumerate(event_dtms):
        stm = dtm
        etm = dtm + dt.timedelta(minutes=deltime_after_onset)
        df_tmp = df_sml.loc[stm:etm, :]
        df_tmp.sort_index(inplace=True)
        if (not df_tmp.empty) and (df_tmp.shape[0] == (deltime_after_onset+1)):
            dff = df_tmp.al.min() - df_tmp.iloc[0, :].al
            ratio = dff / df_tmp.iloc[0, :].al
            SML.append(df_tmp.iloc[0, :].al)
            delSML.append(dff)
            delSMLratio.append(ratio)
        else:
            SML.append(np.nan)
            delSML.append(np.nan)
            delSMLratio.append(np.nan)
    df_onset.loc[:, "SML"] = SML
    df_onset.loc[:, "delSML"] = delSML
    df_onset.loc[:, "delSMLratio"] = delSMLratio
    df_onset.drop(labels="label", axis=1, inplace=True)

    return df_onset


def plot_ss_delSML_dist(pred_files, df_sml, df_onset, colors, labels,
                        fname1="ss_delSML_dist_by_pred1_all_iso.png", 
                        fname2 = "ss_delSML_dist_all_iso.png"):

    # Plot the delSML dist. for onsets
    fig, axes = plt.subplots(nrows=2, ncols=1,
                             figsize=(6, 8))
    fig.subplots_adjust(hspace=0.4)
    dat_range = (-1000, 0)
    delSML_acc = []
    delSML_err = []
    for i, fl in enumerate(pred_files):
        df = pd.read_csv(fl, index_col=0, parse_dates=["datetime"])
        df1 = df.loc[df.label==1, :]
        delSML_acc = df_onset.loc[df1.prob_1 > 0.5, :].delSML.values
        delSML_err = df_onset.loc[df1.prob_1 <= 0.5, :].delSML.values
        
        plot_hist(delSML_acc, axes[0], color=colors[i], label=labels[i], dat_range=dat_range)
        axes[0].set_xlabel("delSML for Correct Prediction") 
        
        plot_hist(delSML_err, axes[1], color=colors[i], label=labels[i], dat_range=dat_range)
        axes[1].set_xlabel("delSML for Incorrect Prediction")

    axes[0].set_ylim([0, 1800])
    axes[1].set_ylim([0, 1800])
    #axes[0].set_ylim([0, 350])
    #axes[1].set_ylim([0, 350])
    axes[0].set_ylabel("Count")
    axes[1].set_ylabel("Count")
    axes[0].legend(fontsize="small", loc="best")
    axes[1].legend(fontsize="small", loc="best")
    axes[0].set_title("Substorms (" + str(df_onset.shape[0]) + " Events)")

    fig.savefig("../plots/" + fname1, dpi=200, bbox_inches="tight")

    # Plot histograms of SML at onset time of all events
    fig, ax = plt.subplots()
    plot_hist(df_onset.delSML.values, ax, dat_range=dat_range)
    ax.set_xlabel("delSML for Substorm Onset")
    ax.set_ylabel("Count")
    #ax.set_ylim([0, 1400])

    fig.savefig("../plots/" + fname2, dpi=200, bbox_inches="tight")

    return

def plot_ss_delSMLratio_dist(pred_files, df_sml, df_onset, colors, labels,
                             fname1="ss_delSMLratio_dist_by_pred1_all_iso.png", 
                             fname2 = "ss_delSMLratio_dist_all_iso.png"):

    # Plot the delSML dist. for onsets
    fig, axes = plt.subplots(nrows=2, ncols=1,
                             figsize=(6, 8))
    fig.subplots_adjust(hspace=0.4)
    dat_range = (0, 10)
    delSMLratio_acc = []
    delSMLratio_err = []
    for i, fl in enumerate(pred_files):
        df = pd.read_csv(fl, index_col=0, parse_dates=["datetime"])
        df1 = df.loc[df.label==1, :]
        delSMLratio_acc = df_onset.loc[df1.prob_1 > 0.5, :].delSMLratio.values
        delSMLratio_err = df_onset.loc[df1.prob_1 <= 0.5, :].delSMLratio.values
        
        plot_hist(delSMLratio_acc, axes[0], color=colors[i], label=labels[i], dat_range=dat_range)
        axes[0].set_xlabel("delSMLratio for Correct Prediction") 
        
        plot_hist(delSMLratio_err, axes[1], color=colors[i], label=labels[i], dat_range=dat_range)
        axes[1].set_xlabel("delSMLratio for Incorrect Prediction")

    axes[0].set_ylim([0, 1600])
    axes[1].set_ylim([0, 1600])
    #axes[0].set_ylim([0, 350])
    #axes[1].set_ylim([0, 350])
    axes[0].set_ylabel("Count")
    axes[1].set_ylabel("Count")
    axes[0].legend(fontsize="small", loc="best")
    axes[1].legend(fontsize="small", loc="best")
    axes[0].set_title("Substorms (" + str(df_onset.shape[0]) + " Events)")

    fig.savefig("../plots/" + fname1, dpi=200, bbox_inches="tight")

    # Plot histograms of SML at onset time of all events
    fig, ax = plt.subplots()
    plot_hist(df_onset.delSMLratio.values, ax, dat_range=dat_range)
    ax.set_xlabel("delSMLratio for Substorm Onset")
    ax.set_ylabel("Count")
    ax.set_ylim([0, 1600])
    #ax.set_ylim([0, 350])

    fig.savefig("../plots/" + fname2, dpi=200, bbox_inches="tight")

    return

#def plot_tpr_fnr_by_delSML(pred_files, df_sml, df_onset, colors, labels,
#                                    fname1="ss_tpr_fnr_by_delSML_all_iso.png", 
#                                    fname2 = "ss_tpr_fnr_by_delSMLratio_all_iso.png"):
#
#    # Plot onset True Positive and False Negative for different delSML range
#    delSML_thresholds = [-50, -100, -150, -200, -250, -300, -350, -400]
#    delSMLratio_thresholds = [1, 2, 3, 4, 5]
#    for k in range(2):
#        if k ==0:
#            thresholds = delSML_thresholds
#        if k == 1:
#            thresholds = delSMLratio_thresholds
#        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True,
#                                 figsize=(6, 8))
#        fig.subplots_adjust(hspace=0.4)
#        for i, fl in enumerate(pred_files):
#            df = pd.read_csv(fl, index_col=0, parse_dates=["datetime"])
#            df1 = df.loc[df.label==1, :]
#            df1 = df1.join(df_onset)
#            delSML_tpr = []
#            delSML_fnr = []
#            for var in thresholds: 
#                if k ==0:
#                    df1_tmp = df1.loc[df1.delSML < var,:]
#                if k ==1:
#                    df1_tmp = df1.loc[df1.delSMLratio > var,:]
#                delSML_tpr.append((df1_tmp.pred_label>0.5).mean())
#                delSML_fnr.append((df1_tmp.pred_label<=0.5).mean())
#            
#            axes[0].plot(thresholds, delSML_tpr, color=colors[i], label=model_labels[i], linewidth=1.5)
#            axes[1].plot(thresholds, delSML_fnr, color=colors[i], label=model_labels[i], linewidth=1.5)
#            
#        axes[0].set_ylim([0.5, 1])
#        axes[1].set_ylim([0, 0.5])
#        axes[0].set_ylabel("True Positive Rate")
#        axes[1].set_ylabel("False Negative")
#        axes[0].legend(fontsize="small", loc="best")
#        axes[1].legend(fontsize="small", loc="best")
#        #axes[0].set_title("Substorms (" + str(df_onset.shape[0]) + " Events)")
#        if k == 0:
#            #axes[0].set_xlabel("delSML") 
#            axes[1].set_xlabel("delSML")
#            fig.savefig("../plots/" + fname1, dpi=200, bbox_inches="tight")
#        if k == 1:
#            axes[0].set_xlabel("delSMLratio") 
#            axes[1].set_xlabel("delSMLratio")
#            fig.savefig("../plots/" + fname2, dpi=200, bbox_inches="tight")
#
#    return

def plot_tpr_fnr_by_delSML(pred_files, df_sml, df_onset, colors, labels,
                                    fname1="ss_tpr_fnr_by_delSML_all_iso.png", 
                                    fname2 = "ss_tpr_fnr_by_delSMLratio_all_iso.png"):

    # Plot onset True Positive and False Negative for different delSML range
    delSML_thresholds = [0, -50, -100, -150, -200, -250, -300, -350, -400, -450, -500, -550, -600, -650, -700]
    delSMLratio_thresholds = [1, 2, 3, 4, 5]
    thresholds = delSML_thresholds
    fig, ax = plt.subplots(figsize=(8,6))
    for i, fl in enumerate(pred_files):
        df = pd.read_csv(fl, index_col=0, parse_dates=["datetime"])
        df1 = df.loc[df.label==1, :]
        df1 = df1.join(df_onset)
        delSML_tpr = []
        delSML_fnr = []
        for var in thresholds: 
            df1_tmp = df1.loc[df1.delSML < var,:]
            delSML_tpr.append((df1_tmp.pred_label>0.5).mean())
            print("# Substorms for delSML_threshold = " + str(var) + " is " + str(df1_tmp.shape[0]))

        ax.plot(thresholds, delSML_tpr, color=colors[i], label=model_labels[i], linewidth=1.5)
        
    ax.set_ylim([0.5, 1])
    #ax.set_ylabel("True Positive Rate")
    ax.set_ylabel("Recall")
    #ax.legend(fontsize="small", loc="best")
    #ax.set_title("Substorms (" + str(df_onset.shape[0]) + " Events)")
    ax.set_xlabel(r"$\Delta SML$ Threshold") 
    ax.annotate("(e)", xy=(0.05, 0.95), xycoords="axes fraction", fontweight="bold")

    fig.savefig("../plots/paper-figures/" + fname1, dpi=200, bbox_inches="tight")
    #fig.savefig("../plots/paper-figures/" + fname1[:-4] + ".pdf", format="pdf", bbox_inches="tight")

    return


if __name__ == "__main__":
    # Set input file names
    file_dir = "../data/pred_files/"
    pred_files = ["resnet"+\
                  "_test_data_pred.csv"]
    model_labels = ["RN"]


    pred_files = [file_dir + x for x in pred_files]
    colors = ["black", "blue", "green", "darkorange", "red"]

    # Load SML index
    df_events = pd.read_csv(pred_files[0], index_col=0, parse_dates=["datetime"])
    stm = df_events.index.min()
    etm = df_events.index.max()
#    stm = dt.datetime(1997, 1, 1)
#    etm = dt.datetime(1997, 2, 1)
    df_sml = load_SML_to_df(stm, etm)

    #############################
    # Overlay ROC curves
#    plot_roc(pred_files, colors, model_labels, fname="roc_curves.png")

    #############################
    # Plot prob. histograms 
#    plot_prob1_hist(pred_files, colors, model_labels, fname="prob_dist.png")

#    #############################
    # Calc delSML and delSMLratio
    df_onset = calc_delSML_delSMLratio(pred_files, df_sml)
    #############################

#    #############################
#    # Plot histograms of SML at the time of onset/non-onset
#    plot_SML_at_event_time_hist(pred_files, df_sml, colors, model_labels,
#                                fname1="ss_nonss_onset_SML_dist_all_data.png", 
#                                fname2 = "events_onset_SML_dist_all_data.png")


#    #############################
#    plot_ss_delSML_dist(pred_files, df_sml, df_onset, colors, model_labels,
#                        fname1="ss_delSML_dist_by_pred1_all_iso.png", 
#                        fname2 = "ss_delSML_dist_all_iso.png")
#
#   #############################
#    plot_ss_delSMLratio_dist(pred_files, df_sml, df_onset, colors, model_labels,
#                             fname1="ss_delSMLratio_dist_by_pred1_all_iso.png", 
#                             fname2 = "ss_delSMLratio_dist_all_iso.png")

    #############################
    # Plot onset True Positive and False Negative for different delSML range
    plot_tpr_fnr_by_delSML(pred_files, df_sml, df_onset, colors, model_labels,
                                    fname1="fig7e.png", 
                                    fname2 = "test_ss_tpr_fnr_by_delSMLratio_test_iso.png")


