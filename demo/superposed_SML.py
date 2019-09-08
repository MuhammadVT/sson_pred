import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
import pandas as pd
import numpy as np
import datetime as dt
import sqlite3

def plot_SML(sdate=dt.datetime(1997, 1, 1),
             edate=dt.datetime(2017, 12, 30),
             onset_gap_minlim=10,
             before_onset=120,
             after_onset=60,
             dbName="au_al_ae.sqlite",
             DBDir="../data/sqlite3/",
             tabName = "aualae",
             sml_onset_file =  "../data/omn_Bx_By_Bz_Vx_Np/" +\
             "sml_nBins_1.binTimeRes_60.onsetFillTimeRes_5."+\
             "onsetDelTCutoff_4.omnHistory_120.omnDBRes_1."+\
             "imfNormalize_True.shuffleData_False.dateRange_19970101_20180101.iso.csv"):

    """Plots superposed SML indices centered at the substorm onsets. Returns None
    
    keyword arguments:
    sdate -- the start datetime
    edate -- the end datetime
    onset_gap_minlim -- minutes between two adjecent substorms
    before_onset -- minutes to plot before the onset
    after_onset -- minutes to plot after the onset
    dbName -- indices sqlite3 file 
    DBDir -- data directory
    sml_onset_file -- the file that stores the data points

    """

    # Read substorm onsets
    df_event_all = pd.read_csv(sml_onset_file, parse_dates={"ut_time":[0]}, index_col="ut_time")
    df_event_all = df_event_all.loc[sdate:edate, :]

    # Balance the two classes
    df_ss = df_event_all.loc[df_event_all.label==1, :]
    df_nonss_all = df_event_all.loc[df_event_all.label==0, :]
    np.random.seed(1)
    df_nonss = df_nonss_all.iloc[np.random.choice(df_nonss_all.shape[0], df_ss.shape[0], replace=False),:]
    df_event = pd.concat([df_ss, df_nonss])

    # Load AUL data
    conn = sqlite3.connect(DBDir + dbName, detect_types = sqlite3.PARSE_DECLTYPES)
    # load data to a dataframe
    command = "SELECT * FROM {tb} " +\
              "WHERE datetime BETWEEN '{stm}' and '{etm}'"
    command = command.format(tb=tabName, stm=sdate, etm=edate)
    df_al = pd.read_sql(command, conn)
    df_al.set_index("datetime", inplace=True)

    # Resample at 1 min and then fill the missing data
    df_al.resample("1Min" ).ffill().bfill()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    # Loop through each event 
    for l in [0,1]:
        rel_dtms = []
        ALs = []
        event_dtms = df_event.loc[df_event.label==l, :].index.to_pydatetime()
        for i, dtm in enumerate(event_dtms):
            stm = dtm - dt.timedelta(minutes=before_onset)
            etm = dtm + dt.timedelta(minutes=after_onset)
            df_tmp = df_al.loc[stm:etm, :]
            df_tmp.sort_index(inplace=True)
            if (not df_tmp.empty) and (df_tmp.shape[0] == (before_onset+after_onset+1)):
                cdtms = df_tmp.index.to_pydatetime().tolist()
                rel_dtms.append([(x-dtm).total_seconds()/60. for x in cdtms])
                ALs.append(df_tmp.al.values)

        if l == 0:
            nonss_num = len(ALs)
        if l == 1:
            ss_num = len(ALs)

        idxs = ALs
        idx_labels = "SML"
        for k in range(len(ALs)):
            axes[l].plot(rel_dtms[k], idxs[k], linewidth=0.3, color="gray")
    
        # Plot the average
        axes[l].plot(range(-before_onset, after_onset+1,1), np.array(idxs).mean(axis=0), linewidth=1.0, color="k")
    
        axes[l].axvline(x=0, linewidth=1.0, color="r")
        axes[l].set_ylabel(idx_labels + " [nT]")
        
    # Set tiles and labels
    #axes[0].set_title("Non-Substorms (" + str(nonss_num) + " Events)", )
    #axes[1].set_title("Substorms (" + str(ss_num) + " Events)")
    axes[0].set_title("Non-Substorms Events)", )
    axes[1].set_title("Substorms Events)")
    axes[0].set_xlabel("Relative Time [Minutes]")
    axes[1].set_xlabel("Relative Time [Minutes]")

    # Set axis limits
    axes[0].set_ylim([-3000, 200])
    axes[1].set_ylim([-3000, 200])
        
    fig_name = "superposed_SML"
    fig.savefig("./plots/" + fig_name + ".png", dpi=200, bbox_inches="tight")

