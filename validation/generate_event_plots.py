import warnings
warnings.filterwarnings("ignore")
import pred_act_cmpr
import datetime
import matplotlib
matplotlib.use('Agg')

omnDBDir = "/home/bharat/Documents/data/ss_onset_dataset/data/sqlite3/"
omnDbName = "omni_sw_imf.sqlite"
omnTabName = "imf_sw"
aulDbName = "au_al_ae.sqlite"
aulTabName = "aualae"
smlDbName = "smu_sml_sme.sqlite"
smlTabName = "smusmlsme"
#predFname = "/home/bharat/Documents/data/ss_onset_dataset/data/all_data.nBins_2.binTimeRes_30.onsetFillTimeRes_1.onsetDelTCutoff_2.omnHistory_120.omnDBRes_1.shuffleData_True.csv"
predFname = "../data/test_data.pred.test"

nBins=1
binTimeRes=60

psObj = pred_act_cmpr.PredSumry(predFname, nBins=nBins, binTimeRes=binTimeRes)
psObj.create_pred_plots(omnDBDir, omnDbName, omnTabName, aulDbName,\
                aulTabName, smlDbName, smlTabName)