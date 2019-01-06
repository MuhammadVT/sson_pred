import event_plot
import datetime

eventDate = datetime.datetime(2001, 10, 3, 3, 0)
actualLab = []
predLab = []
omnDBDir = "/home/bharat/Documents/data/ss_onset_dataset/data/sqlite3/"
omnDbName = "omni_sw_imf.sqlite"
omnTabName = "imf_sw"
aulDbName = "au_al_ae.sqlite"
aulTabName = "aualae"
smlDbName = "smu_sml_sme.sqlite"
smlTabName = "smusmlsme"
predFname = "/home/bharat/Documents/data/ss_onset_dataset/data/all_data.nBins_2.binTimeRes_30.onsetFillTimeRes_1.onsetDelTCutoff_2.omnHistory_120.omnDBRes_1.shuffleData_True.csv"

esObj = event_plot.EventSummary(eventDate, actualLab, predLab,\
				 omnDBDir, omnDbName, omnTabName, aulDbName, aulTabName,\
				  smlDbName, smlTabName, predFname, nBins=2)
esObj.generate_plot()