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

esObj = event_plot.EventSummary(eventDate, actualLab, predLab,\
				 omnDBDir, omnDbName, omnTabName, aulDbName, aulTabName,\
				  smlDbName, smlTabName)