import pandas
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

inpIMF = ["Bx, By, Bz, Vx, Np",
          "Bx, By, Bz, Vx",
          "By, Bz, Vx, Np",
          "Bx, Bz, Vx, Np",
          "Bx, By, Bz, Np",
          "Bx, By, Vx, Np",
          "By, Bz, Vx",
          "Bz, Vx",
          "Bz", "Vx", "By", "Bx", "Np"]

prDict = {}
prDict["$Precision$"] = [ 0.75, 0.75, 0.74, 0.74, 0.73, 0.69, 0.75, 0.75, 0.71, 0.69, 0.64, 0.65, 0.58]
prDict["$Recall$"] = 	[ 0.73, 0.71, 0.73, 0.73, 0.67, 0.58, 0.70, 0.70, 0.68, 0.51, 0.49, 0.43, 0.35]
prDict["$F1-Score$"] =  [ 0.74, 0.73, 0.74, 0.74, 0.70, 0.63, 0.72, 0.72, 0.70, 0.58, 0.55, 0.52, 0.44]
# convert to DF
prDF = pandas.DataFrame(prDict, index=inpIMF)
#prDF.sort_values(by=['$Precision$', '$Recall$'], inplace=True, ascending=False)
prDF = prDF[ ['$Precision$', '$Recall$', '$F1-Score$'] ]
prDF.head()

plt.style.use("fivethirtyeight")
fig, axes = plt.subplots(nrows=1, ncols=1,\
                    figsize=(10,8), sharex=True)
sns.heatmap(prDF, annot=True, cmap="GnBu") #Blues, #BuPu, #GnBu, #PuBu
fig.tight_layout()
fig.savefig("../plots/paper-figures/fig_8.png", bbox_inches="tight")
