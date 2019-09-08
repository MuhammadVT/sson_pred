import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

def plot_param_importance(input_params, metrics_dict):
    """
    Plots a heatmap of metrics for a given set of input parameters

    Arguments:
    input_params -- a list of input parameters
    metrics_dict -- a distionary with Precision, Recall, and F1-Score as keys.
                    Each key has a list of values which corresponds to the input_params.
    """

    # convert to DF
    df = pd.DataFrame(metrics_dict, index=input_params)
    #df.sort_values(by=['$Precision$', '$Recall$'], inplace=True, ascending=False)
    df = df[ ['$Precision$', '$Recall$', '$F1-Score$'] ]
    #df.head()

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,10), sharex=True)
    sns.heatmap(df, annot=True, cmap="GnBu") #Blues, #BuPu, #GnBu, #PuBu
    fig.tight_layout()
    fig.savefig("./plots/param_importance.png", bbox_inches="tight")
