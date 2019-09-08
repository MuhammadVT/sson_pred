![](https://github.com/MuhammadVT/sson_pred/tree/master/demo/plots/resnet_cnn.png)

# Deep Learning for Time Series Prediction of Substorm Onset

This is the repository for our paper titled ["A deep learning based approach to forecast the onset of magnetic substorms"](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2019SW002251) published in [Space Weather](https://agupubs.onlinelibrary.wiley.com/journal/15427390).

## Problem Statement
Use 120-minute time history of solar wind bulk speed (Vx), proton number density (Np), and interplanetary magnetic field (IMF) components (Bx, By, Bz) to predict the occurrence probability of substorm onset.

## Demo for this project
A short summary of this work can be found [here](https://github.com/MuhammadVT/sson_pred/blob/master/demo/prediction_of_Aurora_brightening.ipynb).

## Installation
The codes in this project are develped in Ubuntu 16.04.3 LTS.
Here are the instructions for how to setup an conda environmet and execute the codes:

Use the Terminal or an Anaconda Prompt for the following steps.

#### Create the environment from the environment.yml file:

*conda env create -f environment.yml*

#### Activate the new environment:

Windows: *activate sson_pred*

macOS and Linux: *source activate sson_pred*

#### Verify that the new environment was installed correctly:

*conda list*

## Reference
This work can be cited as:

Maimaiti, M., Kunduri, B., Ruohoniemi, J. M., Baker, J. B. H., and House, L. L.. ( 2019), A deep learning based approach to forecast the onset of magnetic substorms. Space Weather, 17. https://doi.org/10.1029/2019SW002251



