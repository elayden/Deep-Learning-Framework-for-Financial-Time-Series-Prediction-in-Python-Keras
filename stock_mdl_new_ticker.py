# -*- coding: utf-8 -*-
"""
stock_mdl_new_ticker.py
Created on Wed Oct 17, 2018
@author: Elliot Layden

Info: A script that loads a previously optimized neural network and predicts 
    data from a new stock/ETF ticker. After evaluating performance, there is
    the option to train the model further on the new ticker.
"""
##############################################################################
# Imports:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import get_ipython
import os
import pickle
from train_stock_mdl import getSlidingWindowData, calcPerformance, plotPredictions, saveBest

# Functions:

def loadData(fpath, final_fname, csvfile):
    # Load model:
    f = open(final_fname, 'rb')
    (bestModel, scaler, testResults) = pickle.load(f)
    f.close()
    
    # Load new data:
    os.chdir(fpath)
    data = pd.read_csv(csvfile) # DL_data2.csv
    data.fillna(0, inplace=True)
    data = data.values
    today = data[-lookBack:,:]
    
    # Get sliding window data (if lookBack > 1 day) and compound returns (if returnPeriod > 1 day):
    x, y = getSlidingWindowData(data, lookBack, returnPeriod)
    
    # Scale full dataset:
    fullData = np.hstack((x,y))
    fullDataScaled = scaler.transform(fullData)
    return bestModel, scaler, fullData, fullDataScaled, today

def predictNew(bestModel, scaler, fullData, fullDataScaled):
    # Predict and unscale:
    yHat = bestModel.predict(fullDataScaled[:,:-1])
    yHat = scaler.inverse_transform(np.hstack((fullDataScaled[:,:-1], yHat)))[:,-1]
    yHat = yHat.reshape((yHat.shape[0],1))   
        
    testResults = calcPerformance(fullData, yHat)
    print("New Data:  Annualized = %.2f%%, vBuy&Hold = %.2f%%, #Trades = %.0f, "
          "%%Profitable = %.1f%%, Profit Factor = %.2f, Max Drawdown = %.2f%%, "
          "RMSE_improvement = %.2f%%" % (testResults[0], testResults[1],
          testResults[2], testResults[3], testResults[4], 
          testResults[5], testResults[6]))
    
    # Plot test data predictions:
    plotPredictions(fullData, yHat, 'Predictions')
    
    return testResults
    
# Further train model on new data:

def trainOnNew(bestModel, scaler, fullData, fullDataScaled):
    mdlVerbose=0
    if verbose==2:
        mdlVerbose=2

    history = bestModel.fit(fullDataScaled[:,:-1], fullDataScaled[:,-1], epochs=trainingEpochs, batch_size=fullDataScaled.shape[0], verbose=mdlVerbose, shuffle=True)
    
    # Plot training loss:
    if figures0:
        get_ipython().run_line_magic('matplotlib', 'qt')
        plt.plot(history.history['loss'], label='train');
        plt.legend(); plt.show(); plt.xlabel('Epoch'); plt.ylabel('Loss (MSE)'); plt.savefig('optimization_new_data.png')
    
    return bestModel

# Main function:
def newTicker():
    # Load Data:
    bestModel, scaler, fullData, fullDataScaled, today = loadData(fpath, final_fname, csvfile)
    
    # Predict new data with model:
    testResults = predictNew(bestModel, scaler, fullData, fullDataScaled)
    
    # Further train model using new data:
    bestModel = trainOnNew(bestModel, scaler, fullData, fullDataScaled)
    
    # Predict new data with model after training:
    testResults = predictNew(bestModel, scaler, fullData, fullDataScaled);
    
    # Save trained model:
    saveBest(new_fname, bestModel, scaler, testResults)
        
##############################################################################
# Load model and new data:

fpath = '' # path to csv and models files
csvfile = 'DL_FDN.csv' # new data file
final_fname = 'dl_final_model.h' # filename to which final selected model will be stored
new_fname = 'dl_final_model_FDN.h' # rename model after retraining

trainingEpochs = 600 
useBestCheckpoint = True # whether to reload last-saved best model for dev prediction

lookBack = 1 # lookback period / window size (default: 1, i.e., previous day)
returnPeriod = 1 # number of days to calculate forward return over (default: 1)
segmentSize = 10 # used for randomly selecting contiguous data of segmentSize for data partitions (not used for minibatch model training, batch training is default)
buyThreshold = 0.00 # used for calculating model evaluation metrics

# Outputs:
figures0 = False # plot batch Adam optimization
figures1 = False # plot train predictions
figures2 = False # plot dev predictions
verbose = 2 # 0, 1, 2

# Uncomment and run function below to train:
# newTicker()