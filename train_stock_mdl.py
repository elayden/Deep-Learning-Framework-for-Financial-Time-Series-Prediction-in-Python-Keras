# -*- coding: utf-8 -*-
"""
train_stock_mdl.py
Created on Wed Oct 17, 2018
@author: Elliot Layden

Info: this script imports a csv file containing predictors and an outcome 
    variable (last column). It was designed with stock/ETF prediction in mind, 
    but could potentially be adapted for other data. The default neural network
    architecture is very simple (2 hidden layers with 200 units each and 
    L2 regularization). However, this can easily be changed. A number of models
    are trained (specified by nModels) on the Training data, and each is 
    evaluated relative to the out-of-sample development ("Dev") data. An 
    optimal model can then be tested on the final "Test" data, avoiding 
    issues with multiple comparisons. (Note that the model training is very 
    non-deterministic: some models may end up using very different "strategies"
    that are more or less effective at predicting the data. Some will 
    cross-validate better than others, hence why training multiple models on 
    and testing for cross-validation to the Dev data can be helpful. 
    Note that training multiple models in this way, however, necessitates the 
    use of Test data in addition to Dev data, in order to fairly evaluate 
    the final model.)
    
    Important note:  predictors (all columns before last) must be pre-lagged
    relative to outcome variable (last column). I.e., the last column should 
    contain data of interest from one period ahead of the predictors in the 
    other column. These scripts/functions are designed for use with other 
    external scripts functions for preparing the predictors & outcome variables, 
    and lagging the data should be done before utilizing these as part of that
    data preparation process.
"""

##############################################################################
# Imports:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from IPython import get_ipython
import random
import os
import pickle
from keras.models import Sequential, load_model
from keras.layers import Dense, regularizers # LSTM, ConvLSTM2D, TimeDistributed, Convolution1D, Dropout, MaxPooling1D, Flatten, Activation, RepeatVector
from keras.callbacks import ModelCheckpoint

##############################################################################
# Utility Functions:

# If a lookBack period >1 is specified, this converts data into flattened sliding 
#   windows; if a returnPeriod >1 is specified, it converts Y into forward 
#   compound returns
def getSlidingWindowData(data, lookBack, returnPeriod):
    nWindows = data.shape[0] - lookBack; 
    x = np.zeros((nWindows, (data.shape[1]-1) * lookBack))
    y = np.zeros((nWindows, 1))
    for i in range(nWindows - returnPeriod):
    	x[i,:] = data[i:i+lookBack, :-1].flatten()
    	if returnPeriod==1: # use raw return if #days == 1
    		y[i,:] = data[i+lookBack,-1]
    	else:
    		y[i,:] = np.sum(np.log(1 + data[i+lookBack:i+lookBack+returnPeriod, -1])) # use percentages, not scaled
          
    ind = ~np.any(~np.isfinite(np.hstack((x,y))), axis=1) 
    y = y[ind]
    x = x[ind,:]
    return x, y

# Randomly partition segments of data into train, dev, and test:
def partitionData(x, y, segmentSize, trainPercent, devPercent):
    # Get Indices for Each Partition:
    nSegments = int(np.floor(x.shape[0]/segmentSize))
    nSegmentsTrain = np.round(trainPercent * nSegments)
    nSegmentsDev = np.round(devPercent * nSegments)
    trainSegments = np.sort(random.sample(range(int(nSegments)), int(nSegmentsTrain))) # randomly select training
    remaining = np.setdiff1d(np.arange(0,nSegments), trainSegments, assume_unique=False) # get remaining possibilities
    devSegments = np.sort(random.sample(list(remaining), int(nSegmentsDev)))  # randomly select dev from remaining
    
    # Get Data Segments:
    train = np.zeros((0, x.shape[1])); dev = np.zeros((0, x.shape[1])); test = np.zeros((0, x.shape[1]));
    train_y = np.zeros((0,1)); dev_y = np.zeros((0,1)); test_y = np.zeros((0,1))
    for i in range(nSegments):
        segInd = np.arange(i*segmentSize, (i+1)*segmentSize)
        if any(np.in1d(trainSegments, i, assume_unique=True)): # train segment
            train = np.vstack((train, x[segInd, :]))
            train_y = np.vstack((train_y, y[segInd])) 
        elif any(np.in1d(devSegments, i, assume_unique=True)): # dev segment
            dev = np.vstack((dev, x[segInd, :]))
            dev_y = np.vstack((dev_y, y[segInd])) 
        else: # test segment
            test = np.vstack((test, x[segInd, :]))
            test_y = np.vstack((test_y, y[segInd]))            
    dev = np.vstack((dev, x[(i+1)*segmentSize:, :])) # add remainder to dev
    dev_y = np.vstack((dev_y, y[(i+1)*segmentSize:])) # add remainder to dev
    
    # Horizontally concatenate X and Y:
    train = np.hstack((train, train_y))
    dev = np.hstack((dev, dev_y))
    test = np.hstack((test, test_y))
    return train, dev, test

# Scale train and test data to [-1, 1] (note: uses scaler obtained from 
#   training data to rescale dev and test sets; this is important so as to not
#   introduce information from dev and test to training)
def rescaleData(train, dev, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train = train.reshape(train.shape[0], train.shape[1])
    trainScaled = scaler.transform(train)
    dev = dev.reshape(dev.shape[0], dev.shape[1])
    devScaled = scaler.transform(dev)
    test = test.reshape(test.shape[0], test.shape[1])
    testScaled = scaler.transform(test)
    return scaler, trainScaled, devScaled, testScaled

def separateXY(data):
    xUse = data[:,:-1]
    yUse = data[:,-1]
    yUse = yUse.reshape((yUse.shape[0], 1))
    return xUse, yUse

def scaleToday(today, scaler):
    holdY = today[-1,-1].reshape((1,1))
    today = today[:,:-1].flatten()
    today = today.reshape((1,today.shape[0]))
    today = np.hstack((today, holdY))
    today_scaled = scaler.transform(today)
    today_scaled = today_scaled[:,:-1]
    return today_scaled

def compileModel(inputShape):
    mdl = Sequential(); 
    mdl.add(Dense(200, input_shape=(inputShape,), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    mdl.add(Dense(200, activation='relu'))
    mdl.add(Dense(1)) # activation='tanh'
    mdl.compile(loss='MSE', optimizer='adam') 
    return mdl

def plotPredictions(y, yhat, figTitle):
    endRange = y.shape[0]
    if (endRange-200 < 0):
        startRange = 0
    else:
        startRange = endRange-200
    range1 = list(range(startRange, endRange))
    get_ipython().run_line_magic('matplotlib', 'qt')
    plt.figure(figsize=(20,10));  plt.title(figTitle)
    plt.plot(100*y[range1,-1],'.'); plt.plot(100*yhat[range1,0],'.'); 
    for i in np.nditer(np.where(np.logical_and((y[range1,-1]<0), (yhat[range1,0]>0)))):
        plt.plot([i,i], [100*y[range1[i],-1], 100*yhat[range1[i],0]],'r-')
    for i in np.nditer(np.where(np.logical_and((y[range1,-1]>0), (yhat[range1,0]>0)))):
        plt.plot([i,i], [100*y[range1[i],-1], 100*yhat[range1[i],0]],'g-')
    plt.plot(np.zeros((yhat[range1].shape))); 
    plt.xlabel('Observation'); plt.ylabel('%-Change')
    plt.show() 
    plt.savefig(figTitle + '.png')
    
def calcPerformance(yReal, yHat):     
    
    yReal = yReal[:,-1]
    yReal = yReal.reshape((yReal.shape[0],1))
    
    # RMSE:
    rmse = np.sqrt(((yReal - yHat) ** 2).mean(axis=0)) * 100
    rmse_null = np.sqrt(((yReal - np.mean(yReal)) ** 2).mean(axis=0)) * 100
    rmseImprove = (rmse_null-rmse)/rmse_null * 100
    
    # Annualized Return (https://www.investopedia.com/terms/a/annualized-total-return.asp):
        # Note: this is very approximate given that data are from often non-contiguous segments 
    buys = yReal[yHat>0]
    nTrades = np.sum(yHat>0)
    logReturns = np.log(1 + buys)
    annualizedReturn = ((np.sum(logReturns) + 1)**(252/logReturns.shape[0]) - 1)*100
    nullAnnualizedReturn = ((np.sum(np.log(1+yReal)) + 1)**(252/yReal.shape[0]) - 1)*100
    returnImprovement = (annualizedReturn-nullAnnualizedReturn)/nullAnnualizedReturn * 100
    
    # Max Drawdown:
    maxDrawdown = 0
    runSum = 0
    for i in range(0, buys.shape[0]):
        if np.remainder(i,252)==0: # reset running sum after 1 year
            runSum=0
        runSum+=logReturns[i]
        if runSum < maxDrawdown:
            maxDrawdown=runSum
    maxDrawdown*=100
            
    # %Profitable:
    n_loss = np.sum(np.logical_and(yReal<0, (yHat>buyThreshold))) 
    n_win = np.sum(np.logical_and(yReal>0, (yHat>buyThreshold)))
    percentProfitable = n_win/(n_win+n_loss) *100
    
    # Profit factor:
    win1 = np.sum(np.log(1 + yReal[np.logical_and(yReal>0, (yHat>buyThreshold))])) * 100    
    loss1 = np.abs(np.sum(np.log(1 + yReal[np.logical_and(yReal<0, (yHat>buyThreshold))])) * 100)
    profitFactor = win1/loss1
    
    return [annualizedReturn, returnImprovement, nTrades, percentProfitable, profitFactor, maxDrawdown, rmseImprove]
    
# Main function for training models
def trainModels(x, y, today):
    # Outcome metrics:
    models = [] # tuple:  (mdl, scaler, yTest (unscaled), test_X (scaled))
    trainResults = np.zeros((nModels,7)) # Annualized Return, vBuy&Hold, # Trades, %Profitable, Profit Factor, Max Drawdown, RMSEimprove
    devResults = np.zeros((nModels,7))
    
    for ix in range(nModels):
        
        print("Iteration: ", (ix))
    
        # Randomly partition data:
        train, dev, test = partitionData(x, y, segmentSize, trainPercent, devPercent)
        yTest = test[:,-1]
        yTest = yTest.reshape((yTest.shape[0],1))
        
        # Transform the scale of the data to tanh (-1, 1):
        scaler, trainScaled, devScaled, testScaled = rescaleData(train, dev, test)    
        train_X, train_y = separateXY(trainScaled)
        dev_X, dev_y = separateXY(devScaled)
        test_X, test_y = separateXY(testScaled)
        
        # Scale Today's data, for tomorrow's prediction:
        today_scaled = scaleToday(today, scaler)
            
        # Fit Model:
        mdl = compileModel(train_X.shape[1])
        
        if useBestCheckpoint:
            history = mdl.fit(train_X, train_y, epochs=trainingEpochs, batch_size=train_y.shape[0], validation_data=(dev_X, dev_y), verbose=mdlVerbose, shuffle=True, callbacks=[ModelCheckpoint('mdl_checkpoint.h', monitor='val_loss', save_best_only=True, verbose=mdlVerbose)])
        else:
            history = mdl.fit(train_X, train_y, epochs=trainingEpochs, batch_size=train_y.shape[0], validation_data=(dev_X, dev_y), verbose=mdlVerbose, shuffle=True)
        
        # Plot training loss:
        if figures0:
            get_ipython().run_line_magic('matplotlib', 'qt')
            plt.plot(history.history['loss'], label='train'); plt.plot(history.history['val_loss'], label='test'); 
            plt.legend(); plt.show(); plt.xlabel('Epoch'); plt.ylabel('Loss (MSE)'); plt.savefig('optimization.png')
        
        # Reload best model if desired:
        if useBestCheckpoint:
            mdl = load_model('mdl_checkpoint.h');  
            
        # Append trained model to list:
        models.append((mdl, scaler, yTest, test_X))
        
        # Predict Train & Dev Sets:
        yhat_train = mdl.predict(train_X)
        yhat_dev = mdl.predict(dev_X)
        yhat_today = mdl.predict(today_scaled)
        
        # Invert the scaling:
        yhat_train = scaler.inverse_transform(np.hstack((train_X, yhat_train)))[:,-1]
        yhat_train = yhat_train.reshape((yhat_train.shape[0],1))
        
        yhat_dev = scaler.inverse_transform(np.hstack((dev_X, yhat_dev)))[:,-1]
        yhat_dev = yhat_dev.reshape((yhat_dev.shape[0],1))
        
        yhat_today = scaler.inverse_transform(np.hstack((today_scaled, yhat_today)))[:,-1]
        yhat_today = yhat_today.reshape((yhat_today.shape[0],1))
        
        # Train:  Plot Real vs. Predicted:
        if figures1:
            plotPredictions(train, yhat_train, 'train_predictions')
        
        # Dev:  Plot Real vs. Predicted:
        if figures2:
            plotPredictions(dev, yhat_dev, 'dev_predictions')
            
        # Calculate performance metrics:
        trainResults[ix,:] = calcPerformance(train, yhat_train)
        devResults[ix,:] = calcPerformance(dev, yhat_dev)
        
        if verbose>0:
            # Annualized Return, vBuy&Hold, # Trades, %Profitable, Profit Factor, Max Drawdown, RMSEimprove
            print("Train: Annualized = %.2f%%, vBuy&Hold = %.2f%%, #Trades = %.0f, "
                  "%%Profitable = %.1f%%, Profit Factor = %.2f, Max Drawdown = %.2f%%, "
                  "RMSE_improvement = %.2f%%" % (trainResults[ix,0], trainResults[ix,1],
                  trainResults[ix,2], trainResults[ix,3], trainResults[ix,4], 
                  trainResults[ix,5], trainResults[ix,6]))
            
            print("Dev: Annualized = %.2f%%, vBuy&Hold = %.2f%%, #Trades = %.0f, "
                  "%%Profitable = %.1f%%, Profit Factor = %.2f, Max Drawdown = %.2f%%, "
                  "RMSE_improvement = %.2f%%" % (devResults[ix,0], devResults[ix,1],
                  devResults[ix,2], devResults[ix,3], devResults[ix,4], 
                  devResults[ix,5], devResults[ix,6]))
    
        if verbose==2:
            # Tomorrow's prediction:    
            print("Tomorrow:  %.2f%%" % (yhat_today*100))
    return models, trainResults, devResults

# Models data:
def saveData(models, trainResults, devResults):
    f = open(models_fname, 'wb')
    modelsData = (models, trainResults, devResults)
    pickle.dump(modelsData, f)
    f.close()

# Final model data:    
def saveBest(fname, bestModel, scaler, testResults):
    f2 = open(fname, 'wb')
    modelsData = (bestModel, scaler, testResults)
    pickle.dump(modelsData, f2)
    f2.close()
    
def main():
    # Load data:
    os.chdir(fpath)
    data = pd.read_csv(csvfile) # DL_data2.csv
    data.fillna(0, inplace=True)
    data = data.values
    today = data[-lookBack:,:]

    # Get sliding window data (if lookBack > 1 day) and compound returns (if returnPeriod > 1 day):
    x, y = getSlidingWindowData(data, lookBack, returnPeriod)
    
    # Train models:
    models, trainResults, devResults = trainModels(x, y, today)

    ###############################################################################
    # Compare Simulations:  (0) Annualized Return, (1) vBuy&Hold, (2) # Trades, (3) %Profitable, (4) Profit Factor, (5) Max Drawdown, (6) RMSEimprove
    
    # Plot annualized returns distribution:
    plt.hist(devResults[:,0], bins=30, density=True)
    devResults[:,1].min() 
    
    # vBuy&Hold distribution:
    plt.hist(devResults[:,0], bins=30, density=True)
    devResults[:,1].min() 
          
    # Number of Trades distribution:
    plt.hist(devResults[:,2], bins=30, density=True)
    
    # Percent Profitable distribution:
    plt.hist(devResults[:,3], bins=30, density=True)
    
    # Profit Factor distribution:
    plt.hist(devResults[:,4], bins=30, density=True)
    
    # Max Drawdown distribution:
    plt.hist(devResults[:,5], bins=30, density=True)
    
    # RMSE Improvement distribution:
    plt.hist(devResults[:,6], bins=30, density=True)
    np.sum(devResults[:,6]>0)
    
    # SELECT BEST MODEL BASED ON METRIC OF CHOICE (default:  RMSE)
    selectionMetric = 6 # RMSE improvement
    
    if selectionMetric==5:
        bestModel = models[devResults[:,selectionMetric].argmin()][0]
        scaler = models[devResults[:,selectionMetric].argmin()][1]
        yTest = models[devResults[:,selectionMetric].argmin()][2] 
        xTest = models[devResults[:,selectionMetric].argmin()][3] 
    else:
        bestModel = models[devResults[:,selectionMetric].argmax()][0]
        scaler = models[devResults[:,selectionMetric].argmax()][1]
        yTest = models[devResults[:,selectionMetric].argmax()][2] 
        xTest = models[devResults[:,selectionMetric].argmax()][3] 
    
    #bestModel = models[0][0]
    #scaler = models[0][1]
    #yTest = models[0][2] 
    #xTest = models[0][3] 
        
    # Cross-validate final model on Test data:
    yhat_test = bestModel.predict(xTest)
    yhat_test = scaler.inverse_transform(np.hstack((xTest, yhat_test)))[:,-1]
    yhat_test = yhat_test.reshape((yhat_test.shape[0],1))   
    testResults = calcPerformance(np.hstack((xTest,yTest)), yhat_test)
    print("Test: Annualized = %.2f%%, vBuy&Hold = %.2f%%, #Trades = %.0f, "
          "%%Profitable = %.1f%%, Profit Factor = %.2f, Max Drawdown = %.2f%%, "
          "RMSE_improvement = %.2f%%" % (testResults[0], testResults[1],
          testResults[2], testResults[3], testResults[4], 
          testResults[5], testResults[6]))
    
    # Plot test data predictions:
    plotPredictions(np.hstack((xTest,yTest)), yhat_test, 'test_predictions')
    
    saveData(models, trainResults, devResults)
    saveBest(final_fname, bestModel, scaler, testResults)
    
    return bestModel, scaler, trainResults, devResults, testResults, models

###############################################################################
# Define Training Parameters:
    
fpath = '' # path to csv and models files
csvfile = 'DL_SPX.csv' # csv file containing predictors and outcome
models_fname = 'dl_models_data.h' # filename to which models data will be stored
final_fname = 'dl_final_model.h' # filename to which final selected model will be stored

# Data Partitions:
trainPercent = .9 # % of data to use for training set
devPercent = .05 # % of data to use for dev set

# Model Type Parameters:
trainingEpochs = 800 # number of training epochs per model
nModels = 5 # number of models to create and optimize for Dev set
useBestCheckpoint = True # whether to reload last-saved best model for dev prediction

lookBack = 1 # lookback period / window size (default: 1, i.e., previous day)
returnPeriod = 1 # number of days to calculate forward return over (default: 1)
segmentSize = 10 # used for randomly selecting contiguous data of segmentSize for data partitions (not used for minibatch model training, batch training is default)
buyThreshold = 0.00 # used for calculating model evaluation metrics

# Outputs:
figures0 = False # plot batch Adam optimization
figures1 = False # plot train predictions
figures2 = False # plot dev predictions
verbose = 1 # 0, 1, 2

mdlVerbose=0
if verbose==2:
    mdlVerbose=2

# Uncomment and run main():
# bestModel, scaler, trainResults, devResults, testResults, models = main()

#############################################################################
# To load Model(s):
#f = open(models_fname, 'rb')
#(models, trainResults, devResults) = pickle.load(f)
#f.close()
    
# f = open(final_fname, 'rb')
# (bestModel, scaler, testResults) = pickle.load(f)
# f.close()
