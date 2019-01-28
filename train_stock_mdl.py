# -*- coding: utf-8 -*-
"""
train_stock_mdl.py
Created on Wed Oct 17, 2018
Updated on Sun Jan 27, 2019
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
def rescaleData(train, dev, test, scaleColumns):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train[:,scaleColumns])
    train = train.reshape(train.shape[0], train.shape[1])
    trainScaled = np.hstack((train[:, dummyVars], scaler.transform(train[:,scaleColumns])))
    dev = dev.reshape(dev.shape[0], dev.shape[1])
    devScaled = np.hstack((dev[:,dummyVars], scaler.transform(dev[:,scaleColumns])))
    test = test.reshape(test.shape[0], test.shape[1])
    testScaled = np.hstack((test[:,dummyVars], scaler.transform(test[:,scaleColumns])))
    return scaler, trainScaled, devScaled, testScaled

def separateXY(data):
    xUse = data[:,:-1]
    yUse = data[:,-1]
    yUse = yUse.reshape((yUse.shape[0], 1))
    return xUse, yUse

def scaleToday(today, scaler, scaleColumns):
    holdY = today[-1,-1].reshape((1,1))
    today = today[:,:-1].flatten()
    today = today.reshape((1,today.shape[0]))
    today = np.hstack((today, holdY))
    today_scaled = np.hstack((today[:,dummyVars], scaler.transform(today[:,scaleColumns])))
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
    
    rsquared = np.corrcoef(np.transpose(yReal), np.transpose(yHat))[0,1]**2
    
    # Annualized Return (https://www.investopedia.com/terms/a/annualized-total-return.asp):
        # Note: this is very approximate given that data are from often non-contiguous segments 
    buys = yReal[yHat>0]
    nTrades = np.sum(yHat>0)
    logReturns = np.log(1 + buys)
    annualizedReturn = ((np.sum(logReturns) + 1)**(252/yReal.shape[0]) - 1)*100
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
    percentProfitable = n_win/(n_win+n_loss) * 100
    
    # Profit factor:
    win1 = np.sum(yReal[np.logical_and(yReal>0, (yHat>buyThreshold))]) 
    loss1 = np.sum(np.abs(yReal[np.logical_and(yReal<0, (yHat>buyThreshold))])) 
    profitFactor = win1/loss1
    
    return [annualizedReturn, returnImprovement, nTrades, percentProfitable, profitFactor, maxDrawdown, rmseImprove, rsquared]
    
# Main function for training models
def trainModels(x, y, today):
    # Outcome metrics:
    models = [] # tuple:  (mdl, scaler, yTest (unscaled), test_X (scaled))
    trainResults = np.zeros((nModels,8)) # Annualized Return, vBuy&Hold, # Trades, %Profitable, Profit Factor, Max Drawdown, RMSEimprove
    devResults = np.zeros((nModels,8))
    
    for ix in range(nModels):
        
        print("Training Model: ", (ix))
    
        # Randomly partition data:
        train, dev, test = partitionData(x, y, segmentSize, trainPercent, devPercent)
        yTest = test[:,-1]
        yTest = yTest.reshape((yTest.shape[0],1))
        
        # Transform the scale of the data to tanh (-1, 1):
        scaler, trainScaled, devScaled, testScaled = rescaleData(train, dev, test, scaleColumns)    
        train_X, train_y = separateXY(trainScaled)
        dev_X, dev_y = separateXY(devScaled)
        test_X, test_y = separateXY(testScaled)
        
        # Scale Today's data, for tomorrow's prediction:
        today_scaled = scaleToday(today, scaler, scaleColumns)
            
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
        trainUnscale = np.hstack((train_X, yhat_train))
        yhat_train = scaler.inverse_transform(trainUnscale[:,scaleColumns])[:,-1]
        yhat_train = yhat_train.reshape((yhat_train.shape[0],1))
        
        devUnscale = np.hstack((dev_X, yhat_dev))
        yhat_dev = scaler.inverse_transform(devUnscale[:,scaleColumns])[:,-1]
        yhat_dev = yhat_dev.reshape((yhat_dev.shape[0],1))
        
        todayUnscale = np.hstack((today_scaled, yhat_today))
        yhat_today = scaler.inverse_transform(todayUnscale[:,scaleColumns])[:,-1]
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
                  "%%RMSE_improve = %.2f%%, R^2 = %.3f" % (trainResults[ix,0], trainResults[ix,1],
                  trainResults[ix,2], trainResults[ix,3], trainResults[ix,4], 
                  trainResults[ix,5], trainResults[ix,6], trainResults[ix,7]))
            
            print("Dev: Annualized = %.2f%%, vBuy&Hold = %.2f%%, #Trades = %.0f, "
                  "%%Profitable = %.1f%%, Profit Factor = %.2f, Max Drawdown = %.2f%%, "
                  "%%RMSE_improve = %.2f%%, R^2 = %.3f" % (devResults[ix,0], devResults[ix,1],
                  devResults[ix,2], devResults[ix,3], devResults[ix,4], 
                  devResults[ix,5], devResults[ix,6], devResults[ix,7]))
    
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
    print('Successfully saved models.')

# Final model data:    
def saveBest(fname, bestModel, scaler, testResults):
    f2 = open(fname, 'wb')
    modelsData = (bestModel, scaler, testResults)
    pickle.dump(modelsData, f2)
    f2.close()
    print('Successfully saved best model.')
    
# AUC of ROC curve:  
def auc(modelNum, models):
    bestModel = models[modelNum][0]
    scaler = models[modelNum][1]
    yTest = models[modelNum][2] 
    xTest = models[modelNum][3] 
    yhat_test = bestModel.predict(xTest)
    testUnscale = np.hstack((xTest, yhat_test))
    yhat_test = scaler.inverse_transform(testUnscale[:,scaleColumns])[:,-1]
    yhat_test = yhat_test.reshape((yhat_test.shape[0],1))   
        
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(yTest>0, yhat_test)
    roc_auc = auc(fpr, tpr)
    
    ixOpt = np.argmax(tpr - fpr)
    optThresh = thresholds[ixOpt]
    
    #get_ipython().run_line_magic('matplotlib', 'qt')
    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot(fpr[ixOpt], tpr[ixOpt], marker='o', color='red')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    print('Optimal Threshold:  %.4f' % (optThresh))

def testStats(modelNum, models, trainResults, devResults):
    bestModel = models[modelNum][0]
    scaler = models[modelNum][1]
    yTest = models[modelNum][2] 
    xTest = models[modelNum][3] 
        
    print("MODEL %.0f ------------------------------------------------------------------------------------------------------------- MODEL %.0f " % (modelNum, modelNum))
    # Annualized Return, vBuy&Hold, # Trades, %Profitable, Profit Factor, Max Drawdown, RMSEimprove
    print("Train: Annualized = %.2f%%, vBuy&Hold = %.2f%%, #Trades = %.0f, "
          "%%Profitable = %.1f%%, Profit Factor = %.2f, Max Drawdown = %.2f%%, "
          "%%RMSE_improve = %.2f%%, R^2 = %.3f" % (trainResults[modelNum,0], trainResults[modelNum,1],
          trainResults[modelNum,2], trainResults[modelNum,3], trainResults[modelNum,4], 
          trainResults[modelNum,5], trainResults[modelNum,6], trainResults[modelNum,7]))
    
    print("Dev: Annualized = %.2f%%, vBuy&Hold = %.2f%%, #Trades = %.0f, "
          "%%Profitable = %.1f%%, Profit Factor = %.2f, Max Drawdown = %.2f%%, "
          "%%RMSE_improve = %.2f%%, R^2 = %.3f" % (devResults[modelNum,0], devResults[modelNum,1],
          devResults[modelNum,2], devResults[modelNum,3], devResults[modelNum,4], 
          devResults[modelNum,5], devResults[modelNum,6], devResults[modelNum,7]))
    
    # Cross-validate final model on Test data:
    yhat_test = bestModel.predict(xTest)
    testUnscale = np.hstack((xTest, yhat_test))
    yhat_test = scaler.inverse_transform(testUnscale[:,scaleColumns])[:,-1]
    yhat_test = yhat_test.reshape((yhat_test.shape[0],1))   
    testResults = calcPerformance(np.hstack((xTest,yTest)), yhat_test)    
    print("Test: Annualized = %.2f%%, vBuy&Hold = %.2f%%, #Trades = %.0f, "
          "%%Profitable = %.1f%%, Profit Factor = %.2f, Max Drawdown = %.2f%%, "
          "%%RMSE_improve = %.2f%%, R^2 = %.3f" % (testResults[0], testResults[1],
          testResults[2], testResults[3], testResults[4], 
          testResults[5], testResults[6], testResults[7]))
    
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
    # Compare Simulations:  (0) Annualized Return, (1) vBuy&Hold, (2) # Trades, 
        #  (3) %Profitable, (4) Profit Factor, (5) Max Drawdown, (6) RMSEimprove, 
        # (7) R^2
    if verbose==2:
        # Plot annualized returns distribution:
        plt.title('Annualized Returns')
        plt.hist(devResults[:,0], bins=30)
        
        # vBuy&Hold distribution:
        plt.title('% Improvement vs. Buy/Hold')
        plt.hist(devResults[:,1], bins=30)
              
        # Number of Trades distribution:
        plt.title('# Trades')
        plt.hist(devResults[:,2], bins=30)
        
        # Percent Profitable distribution:
        plt.title('% Profitable')
        plt.hist(devResults[:,3], bins=30)
        
        # Profit Factor distribution:
        plt.title('Profit Factor')
        plt.hist(devResults[:,4], bins=30)
        
        # Max Drawdown distribution:
        plt.title('Max Drawdown')
        plt.hist(devResults[:,5], bins=30)
        
        # RMSE Improvement distribution:
        plt.title('% RMSE Improvement')
        plt.hist(devResults[:,6], bins=30)
        
        # Rsquared distribution:
        plt.title('Rsquared')
        plt.hist(devResults[:,7], bins=30)
    
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
        
    # Cross-validate final model on Test data:
    yhat_test = bestModel.predict(xTest)
    testUnscale = np.hstack((xTest, yhat_test))
    yhat_test = scaler.inverse_transform(testUnscale[:,scaleColumns])[:,-1]
    yhat_test = yhat_test.reshape((yhat_test.shape[0],1))   
    testResults = calcPerformance(np.hstack((xTest,yTest)), yhat_test)    
    print("Test: Annualized = %.2f%%, vBuy&Hold = %.2f%%, #Trades = %.0f, "
          "%%Profitable = %.1f%%, Profit Factor = %.2f, Max Drawdown = %.2f%%, "
          "%%RMSE_improve = %.2f%%, R^2 = %.3f" % (testResults[0], testResults[1],
          testResults[2], testResults[3], testResults[4], 
          testResults[5], testResults[6], testResults[7]))
    
    # Plot test data predictions:
    plotPredictions(np.hstack((xTest,yTest)), yhat_test, 'test_predictions')
    
    saveData(models, trainResults, devResults)
    saveBest(final_fname, bestModel, scaler, testResults)
    
    return bestModel, scaler, trainResults, devResults, testResults, models

###############################################################################
# Define Training Parameters:
    
fpath = '' # path to csv and models files
csvfile = '' # csv file containing predictors and outcome
models_fname = 'dl_model_01272019.h' # filename to which models data will be stored
final_fname = 'dl_final_model_01272019.h' # filename to which final selected model will be stored

anyDummyVars = True  # does .csv contain dummy variables?
if anyDummyVars:
    nVars = 57 # number of columns in Excel  
    dummyVars = np.arange(17) # list the indices of dummy var columns in .csv (0-based indexing)
    scaleColumns =  np.setdiff1d(np.arange(nVars), dummyVars)

# Data Partitions:
trainPercent = .9 # % of data to use for training set
devPercent = .05 # % of data to use for dev set

# Model Type Parameters:
trainingEpochs = 1000 # number of training epochs per model
nModels = 10 # number of models to create and optimize for Dev set
useBestCheckpoint = True # whether to reload last-saved best model for dev prediction
    
# SELECT BEST MODEL BASED ON METRIC OF CHOICE (default:  RMSE)
selectionMetric = 7 # R^2
        
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

#### Uncomment and run main():
# bestModel, scaler, trainResults, devResults, testResults, models = main();

#### Evaluate Selected Model:
# modelNum = 5
# auc(modelNum)
# testStats(modelNum, models, trainResults, devResults)
    
#### Save best model:
#bestModel = models[modelNum][0]
#scaler = models[modelNum][1]
#yTest = models[modelNum][2] 
#xTest = models[modelNum][3] 
#    
#saveData(models, trainResults, devResults) 
#saveBest(final_fname, bestModel, scaler, testResults)
    
#############################################################################
# To load Model(s):
#f = open(models_fname, 'rb')
#(models, trainResults, devResults) = pickle.load(f)
#f.close()
    
# f = open(final_fname, 'rb')
# (bestModel, scaler, testResults) = pickle.load(f)
# f.close()
