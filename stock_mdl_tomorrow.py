# -*- coding: utf-8 -*-
"""
stock_mdl_tomorrow.py
Created on Wed Oct 17, 2018
@author: Elliot Layden

Info:
Script to load previously trained model, provide stats on full dataset, and 
    predict % gain/loss for tomorrow
"""
##############################################################################
# Imports:
import numpy as np
from train_stock_mdl import scaleToday
from stock_mdl_new_ticker import loadData, predictNew

def mdl_tomorrow():
    
    # Load Data:
    bestModel, scaler, fullData, fullDataScaled, today = loadData(fpath, final_fname, csvfile)
    
    # Model Stats on Current Full Dataset:
    predictNew(bestModel, scaler, fullData, fullDataScaled);
    
    # Tomorrow's prediction:
    today_scaled = scaleToday(today, scaler)
    yhat_today = bestModel.predict(today_scaled)
    yhat_today = scaler.inverse_transform(np.hstack((today_scaled, yhat_today)))[:,-1]
    yhat_today = yhat_today.reshape((yhat_today.shape[0],1))
    print("Tomorrow:  %.3f%%" % (yhat_today*100))
    

fpath = '' # path to csv and models files
#csvfile = 'DL_FDN.csv'
#final_fname = 'dl_final_model_FDN.h'
csvfile = 'DL_SPX.csv' # csv file containing predictors and outcome
final_fname = 'dl_final_model.h' # filename to which final selected model will be stored

mdl_tomorrow()
