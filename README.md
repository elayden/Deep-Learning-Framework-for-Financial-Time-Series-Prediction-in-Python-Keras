# Deep Learning Framework for Financial Time Series Prediction in Python Keras
-Randomly partitions time series segments into train, development, and test sets
-Trains multiple models optimizing parameters for development set, and performs final cross-validation in test sets
-Calculates model’s annualized return, improvement from buy/hold, percent profitable trades, profit factor, max drawdown

train_stock_mdl.py
    this script imports a csv file containing predictors and an outcome 
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
    
stock_mdl_new_ticker.py
    A script that loads a previously optimized neural network and predicts 
    data from a new stock/ETF ticker. After evaluating performance, there is
    the option to train the model further on the new ticker.
    
stock_mdl_tomorrow.py
    Script to load previously trained model, provide stats on full dataset, and 
    predict % gain/loss for tomorrow
