'''
 Author      : Shiva Agrawal
 Date        : 04.09.2018
 Version     : 1.0
 Description : Different Regression metrics to evaluate the acuuracy of the ML model
'''

'''
Three Most common regression metrics to predict the performance of the ML model are:
1. Mean Absolute Error
2. Mean Square Error
3. R2 (R square)
'''

'''
Used Dataset - Boston Housing Price
13 features
1 output (PRICE)
506 samples

Attribute Information:

    1. CRIM      per capita crime rate by town
    2. ZN        proportion of residential land zoned for lots over 
                 25,000 sq.ft.
    3. INDUS     proportion of non-retail business acres per town
    4. CHAS      Charles River dummy variable (= 1 if tract bounds 
                 river; 0 otherwise)
    5. NOX       nitric oxides concentration (parts per 10 million)
    6. RM        average number of rooms per dwelling
    7. AGE       proportion of owner-occupied units built prior to 1940
    8. DIS       weighted distances to five Boston employment centres
    9. RAD       index of accessibility to radial highways
    10. TAX      full-value property-tax rate per $10,000
    11. PTRATIO  pupil-teacher ratio by town
    12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks 
                 by town
    13. LSTAT    % lower status of the population
    14. HOUSE_PRICE     Median value of owner-occupied homes in $1000's

'''

import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression  # for Regression Metrics


# step 1 : Load the data
#--------------------------------------------------------

CsvFileName = 'housing.data.csv'
header_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','HOUSE_PRICE' ]
ML_data = pd.read_csv(CsvFileName,sep='\s+' ,names = header_names)  # sep='\s+' is used to have multi-space separated values
print(ML_data.shape)
print(ML_data)

# step 2: Separate input and output data for ML
#--------------------------------------------------------
ML_data_array = ML_data.values
ML_data_input = ML_data_array[:,0:13]  # all rows and columns from index 0 to 12 (all 13 input features)
ML_data_output = ML_data_array[:,13] # all rows and column index 13 (last column - House_price (output))

# step 3: Calculate different metrics
#--------------------------------------------------------


# Metric 1: Mean Absolute error
'''
The Mean Absolute Error (or MAE) is the sum of the absolute differences between predictions
and actual values. It gives an idea of how wrong the predictions were. The measure gives an
idea of the magnitude of the error, but no idea of the direction (e.g. over or under predicting).
'''

num_folds = 10 # k=10
seed = 7

kfold = KFold(n_splits=num_folds, random_state=seed)
model = LinearRegression()
scoringType = 'neg_mean_absolute_error'
results = cross_val_score(model, ML_data_input, ML_data_output, cv=kfold, scoring=scoringType)  # here scoring decides the type of metrics used

print('\n-------Regression Metric 1: Mean Absolute Error-----------')
print(results.mean())
print(results.std())


# Metric 2: Mean Squared Error
'''
The Mean Squared Error (or MSE) is much like the mean absolute error in that it provides a
gross idea of the magnitude of error. Taking the square root of the mean squared error converts
the units back to the original units of the output variable and can be meaningful for description
and presentation. This is called the Root Mean Squared Error (or RMSE).
'''

num_folds = 10 # k=10
seed = 7

kfold = KFold(n_splits=num_folds, random_state=seed)
model = LinearRegression()
scoringType = 'neg_mean_squared_error'
results = cross_val_score(model, ML_data_input, ML_data_output, cv=kfold, scoring=scoringType)  # here scoring decides the type of metrics used

print('\n-------Regression Metrics 2: Mean Squared Error-----------')
print(results.mean())
print(results.std())
'''
This metric too is inverted so that the results are increasing. Remember to take the absolute
value before taking the square root if you are interested in calculating the RMSE.
'''


# Method 3: R2 Metric
'''
The R 2 (or R Squared) metric provides an indication of the goodness of fit of a set of predictions
to the actual values. In statistical literature this measure is called the coefficient of determination.
This is a value between 0 and 1 for no-fit and perfect fit respectively.
'''

num_folds = 10 # k=10
seed = 7

kfold = KFold(n_splits=num_folds, random_state=seed)
model = LinearRegression()
scoringType = 'r2'
results = cross_val_score(model, ML_data_input, ML_data_output, cv=kfold, scoring=scoringType)  # here scoring decides the type of metrics used

print('\n-------Regression Metric 3: R2 squared Error -----------')
print(results.mean())
print(results.std())

'''
R2  = 0.20252899006055963
Predictions have a poor fit to the actual values with a value closer to zero and less than 0.5.
'''

