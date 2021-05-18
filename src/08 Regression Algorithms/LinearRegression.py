'''
 Author      : Shiva Agrawal
 Date        : 05.09.2018
 Version     : 1.0
 Description : Linear Regression Machine Learning Algorithm
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

'''
Linear regression assumes that the input variables have a Gaussian distribution. It is also
assumed that input variables are relevant to the output variable and that they are not highly
correlated with each other (a problem called collinearity). You can construct a linear regression
model using the LinearRegression class
'''

import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

# step 1 : Load the data
#------------------------------------------------

CsvFileName = 'housing.data.csv'
header_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','HOUSE_PRICE' ]
ML_data = pd.read_csv(CsvFileName,sep='\s+' ,names = header_names)  # sep='\s+' is used to have multi-space separated values
print(ML_data.shape)
print(ML_data)

# step 2: Separate input and output data for ML
#------------------------------------------------

ML_data_array = ML_data.values
ML_data_input = ML_data_array[:,0:13]  # all rows and columns from index 0 to 12 (all 13 input features)
ML_data_output = ML_data_array[:,13] # all rows and column index 13 (last column - House_price (output))


# step 3: Model Development
#------------------------------------------------

num_folds = 10  # k= 10
seed = 7

kfold = KFold(n_splits=num_folds,random_state=seed) # this is for Kfold
model = LinearRegression() # this is model instance

model.fit(ML_data_input,ML_data_output) # this fits the model on the dataset
print(model.predict(ML_data_input[0:5,:])) # just prediction for some inputs (for understanding)

cv_result1 = cross_val_score(model,ML_data_input,ML_data_output,cv=kfold, scoring='r2')
print(cv_result1.mean())

cv_result2 = cross_val_score(model,ML_data_input,ML_data_output,cv=kfold, scoring='neg_mean_squared_error')
print(cv_result2.mean())

'''
output: R2 value: 0.20252
output: MSE value: -34.705
'''
