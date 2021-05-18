'''
 Author      : Shiva Agrawal
 Date        : 05.09.2018
 Version     : 1.0
 Description : Ridge Regression Machine Learning Algorithm
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
Ridge Regression is an extension of linear regression where the loss function is modified to
minimize the complexity of the model measured as the sum squared value of the coefficient
values (also called the L2-norm). We can construct a ridge regression model by using the Ridge Class
'''

import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge

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
model = Ridge() # this is model instance

model.fit(ML_data_input,ML_data_output) # this fits the model on the dataset
print(model.predict(ML_data_input[0:5,:])) # just prediction for some inputs (for understanding)

cv_result1 = cross_val_score(model,ML_data_input,ML_data_output,cv=kfold, scoring='r2')
print(cv_result1.mean())

cv_result2 = cross_val_score(model,ML_data_input,ML_data_output,cv=kfold, scoring='neg_mean_squared_error')
print(cv_result2.mean())

'''
output: R2 value: 0.2561
output: MSE value: -34.0782
'''
