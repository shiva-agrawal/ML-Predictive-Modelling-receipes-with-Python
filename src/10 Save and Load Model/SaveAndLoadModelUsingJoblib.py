'''
 Author      : Shiva Agrawal
 Date        : 05.09.2018
 Version     : 1.0
 Description : Save and Load ML Model in Python using package Joblib from scikit-learn library.
'''

'''
Informtion about the dataset: pima-indians-diabetes

About Data (ML_Data)
1 to 8 are features (inputs)
# 1. Number of times pregnant
# 2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 3. Diastolic blood pressure (mm Hg)
# 4. Triceps skin fold thickness (mm)
# 5. 2-Hour serum insulin (mu U/ml)
# 6. Body mass index (weight in kg/(height in m)^2)
# 7. Diabetes pedigree function
# 8. Age (years)
9 is output. It is logistoc regression or binary classification example
# 9. Class variable (0 or 1)
'''

'''
The Joblib library is part of the SciPy ecosystem and provides utilities for pipelining Python
jobs. It provides utilities for saving and loading Python objects that make use of NumPy data
structures, efficiently. This can be useful for some machine learning algorithms that require a
lot of parameters or store the entire dataset (e.g. k-Nearest Neighbors). The example below
demonstrates how we can train a logistic regression model on the Pima Indians onset of diabetes
dataset, save the model to file using Joblib and load it to make predictions on the unseen test
set.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.externals.joblib as jblib


# step 1 : Load the data
#----------------------------------------------------------------

CsvFileName = 'pima-indians-diabetes_without_header.csv'
header_names = ['PregnantCount','PlasmaGlu','BP','SkinThickness','SerumInsulinIn2Hours','BMI','PedigreeFunc','Age','Class']
ML_data = pd.read_csv(CsvFileName,names = header_names)
print(ML_data.shape)
print(ML_data)

# step 2: Separate input and output data for ML
#----------------------------------------------------------------

ML_data_array = ML_data.values
ML_data_input = ML_data_array[:,0:8]  # all rows and columns from index 0 to 7 (all input features)
ML_data_output = ML_data_array[:,8] # all rows and column index 8 (last column - Class (output))

# step 3: generate model (here using train_test split for CV  and LogisticRegression)
#----------------------------------------------------------------
[X_train, X_test, Y_train, Y_test] = train_test_split(ML_data_input,ML_data_output,test_size=0.33, random_state=7)
model = LogisticRegression()
model.fit(X_train, Y_train)


# step 4: Save the model uisng Joblib
#----------------------------------------------------------------

filename = 'model_using_joblib.sav'
jblib.dump(model, filename)


# step 5: Later...... Load the saved model using Joblib
#----------------------------------------------------------------

loaded_model = jblib.load(filename)
result = loaded_model.score(X_test, Y_test)

# # test the loaded model
print(result)

'''
Accuracy is 75.59 % 
'''
