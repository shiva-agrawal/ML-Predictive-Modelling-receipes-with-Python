'''
 Author      : Shiva Agrawal
 Date        : 04.09.2018
 Version     : 1.0
 Description : Cross validation using Leave one out technique to find performance of the ML model

'''

'''
Dataset used: pima-indians-diabetes_without_header.csv

About Data Headers (ML_Data)
1 to 8 are features (inputs)
# 1. Number of times pregnant
# 2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 3. Diastolic blood pressure (mm Hg)
# 4. Triceps skin fold thickness (mm)
# 5. 2-Hour serum insulin (mu U/ml)
# 6. Body mass index (weight in kg/(height in m)^2)
# 7. Diabetes pedigree function
# 8. Age (years)
# 9 is output. It is logistic regression or binary classification example
# 9. Class variable (0 or 1)
'''

'''
when the cross validation is with size of the fold = 1 (k is set to the number of
observations in the dataset), it is leave one out cross validation. The result is a large number of 
performance measures that can be summarized in an effort to give a more reasonable estimate of 
the accuracy of your model on unseen data.

A downside is that it can be a computationally more expensive procedure than k-fold cross
validation. In the example below we use leave-one-out cross validation.
'''

import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_score 
from sklearn.linear_model import LogisticRegression

# step 1 : Load the data 
#-----------------------------------------------

CsvFileName = 'pima-indians-diabetes_without_header.csv'
header_names = ['PregnantCount','PlasmaGlu','BP','SkinThickness','SerumInsulinIn2Hours','BMI','PedigreeFunc','Age','Class']
ML_data = pd.read_csv(CsvFileName,names = header_names)
print(ML_data.shape)
print(ML_data)


#step 2: separate input and output for ML
#-----------------------------------------------

ML_data_array = ML_data.values
ML_data_input = ML_data_array[:,0:8]  # all rows and columns from index 0 to 7 (all input features)
ML_data_output = ML_data_array[:,8] # all rows and column index 8 (last column - Class (output))


#step 3: Leave one out implemntation and finding accuracy of the model by cross validation
#-----------------------------------------------

num_folds = 1  # k = 1  (for leave one out cross validation)

leaveOneOutCV = LeaveOneOut()
model = LogisticRegression()

score = cross_val_score(model, ML_data_input, ML_data_output,cv = leaveOneOutCV)
print(score.mean()*100.0)
print(score.std()*100.0)

'''
mean and standard deviation helps to understand the accuracy of the model. Mean*100 = 76.95 % and Accuracy with
distribution of std()*100 = 42.19 % 
'''