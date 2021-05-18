'''
 Author      : Shiva Agrawal
 Date        : 04.09.2018
 Version     : 1.0
 Description : Cross validation using test and train split to find performance of the ML model

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
The simplest method that we can use to evaluate the performance of a machine learning
algorithm is to use different training and testing datasets. We can take our original dataset and
split it into two parts. Train the algorithm on the first part, make predictions on the second
part and evaluate the predictions against the expected results. The size of the split can depend
on the size and specifics of your dataset, although it is common to use 67% of the data for
training and the remaining 33% for testing.
This algorithm evaluation technique is very fast. It is ideal for large datasets (millions of
records) where there is strong evidence that both splits of the data are representative of the
underlying problem. Because of the speed, it is useful to use this approach when the algorithm
you are investigating is slow to train. A downside of this technique is that it can have a high
variance. This means that differences in the training and test dataset can result in meaningful
differences in the estimate of accuracy. 

In the example below we split the Pima Indians dataset into 67%/33% splits for training and test and evaluate 
the accuracy of a Logistic Regression model
'''

import pandas as pd
from sklearn.model_selection import train_test_split 
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


#step 3:  split the dataset and train logistic regression model and then find accuracy by suing test dataset
#-----------------------------------------------

test_size = 0.33        # 33 % are used for test and 67 % for training out of total 768 samples
seed = 7                # to split test and train dataset randomly but same way each time algorithm is run

[X_train, X_test, Y_train, Y_test] = train_test_split(ML_data_input,ML_data_output, test_size= test_size, random_state= seed)
model = LogisticRegression()

model.fit(X_train, Y_train)              # generate or fit the model with training dataset
result = model.score(X_test, Y_test)      # find score of the model with test dataset

print(result)  # this gives score as 0.7559 = 75.59 % accuracy of the model.
