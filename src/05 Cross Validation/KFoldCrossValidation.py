'''
 Author      : Shiva Agrawal
 Date        : 04.09.2018
 Version     : 1.0
 Description : Cross validation using Kfold technique to find performance of the ML model

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
Cross validation is an approach that is use to estimate the performance of a machine
learning algorithm with less variance than a single train-test set split. It works by splitting
the dataset into k-parts (e.g. k = 5 or k = 10). Each split of the data is called a fold. The
algorithm is trained on k − 1 folds with one held back and tested on the held back fold. This is
repeated so that each fold of the dataset is given a chance to be the held back test set. After
running cross validation you end up with k different performance scores that we can summarize
using a mean and a standard deviation.

The result is a more reliable estimate of the performance of the algorithm on new data. It is
more accurate because the algorithm is trained and evaluated multiple times on different data.
The choice of k must allow the size of each test partition to be large enough to be a reasonable
sample of the problem, whilst allowing enough repetitions of the train-test evaluation of the
algorithm to provide a fair estimate of the algorithms performance on unseen data. For modest
sized datasets in the thousands or tens of thousands of records, k values of 3, 5 and 10 are
common. 
In the example below we use 10-fold cross validation.
'''

import pandas as pd
from sklearn.model_selection import KFold, cross_val_score 
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


#step 3: K-fold implemntation and finding accuracy of the model by cross validation
#-----------------------------------------------

seed = 7        # to split test and train dataset randomly but same way each time algorithm is run
num_folds = 10  # k = 10
kfold = KFold(n_splits=num_folds,random_state=seed)
model = LogisticRegression()

score = cross_val_score(model, ML_data_input, ML_data_output,cv = kfold)

print(score.mean()*100.0)
print(score.std()*100.0)

'''
mean and standard deviation helps to understand the accuracy of the model. Here mean*100 = 76.95 % i.e. Accuracy with
distribution of std()*100  = 4.84 % is good value
'''