'''
 Author      : Shiva Agrawal
 Date        : 04.09.2018
 Version     : 1.0
 Description : Cross validation using Random repeat test-train split to find performance of the ML model

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
This is another variation on k-fold cross validation to create a random split of the data like the
train/test split described above, but repeat the process of splitting and evaluation of the
algorithm multiple times, like cross validation. This has the speed of using a train/test split and
the reduction in variance in the estimated performance of k-fold cross validation. 
We can also repeat the process many more times as needed to improve the accuracy. A down side is that
repetitions may include much of the same data in the train or the test split from run to run,
introducing redundancy into the evaluation. 
The example below splits the data into a 67%/33% train/test split and repeats the process 10 times.
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


#step 3: Repeat Random Train-Test splits implemntation and finding accuracy of the model by cross validation
#-----------------------------------------------

num_folds = 10       # k = 10
seed = 7             # to split test and train dataset randomly but same way each time algorithm is run
test_size = 0.33     # 33 % samples randomly selected as test samples

Repeatkfold = ShuffleSplit(n_splits=num_folds,test_size = test_size,random_state=seed)
model = LogisticRegression()

score = cross_val_score(model, ML_data_input, ML_data_output,cv = Repeatkfold)

print(score.mean()*100.0)
print(score.std()*100.0)

