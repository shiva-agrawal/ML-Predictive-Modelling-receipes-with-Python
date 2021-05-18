'''
 Author      : Shiva Agrawal
 Date        : 04.09.2018
 Version     : 1.0
 Description : Naive Bayes classification Machine Learning Algorithm

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
Naive Bayes calculates the probability of each class and the conditional probability of each class
given each input value. These probabilities are estimated for new data and multiplied together,
assuming that they are all independent (a simple or naive assumption). When working with
real-valued data, a Gaussian distribution is assumed to easily estimate the probabilities for
input variables using the Gaussian Probability Density Function. 
We can construct a Naive Bayes model using the GaussianNB class
'''

import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB

# step 1 : Load the data
#--------------------------------------------------

CsvFileName = 'pima-indians-diabetes_without_header.csv'
header_names = ['PregnantCount','PlasmaGlu','BP','SkinThickness','SerumInsulinIn2Hours','BMI','PedigreeFunc','Age','Class']
ML_data = pd.read_csv(CsvFileName,names = header_names)
print(ML_data.shape)
print(ML_data)


# step 2: Separate input and output data for ML
#--------------------------------------------------

ML_data_array = ML_data.values
ML_data_input = ML_data_array[:,0:8]  # all rows and columns from index 0 to 7 (all input features)
ML_data_output = ML_data_array[:,8] # all rows and column index 8 (last column - Class (output))

# step 3: Model development
#--------------------------------------------------

num_folds = 10  # k= 10
seed = 7

kfold = KFold(n_splits=num_folds,random_state=seed) # this is for Kfold
model = GaussianNB() # this is model instance

model.fit(ML_data_input,ML_data_output) # this fits the model on the dataset
print(model.predict(ML_data_input[0:5,:])) # just prediction for some inputs (for understanding)

results = cross_val_score(model,ML_data_input,ML_data_output,cv=kfold, scoring='accuracy')
print(results.mean())

'''
output accuracy of the model using Kfold (k=10) CV is 75.51 %. We can also try other CV techniques to find accuracies.
I will recomment to use Kfold and test train split method.
Here we have developed the model in two steps. 
1. create instance of model
2. fit the model for the given dataset
Hence the variable 'model' contains the Naive Bayes model. 
'''


