'''
 Author      : Shiva Agrawal
 Date        : 05.09.2018
 Version     : 1.0
 Description : Save and Load ML Model in Python using package Pickle.
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
Pickle is the standard way of serializing objects in Python. We can use the pickle operation
to serialize your machine learning algorithms and save the serialized format to a file. 
Later the saved model can be loaded and deserialize can be used to make new predictions.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle


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


# step 4: Save the model uisng pickle
#----------------------------------------------------------------

filename = 'model_using_pickle.sav'
pickle.dump(model, open(filename, 'wb'))

# step 5: Later...... Load the saved model using pickle
#----------------------------------------------------------------

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)

# test the loaded model
print(result)

'''
Accuracy is 75.59 % 
'''
