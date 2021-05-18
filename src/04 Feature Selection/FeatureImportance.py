'''
 Author      : Shiva Agrawal
 Date        : 03.09.2018
 Version     : 1.0
 Description : Feraute selection by finding feature importance for ML model development

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

9 is output. It is logistoc regression or binary classification example
# 9. Class variable (0 or 1)
'''

'''
Bagged decision trees like Random Forest and Extra Trees can be used to estimate the importance
of features. In the example below we construct a ExtraTreesClassifier classifier for the Pima
Indians onset of diabetes dataset. You can learn more about the ExtraTreesClassifier class
in the scikit-learn API.

ref: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
'''


import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier


# step 1 : Load the data 
#-----------------------------------------

CsvFileName = 'pima-indians-diabetes_without_header.csv'
header_names = ['PregnantCount','PlasmaGlu','BP','SkinThickness','SerumInsulinIn2Hours','BMI','PedigreeFunc','Age','Class']
ML_data = pd.read_csv(CsvFileName,names = header_names)
print(ML_data.shape)
print(ML_data)


# step 2: separate input and output for ML
#------------------------------------------

ML_data_array = ML_data.values
ML_data_input = ML_data_array[:,0:8]  # all rows and columns from index 0 to 7 (all input features)
ML_data_output = ML_data_array[:,8] # all rows and column index 8 (last column - Class (output))

# step 3: finding feature importance using bagged decision trees
#-------------------------------------------

model = ExtraTreesClassifier()
model.fit(ML_data_input, ML_data_output)
print(model.feature_importances_)

'''
NOTE: The output of model.feature_importances_ is [0.111 0.222 0.109 0.086 0.077 0.138 0.125 0.131]
The more the importance score, the important is feature. 
One can then select the more importance features and develop model.
'''
