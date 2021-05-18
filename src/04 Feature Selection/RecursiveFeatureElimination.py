'''
 Author      : Shiva Agrawal
 Date        : 03.09.2018
 Version     : 1.0
 Description : Recursive Feature elimination for ML model development

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
The Recursive Feature Elimination (or RFE) works by recursively removing attributes and
building a model on those attributes that remain. It uses the model accuracy to identify which
attributes (and combination of attributes) contribute the most to predicting the target attribute.
One can learn more about the RFE class in the scikit-learn documentation. The example below
uses RFE with the logistic regression algorithm to select the top 3 features. The choice of
algorithm does not matter too much as long as it is skillful and consistent.

ref: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE
'''


import pandas as pd
from numpy import set_printoptions
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# step 1 : Load the data 
#----------------------------------------

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


#step 3: Recursive Feature Elimination (RFE)
#-----------------------------------------

model = LogisticRegression()   # to create model for RFE
rfe_result = RFE(model, n_features_to_select= 3)  # this will recursively eliminate features upto remaining best fit 3
fits = rfe_result.fit(ML_data_input, ML_data_output)

selected_features = fits.transform(ML_data_input)

print("Num Features: " + str(fits.n_features_))
print("Selected Features: " + str(fits.support_))
print("Feature Ranking: %s" + str(fits.ranking_))
print(selected_features)
