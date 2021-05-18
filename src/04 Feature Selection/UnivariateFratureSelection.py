'''
 Author      : Shiva Agrawal
 Date        : 03.09.2018
 Version     : 1.0
 Description : Univariate selection of the features for ML model development

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
Statistical tests can be used to select those features that have the strongest relationship with
the output variable. The scikit-learn library provides the SelectKBest class 2 that can be used
with a suite of different statistical tests to select a specific number of features. The example
below uses the chi-squared (chi 2 ) statistical test for non-negative features to select 4 of the best
features from the Pima Indians onset of diabetes dataset. 
There are also other methods to extract features based on statics

ref: http://scikit-learn.org/stable/modules/feature_selection.html#feature-selection

Further details of chi square test: http://www.statisticshowto.com/probability-and-statistics/chi-square/
'''

import pandas as pd
from numpy import set_printoptions
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

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


#step 3: Univariate feature extraction
#-----------------------------------------

test = SelectKBest(score_func = chi2, k = 4 ) # function is chi2 and for reducing to 4 features
fits = test.fit(ML_data_input, ML_data_output)


# step 4: summarize scores
#-----------------------------------------

set_printoptions(precision=3)
print(fits.scores_)


#step 5: summarize selected features
#------------------------------------------
features = fits.transform(ML_data_input)
print(features[0:5,:])

