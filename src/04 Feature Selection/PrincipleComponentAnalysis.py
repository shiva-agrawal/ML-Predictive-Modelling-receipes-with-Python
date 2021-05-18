'''
 Author      : Shiva Agrawal
 Date        : 03.09.2018
 Version     : 1.0
 Description : Principle Component Analysis (PCA) for ML model development

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
Principal Component Analysis (or PCA) uses linear algebra to transform the dataset into a
compressed form. Generally this is called a data reduction technique. A property of PCA is that
you can choose the number of dimensions or principal components in the transformed result. In
the example below, we use PCA and select 3 principal components. 

Note that PCA reduces the initial features into completely new features derived from initial features but number of 
features are less. Hence after using PCA, the input features from the dataset are no more valid to generate model. 
Model must be generated from new features only.

How to select number of new features: Always start with lowest possible and then generate model and if model 
doesn't provide required results then increase number of features for PCA

ref: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

Source (Anrew NG lectures - Stanford University)
- At first reduce parameters/features without using PCA if possible.
- Do not use PCA to remove overfitting, instead use regularization
- Use PCA only when the feature reduction is very necessary and it is not possible with other mormal methods
'''


import pandas as pd
from numpy import set_printoptions
from sklearn.decomposition import PCA


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


# step 3: Feature extraction using PCA
#------------------------------------------

pca_results = PCA(n_components= 4)      # create new 4 features from input 8 features
fits = pca_results.fit_transform(ML_data_input)

# step 4: summarize components
#------------------------------------------
print(fits)  # this contains 4 new features
