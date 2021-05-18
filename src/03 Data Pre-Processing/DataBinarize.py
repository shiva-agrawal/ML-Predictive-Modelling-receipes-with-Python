'''
 Author      : Shiva Agrawal
 Date        : 03.09.2018
 Version     : 1.0
 Description : Data pre-processing using Binarization 

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
It means transforming the data using a binary threshold. All values above the threshold are
marked 1 and all equal to or below are marked as 0. This is called binarizing data or
thresholding data. It can be useful when we have probabilities that we want to make crisp
values. It is also useful when feature engineering and we want to add new features that indicate
something meaningful. One can create new binary attributes in Python using scikit-learn with
the Binarizer class

ref: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html#sklearn.preprocessing.Binarizer
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import Binarizer # Binarization method

# step 1 : Load the data 
#----------------------------------------
CsvFileName = 'pima-indians-diabetes_without_header.csv'
header_names = ['PregnantCount','PlasmaGlu','BP','SkinThickness','SerumInsulinIn2Hours','BMI','PedigreeFunc','Age','Class']
ML_data = pd.read_csv(CsvFileName,names = header_names)

print(ML_data.shape)
print(ML_data)


# step 2: separate input and output for ML
#-----------------------------------------
ML_data_array = ML_data.values
ML_data_input = ML_data_array[:,0:8]  # all rows and columns from index 0 to 7 (all input features)
ML_data_output = ML_data_array[:,8] # all rows and column index 8 (last column - Class (output))


# Step 3: Binarize Data
#-----------------------------------------
scalar = Binarizer(threshold=10).fit(ML_data_input)   # threshold = 10  is used just for example
ML_data_input_binarized = scalar.transform(ML_data_input)


# step 4: summarize the transform data
#-----------------------------------------
np.set_printoptions(3)
print(ML_data_input_rescaled[0:5,:])

