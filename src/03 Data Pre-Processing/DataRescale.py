'''
 Author      : Shiva Agrawal
 Date        : 03.09.2018
 Version     : 1.0
 Description : Data pre-processing using rescaling 

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
When the data is comprised of attributes with varying scales, many machine learning algorithms
can benefit from rescaling the attributes to all have the same scale. Often this is referred to
as normalization and attributes are often rescaled into the range between 0 and 1. This is
useful for optimization algorithms used in the core of machine learning algorithms like gradient
descent. It is also useful for algorithms that weight inputs like regression and neural networks
and algorithms that use distance measures like k-Nearest Neighbors. You can rescale your data
using scikit-learn using the MinMaxScaler class.

For example: Attribute Pregnent count Xmin = 0, Xmax = 17

newValue = (x (oldvalue) - Xmin) / (Xmax - Xmin)

During scaling, in order to make all the 768 samples of this attribute between 0 and 1, 
Each sample is first subtracted by minValue = 0 and then divided by 17. 

ref: https://sebastianraschka.com/Articles/2014_about_feature_scaling.html
ref: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler   # for rescaling method

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


# Step 3: Rescale Data
#-----------------------------------------
scalar = MinMaxScaler(feature_range= (0,1))
ML_data_input_rescaled = scalar.fit_transform(ML_data_input);


# step 4: summarize the transform data
#-----------------------------------------
np.set_printoptions(3)
print(ML_data_input_rescaled[0:5,:])

