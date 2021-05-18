'''
 Author      : Shiva Agrawal
 Date        : 01.09.2018
 Version     : 1.0
 Description : Import dataset in CSV format into Python for machine learning model development using Pandas
 
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

# Import Dataset in CSV format using Pandas and storing in dataframe

import pandas as pd

CsvFileName = 'pima-indians-diabetes_without_header.csv'
header_names = ['PregnantCount','PlasmaGlu','BP','SkinThickness','SerumInsulinIn2Hours','BMI','PedigreeFunc','Age','Class']

ML_data = pd.read_csv(CsvFileName,names = header_names)

print(ML_data.shape)    # this will print complete dataset with all the samples
print(ML_data)          # it is (768, 9) - Hence 768 rows (samples) and 9 columns (features and output)

