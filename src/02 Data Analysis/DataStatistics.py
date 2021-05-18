'''
 Author      : Shiva Agrawal
 Date        : 02.09.2018
 Version     : 1.0
 Description : Different ways to find statistics of the dataset to do analysis

'''

'''
Informtion about the used dataset: pima-indians-diabetes

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
9 is output. It is logistic regression or binary classification example
# 9. Class variable (0 or 1)
'''

'''
Following are the statistics covered for the data

1. Take a peek / look at your raw data.
2. Review the dimensions of your dataset.
3. Review the data types of attributes in your data.
4. Summarize the distribution of instances across classes in your dataset.
5. Summarize your data using descriptive statistics.
6. Understand the relationships in your data using correlations.
7. Review the skew of the distributions of each attribute.
'''

import pandas as pd
import matplotlib.pyplot as plt

# step 1 : Load the data
#-------------------------------------

CsvFileName = 'pima-indians-diabetes_without_header.csv'
header_names = ['PregnantCount','PlasmaGlu','BP','SkinThickness','SerumInsulinIn2Hours','BMI','PedigreeFunc','Age','Class']
ML_data = pd.read_csv(CsvFileName,names = header_names)
print(ML_data.shape)
print(ML_data)


# Step 2: Statistics of the raw data
#--------------------------------------

# 1. Take a peek / look at raw data.
# as in console we can not see the data completely, it is advisable to use debugger mode and then variable pane.
# then click the variable and it will open just like Matlab view for visualizing complete data
peek = ML_data.head(20)  # provides first 20 rows of the data
print(peek)


# 2. Review the dimensions of your dataset.
print(ML_data.shape)  # returns (768, 9) - 768 samples and 9 (features and ouput)

# 3. Review the data types of attributes in your data.
print(ML_data.dtypes) # returns data types of each attribute
ML_data['BP'] = ML_data['BP'].astype(float)  # way to change the data type
print(ML_data.dtypes)


# 4. Summarize the distribution of instances across classes in your dataset. (only for classification problems)
'''
On classification problems you need to know how balanced the class values are. Highly imbalanced
problems (a lot more observations for one class than another) are common and may need special
handling in the data preparation stage of your project.
'''
print(ML_data.groupby('Class').size())  # provides sample count with class 0 value and 1 value


# 5. Summarize your data using descriptive statistics.
'''
This means finding some statistical parameters like mean, median, std dev, etc of each attribute 
and then analyze them to understand the distribution of the data. We can also plot differet attributes together or in
some combinations to understand the underlysing relationships
'''

'''
The describe() function on the Pandas DataFrame lists 8 statistical properties of each attribute. They are:
1. Count.
2. Mean.
3. Standard Deviation.
4. Minimum Value.
5. 25th Percentile.
6. 50th Percentile (Median).
7. 75th Percentile.
8. Maximum Value.
'''
pd.set_option('display.width', 100)
pd.set_option('precision',3)
statistics = ML_data.describe()

print(statistics)   # display of all the 9 attributes
print(statistics['Class']) # display only of class attribute

# 6. Understand the relationships in your data using correlations.
'''
Correlation refers to the relationship between two variables and how they may or may not change together. 
The most common method for calculating correlation is Pearson’s Correlation Coefficient, that assumes 
a normal distribution of the attributes involved. A correlation of -1 or 1 shows a full negative or positive correlation respectively.
Whereas a value of 0 shows no correlation at all.  Some machine learning algorithms like linear and logistic regression can suffer 
poor performance if there are highly correlated attributes in your dataset. 

As such, it is a good idea to review all of the pairwise correlations of the attributes in your dataset. 
You can use the corr() function on the Pandas DataFrame to calculate a correlation matrix.
'''

correlation_matrix = ML_data.corr()
print(correlation_matrix)

# 7. Review the skew of the distributions of each attribute

'''
Skewness: In probability theory and statistics, skewness is a measure of the asymmetry of the probability distribution 
of a real-valued random variable about its mean. The skewness value can be positive or negative, or undefined. 
https://en.wikipedia.org/wiki/Skewness

Skew refers to a distribution that is assumed Gaussian (normal or bell curve) that is shifted or squashed in one 
direction or another. Many machine learning algorithms assume a Gaussian distribution. 
Knowing that an attribute has a skew may allow you to perform data preparation to correct the skew and  
later improve the accuracy of your models.
The skew result show a positive (right) or negative (left) skew. Values closer to zero show
less skew.
'''

skewness = ML_data.skew()
print(skewness)

