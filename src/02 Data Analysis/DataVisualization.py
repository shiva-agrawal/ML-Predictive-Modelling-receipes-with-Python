'''
 Author      : Shiva Agrawal
 Date        : 01.09.2018
 Version     : 1.0
 Description : Different types of univariate and multivariate plots to understand the data and to do analysis

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
9 is output. It is logistic regression or binary classification example
# 9. Class variable (0 or 1)
'''

'''
Visulaization plots are:

1. Univariate plots
    1. Histograms
    2. Density Plots
    3. Box and Whisker plots

2. Multivariate plots
    1. Correlation matrix plot
    2. Scatter plot Matrix 
	
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# step 1 : Load the data
#-----------------------------------------

CsvFileName = 'pima-indians-diabetes_without_header.csv'
header_names = ['PregnantCount','PlasmaGlu','BP','SkinThickness','SerumInsulinIn2Hours','BMI','PedigreeFunc','Age','Class']
ML_data = pd.read_csv(CsvFileName,names = header_names)
print(ML_data.shape)
print(ML_data)


# step 2: Univariate plots
#-----------------------------------------

# 1. Histogram
'''
A fast way to get an idea of the distribution of each attribute is to look at histograms. Histograms group data 
into bins and provide you a count of the number of observations in each bin. 
From the shape of the bins you can quickly get a feeling for whether an attribute is Gaussian, skewed or even has an exponential distribution. 
It can also help you see possible outliers.
'''
ML_data.hist()  # for all the attributes

# for only one attribute (example)
ML_data['BP'].hist()  
plt.xlabel('BP')
plt.ylabel('count')


# 2. Density plots
'''
Density plots are another way of getting a quick idea of the distribution of each attribute. The
plots look like an abstracted histogram with a smooth curve drawn through the top of each bin,
much like your eye tried to do with the histograms.
'''

ML_data.plot(kind = 'density', subplots = True, layout = (3,3), sharex = False)

# 3. Box and Whisker plots
'''
Boxplots summarize the distribution of each attribute, drawing a line for the median (middle value) 
and a box around the 25th and 75th percentiles (the middle 50% of the data). 
The whiskers give an idea of the spread of the data and dots outside of the whiskers
show candidate outlier values (values that are 1.5 times greater than the size of spread of the middle 50% of the data).
'''
ML_data.plot(kind = 'box', subplots = True, layout = (3,3), sharex = False, sharey = False)  # one way for all attributes

plt.figure()
plt.boxplot(ML_data['BP']) # to plot single attribute



# step 3: Multivariate plots
#-----------------------------------------

# 1. Correlation Matrix plot  - help to understand the interection between attributes of the dataset
'''
First find correlation between features and then plot in matrix form
'''

correlationMatrix = ML_data.corr()
print(correlationMatrix)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlationMatrix)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(header_names)
ax.set_yticklabels(header_names)

# 2 Scatter plot Matrix
'''
A scatter plot shows the relationship between two variables as dots in two dimensions, one axis for each attribute. 
You can create a scatter plot for each pair of attributes in your data. Drawing all these scatter plots together 
is called a scatter plot matrix. Scatter plots are useful for spotting structured relationships between variables, 
like whether you could summarize the relationship between two variables with a line. 
Attributes with structured relationships may also be correlated and good candidates for removal from dataset.
'''

pd.tools.plotting.scatter_matrix(ML_data)  # Careful: this is complex calling ....
plt.plot()

# suggestion / note: The resultant plots are two many and hence difficult to analyze and it is good to develop less or individual
#  plots and then analyze. Ex. Individual scatter plots or multiple scatter plots as subplots

# show all the plots
plt.show()


