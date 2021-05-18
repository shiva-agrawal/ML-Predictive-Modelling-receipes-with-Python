'''
 Author      : Shiva Agrawal
 Date        : 05.09.2018
 Version     : 1.0
 Description : Comparison of ML models using common valiation, test and metrics. Here classification type of models are compared
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
Six different classification algorithms are compared on a single dataset:
1. Logistic Regression.
2. Linear Discriminant Analysis.
3. k-Nearest Neighbors.
4. Classification and Regression Trees.
5. Naive Bayes.
6. Support Vector Machines.
'''


# for importing dataset from CSV and for Dataframe
import pandas as pd

# for CV of type K-fold
from sklearn.model_selection import KFold, cross_val_score

# for each model development
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import  GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# for Boxplot - visualization
import matplotlib.pyplot as pyplt



# step 1 : Load the data 
#-------------------------------------------------------

CsvFileName = 'pima-indians-diabetes_without_header.csv'
header_names = ['PregnantCount','PlasmaGlu','BP','SkinThickness','SerumInsulinIn2Hours','BMI','PedigreeFunc','Age','Class']
ML_data = pd.read_csv(CsvFileName,names = header_names)
print(ML_data.shape)
print(ML_data)

# step 2: Separate input and output data for ML
#-------------------------------------------------------

ML_data_array = ML_data.values
ML_data_input = ML_data_array[:,0:8]  # all rows and columns from index 0 to 7 (all input features)
ML_data_output = ML_data_array[:,8] # all rows and column index 8 (last column - Class (output))


# step 3: Prepare all the models
#-------------------------------------------------------

models = []  # creating an empty list and then adding each model into it
models.append(('LR',LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM',SVC()))

print(models)


# Step 4: Comapare each ML model
#-------------------------------------------------------

# It means finding accuracies (mean and std) of each model.
# CV used here is K-fold (k=10)
cv_results = []
model_names = []

for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_result = cross_val_score(model,ML_data_input,ML_data_output,cv=kfold,scoring='accuracy')
    cv_results.append(cv_result)
    model_names.append(name)
    print ('MODEL - ' + str(name) + ' : ' + str(cv_result.mean()) + ' ('+ str(cv_result.std()) + ' )')

'''
Results:
MODEL - LR : 0.7695146958304853 (0.04841051924567195 )
MODEL - LDA : 0.773462064251538 (0.05159180390446138 )
MODEL - KNN : 0.7265550239234451 (0.06182131406705549 )
MODEL - CART : 0.7017088174982912 (0.07091634617635803 )
MODEL - NB : 0.7551777170198223 (0.04276593954064409 )
MODEL - SVM : 0.6510252904989747 (0.07214083485055327 )

From results, we can see that LR and LDA are best fit as compared to other models. They have high mean accuracies and low std dev.
'''

# step 5: Compare by Visualization
#-------------------------------------------------------

# here we have used Box plot - but other visualizations can also be used
fig = pyplt.figure()
fig.suptitle('Compariosn of Classification Algorithm results')
ax = fig.add_subplot(111)
pyplt.boxplot(cv_results)
ax.set_xticklabels(model_names)
pyplt.show()

