'''
 Author      : Shiva Agrawal
 Date        : 04.09.2018
 Version     : 1.0
 Description : Different classification metrics to evaluate the acuuracy of the ML model

'''

'''
Classification Metrics
1. Classification Accuracy 
2. Logarithmic Loss
3. Area under ROC Curve
4. Confusion Matrics
5. Classification Report
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

import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression  # for classification metrics
from sklearn.metrics import confusion_matrix, classification_report

# step 1 : Load the data 
#--------------------------------------------------------

CsvFileName = 'pima-indians-diabetes_without_header.csv'
header_names = ['PregnantCount','PlasmaGlu','BP','SkinThickness','SerumInsulinIn2Hours','BMI','PedigreeFunc','Age','Class']
ML_data = pd.read_csv(CsvFileName,names = header_names)
print(ML_data.shape)
print(ML_data)


# step 2: Separate input and output data for ML
#--------------------------------------------------------

ML_data_array = ML_data.values
ML_data_input = ML_data_array[:,0:8]  # all rows and columns from index 0 to 7 (all input features)
ML_data_output = ML_data_array[:,8] # all rows and column index 8 (last column - Class (output))


# step 3: Calculate different metrics
#--------------------------------------------------------


# Metric 1: Classification Accuracy
'''
Classification accuracy is the number of correct predictions made as a ratio of all predictions
made. This is the most common evaluation metric for classification problems, it is also the most
misused. It is really only suitable when there are an equal number of observations in each class
(which is rarely the case) and that all predictions and prediction errors are equally important,
which is often not the case.
'''
num_folds = 10 # k=10
seed = 7

kfold = KFold(n_splits=num_folds, random_state=seed)
model = LogisticRegression()
scoringType = 'accuracy'
results = cross_val_score(model, ML_data_input, ML_data_output, cv=kfold, scoring=scoringType)  # here scoring decides the type of metrics used

print('-------Classification Metric 1: Classification Accuracy')
print(results.mean()*100.0)
print(results.std()*100.0)


# Metric 2: Logarithmic Loss metrics
'''
Logarithmic loss (or logloss) is a performance metric for evaluating the predictions of probabilities
of membership to a given class. The scalar probability between 0 and 1 can be seen as a measure
of confidence for a prediction by an algorithm. Predictions that are correct or incorrect are
rewarded or punished proportionally to the confidence of the prediction.
Smaller logloss is better with 0 representing a perfect logloss.
'''
num_folds = 10 # k=10
seed = 7

kfold = KFold(n_splits=num_folds, random_state=seed)
model = LogisticRegression()
scoringType = 'neg_log_loss'
results = cross_val_score(model, ML_data_input, ML_data_output, cv=kfold, scoring=scoringType)  # here scoring decides the type of metrics used

print('-------Classification Metric 2: Logarithmic Loss')
print(results.mean())
print(results.std())


# Metric 3: Area under ROC Curve
'''
Area under ROC Curve (or AUC for short) is a performance metric for binary classification problems. 
The AUC represents a model’s ability to discriminate between positive and negative classes. 
An area of 1.0 represents a model that made all predictions perfectly. An area of 0.5 represents a model that is 
as good as random. ROC can be broken down into sensitivity and specificity. 
A binary classification problem is really a trade-off between sensitivity and specificity.

- Sensitivity is the true positive rate also called the recall. It is the number of instances
from the positive (first) class that actually predicted correctly.
- Specificity is also called the true negative rate. Is the number of instances from the
negative (second) class that were actually predicted correctly.
'''

num_folds = 10 # k=10
seed = 7

kfold = KFold(n_splits=num_folds, random_state=seed)
model = LogisticRegression()
scoringType = 'roc_auc'
results = cross_val_score(model, ML_data_input, ML_data_output, cv=kfold, scoring=scoringType)  # here scoring decides the type of metrics used

print('-------Classification Metric 3: Area under ROC Curve')
print(results.mean())
print(results.std())


# Metric 4: Confusion Matrix
'''
The confusion matrix is a handy presentation of the accuracy of a model with two or more
classes. The table presents predictions on the x-axis and accuracy outcomes on the y-axis. The
cells of the table are the number of predictions made by a machine learning algorithm. For
example, a machine learning algorithm can predict 0 or 1 and each prediction may actually have
been a 0 or 1. Predictions for 0 that were actually 0 appear in the cell for prediction = 0 and
actual = 0, whereas predictions for 0 that were actually 1 appear in the cell for prediction = 0
and actual = 1. And so on.

ref for confusion matrix and related terms: https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/
'''

test_size = 0.33  # this is test and train sample split with test_size = 33%
seed = 7
[X_train, X_test, Y_train, Y_test] = train_test_split(ML_data_input,ML_data_output,test_size=test_size,random_state=seed)
model = LogisticRegression()
model.fit(X_train,Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test,predicted)

print('------------Classification Metric 4: Confusion Matrix----------')
print(matrix)

'''
Resultant matrix is 
[[141  21]
 [ 41  51]]
 Diagonal elements are correct predictions. We can see that compared to non-digonal, diadonal elemnets have high numbers.
 Hence model has comparatively good prerformance 
'''

# Metric 5: Classification Report
'''
The scikit-learn library provides a convenience report when working on classification prob-
lems to give you a quick idea of the accuracy of a model using a number of measures. The
classification report() function displays the precision, recall, F1-score and support for each
class. The example below demonstrates the report on the binary classification problem.

ref for all the terminologies: https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/

Precision (P) =  TP / (TP + FP)
Recall (R)    =  TP / (TP+ FN)
F-score       =  (2*P*R) / (P+R)   -  higher Fscore, good is algorithm
Support       = Total of actual 0 or 1
'''

test_size = 0.33  # this is test and train sample split with test_size = 33%
seed = 7

[X_train, X_test, Y_train, Y_test] = train_test_split(ML_data_input,ML_data_output,test_size=test_size,random_state=seed)
model = LogisticRegression()
model.fit(X_train,Y_train)
predicted = model.predict(X_test)
claReport = classification_report(Y_test,predicted)

print('------Classification Metric 5: Classification Report----------')
print(claReport)
