# Machine Learning Recipes for Predictive Modelling using Python.

Model development using Machine Learning involves mutliple steps of process from Import of the Dataset , pre-processing... to finally save the developed model

Aim of this project is to provide all the recipes well documented and tested by me in python. I have developed these using Pycharm community IDE with Ubuntu 16.04 OS.

Scikit-learn library developed for python is used extensively during the development. Other packages include scipy, numpy, pandas and matplotlib.

Two datasets are used:
1. Pima Indian Diabetes data (Binary classification type)
2. Boston House prices data (Regression type)

Folder structure:
1. doc
	* It contains project details.pdf which contains complete project details in form of short report.
2. src
	* it contains all the recipes in sub folders format and also the datasets.
  
# Table of Content

1. Load CSV data into Python	
	* using python standard library
	* using Numpy
	* Using Pandas - Dataframe 
2. Data analysis	
	* Using statistics
		* peek / look at raw data.
   	 	* Dimensions of your data
   		* Data types of attributes of the data.
   		* Summarize the distribution of instances across classes in dataset
	 	* Summarize data using descriptive statistics like mean, median, variance, etc.
	  	* Understand the relationships in data using correlations
	 	* Skew of the distributions of each attribute
	* Using plots
		* Histograms
	 	* Density Plots
	  	* Box and Whisker plots
	  	* Correlation matrix plot
	 	* Scatter plot Matrix 
3. Data pre-processing	
	* Rescale Data
	* Standardize Data
	* Normalize Data
	* Binarize Data
4. Feature Selection	
	* Univariate selection
	* Recursive Feature Elimination
	* Principle Component Analysis (PCA)
	* Feature Importance
5. Performance Evaluation with cross validation	
	* Train and Test datasets
	* k-fold Cross Validation
	* Leave One out Cross Validation
	* Repeat Random Test-Train splits
6. Performance Metrics	
	* Classification Metrics
		* Classification Accuracy 
		* Logarithmic Loss
		* Area under ROC Curve
		* Confusion Matrix
		* Classification Report
	* Regression Metrics
		* Mean Absolute Error
		* Mean Square Error
		* R2 (R square)
7. ML Classification Algorithms
	* Linear
		* Logistic Regression
		* Linear Discriminant Analsis
	* Non-linear
		* K-Nearest Neighbour (KNN)
		* Naive Byes
		* Classification and Regression Trees
		* Support Vector Machine (SVM)
8. ML Regression Algorithms
	* Linear
		* Linear Regression
		* Ridge Regression
		* LASSO Linear Regression
		* Elastic net Regression
	* Non-linear
		* K-Nearest Neighbour (KNN)
		* Classification and Regression Trees
		* Support Vector Machine (SVM)
9. Compare algorithms	
10. Save and Load ML model(s)	
	* using Pickle package
	* suing Joblib package




