# ---------------------------------------------------------------------
# -------------------Classification Analysis v.1.0.0-------------------
# Created by: David Hodge
# Created on: 2020-2-12
# Description: This script is used for performing a classification
#            analysis on a given dataset
# Version:
#       1.0.0 - Initial script created 
#
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
#
# Method: BuildCAM
    # Inputs:
    #     pipeline = A pipeline object
    #     parameters = A dictionary object that holds the parameter
    #         array
    #     X_train = Matrix of predictor variables from a test dataframe
    #         object
    #     y_train = Matrix of target variable from a test dataframe
    #         object
    # Processing:
    #     The BuildCAM method builds the classification analysis model
    # Output:
    #     Returns the optimized fitted model
#
def BuildCAM(pipeline, parameters, X_train, y_train):
    # Use grid search to identify the hyperparameters to optimize the
    # classification analysis
    print("Using grid search to tune the hyperparameters...")
    cm = GridSearchCV(pipeline,
                      param_grid = parameters)
#
    # Fit the training dataset to the pipeline object
    print("Fitting the classification model...")
    cm.fit(X_train,
           y_train)
    return cm
#
# Method: CleanDF
    # Inputs:
    #     df = A Pandas dataframe object
    #     output = Folder directory as string object
    # Processing:
    #     The csvWrite method writes the object values to a CSV file.
    # Output:
    #     Returns a new "cleaned" dataframe object
#
def CleanDF(df, output, fields):
#
    # Create a copy of the dataframe to be used for cleaning
    print("Cleaning the dataset...")
    df_clean = df.copy()
#
    # Delete any column from the dataframe that is in the list of
    # fields to be removed
    for column_name in df_clean.columns:
        if column_name in fields:
            del df_clean[column_name]
#
    # Create a dummy variable for any non-numeric column in the
    # dataframe
    print("Creating dummy variables...")
    df_clean = pd.get_dummies(df_clean,
                              drop_first = True)
#
    # Save the dataframe to the output directory
    df_clean.to_csv(output +
                    'Churn_Final_Results.csv')
    return df_clean
#
# Method: CreatePipeline
    # Inputs:
    #     None
    # Processing:
    #     The CreatePipeline creates a pipeline object used to build a
    #     model
    # Output:
    #     Returns a KNN pipeline object and the parameters used for
    #     hyperparameter tuning
#
def CreatePipeline():
    # Setup the Pipeline that will be used to classify the dataset.
    # Define the hyperparameter settings
    print("Creating pipeline object...")
    steps = [('scaler', StandardScaler()),
             ('select', RFE(estimator = DecisionTreeClassifier())),
             ('knn', KNeighborsClassifier()),]
    pipeline = Pipeline(steps)
#
    parameters = {'knn__n_neighbors': np.arange(1, 25)}
    return pipeline, parameters
#
# Method: ModelMetrics
    # Inputs:
    #     model = A classification model object
    #     X_test = Matrix of predictor variables from a test dataframe
    #         object
    #     y_test = Matrix of target variable from a test dataframe
    #         object
    #     y_pred = Matrix of predicted target variable values from a
    #         test dataframe object
    # Processing:
    #     The ModelMetrics method prints the metrics from a
    #     classification model
    # Outputs:
    #     None
#
def ModelMetrics(model, X_test, y_test, y_pred):
    # Determine the metrics of the model, and print the results
    score = accuracy_score(y_test,
                           y_pred)
    report = classification_report(y_test,
                                   y_pred)
    params = model.best_params_
#
    print("")
    print("Model Accuracy: {0}".format(score))
    print("")
    print("Classification Report:\n{0}".format(report))
    for value in params.values():
        print("Optimized Model Parameters: {0}".format(value))
#
# Method: PredictModel
    # Inputs:
    #     model = A classification model object
    #     X_test = Matrix of predictor variables from a test dataframe
    #         object
    #     y_test = Matrix of target variable from a test dataframe
    #         object
    #     X_select = List of target variables selected to use in an
    #         optimized linear regression model
    #     output = Folder directory as string object
    # Processing:
    #     The PredictModel method predicts the target variable for
    #     the test dataframe, and creates a plot of the residual error
    #     for the predictions.
    # Output:
    #     None
#
def PredictModel(model, X_test, y_test, output, name):
#
    # Predict the target variable, and save the predictions to a .csv
    # file
    print("Running predictions...")
    y_pred = model.predict(X_test)
    df = pd.DataFrame({'Actual': y_test,
                       'Predicted': y_pred})
    df.to_csv(output +
              '{}_PredictResults.csv'.format(name))
#
    # Determine the AUC scores
    y_prob = model.predict_proba(X_test)[:,1]
    falsePos, truePos, threshhold = roc_curve(y_test,
                                              y_prob)
#
    # Create the ROC curve
    print("Plotting ROC curve...")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(falsePos,
             truePos,
             label='K Nearest Neighbor')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('KNN ROC Curve')
    plt.savefig(output +
                '{}.png'.format('KNN ROC Curve'))
    plt.close() 
    print("")
    print("AUC Score: {0}".format(roc_auc_score(y_test,
                                                y_prob)))
    return y_pred
#
# Method: SplitDataset
    # Inputs:
    #     X = Matrix of predictor variables from a dataframe object
    #     y = Matrix of target variable from a dataframe object
    # Processing:
    #     The SplitDataset method splits the input dataframe object
    #     into seperate training and testing dataframes
    # Output:
    #     X_train = Matrix of predictor variables from a training
    #         dataframe object
    #     y_train = Matrix of target variable from a training dataframe
    #         object
    #     X_test = Matrix of predictor variables from a test dataframe
    #         object
    #     y_test = Matrix of target variable from a test dataframe
    #         object
#
def SplitDataset(X,y,output):
#
    # Split the dataframe into training and testing dataframes
    print("Splitting the dataframe into training and testing datasets...")
    X_train, X_test, y_train, y_test = train_test_split(\
        X,
        y,
        test_size=0.3,
        random_state=21)
    train = [X_train, y_train]
    trainDF = pd.concat(train, axis=1, join="inner")
    trainDF.to_csv(output +
                   'Churn_train.csv')
    test = [X_test, y_test]
    testDF = pd.concat(test, axis=1, join="inner")
    testDF.to_csv(output +
                   'Churn_test.csv')
    return X_train, X_test, y_train, y_test
#
# Method: SummarizeData
    # Inputs:
    #     df = A Pandas dataframe object
    #     output = Folder directory as string object
    # Processing:
    #     The SummarizeData method creates a .csv file with information
    #     summarizing each variable in the dataframe
    # Output:
    #     None
#
def SummarizeData(df, output):
#
    # Summarize the variables in the dataframe
    print("Summarizing the dataset...")
    Churn_clean.describe().to_csv(output +
                                  'Churn_summary.csv')
#
# Method: Main
    # Inputs:
    #     None
    # Processing:
    #     The main method initializes the variables and runs the
    #     program
    # Output:
    #     None
#
if __name__ == '__main__':
#
    # Initialize the variables to use in the program
    directory = ('I:\\david\\Documents\\School\\WGU\\Courses\\' +
               'D209_Data_Mining_I\\' +
               'NVM2 Task 1 - Classification Analysis\\')
    outDir = (directory +
              'Output\\')
    dataset = (directory +
               'churn_clean.csv')
    removeFields = ['CaseOrder',
                    'Customer_id',
                    'Interaction',
                    'UID',
                    'County',
                    'City',
                    'State',
                    'Lat',
                    'Lng',
                    'TimeZone',
                    'Job',
                    'Zip',
                    'Population']
    target = 'InternetService_Fiber Optic'
#
    # Create a Pandas dataframe object from the .csv dataset
    Churn = pd.read_csv(dataset)
#
    # Clean the dataframe, summarize, and then standardize the data
    Churn_clean = CleanDF(Churn,
                          outDir,
                          removeFields)
    SummarizeData(Churn_clean,
                  outDir)
#
    # Set the feature and target variables
    X = Churn_clean.loc[:,Churn_clean.columns != target]
    y = Churn_clean[target]
#
    # Create a pipeline object and get model parameters
    pipeline, parameters = CreatePipeline()
#
    # Split the dataframe into training and testing datasets
    X_train, X_test, y_train, y_test = SplitDataset(X, y, outDir)
#
    # Build the classification model
    cm = BuildCAM(pipeline,
                  parameters,
                  X_train,
                  y_train)
#
    # Predict the target variable using the classification model
    y_pred = PredictModel(cm,
                          X_test,
                          y_test,
                          outDir,
                          'classification')
#
    # Print the model metrics to the console
    ModelMetrics(cm,
                 X_test,
                 y_test,
                 y_pred)
#