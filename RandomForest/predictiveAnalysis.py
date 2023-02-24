# ---------------------------------------------------------------------
# ---------------------Predictive Analysis v.1.0.0---------------------
# Created by: David Hodge
# Created on: 2020-2-21
# Description: This script is used for performing a predictive analysis
#            on a given dataset
# Version:
#       1.0.0 - Initial script created 
#
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------

# Import libraries
import pandas as pd

from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
#
# Method: BuildPAM
    # Inputs:
    #     pipeline = A pipeline object
    #     parameters = A dictionary object that holds the parameter
    #         array
    #     X_train = Matrix of predictor variables from a test dataframe
    #         object
    #     y_train = Matrix of target variable from a test dataframe
    #         object
    # Processing:
    #     The BuildPAM method builds the classification analysis model
    # Output:
    #     Returns the optimized fitted model
#
def BuildPAM(pipeline, parameters, X_train, y_train):
    # Use grid search to identify the hyperparameters to optimize the
    # classification analysis
    print("Using grid search to tune the hyperparameters...")
    cm = GridSearchCV(pipeline,
                      param_grid = parameters,
                      scoring='accuracy',
                      cv=10,
                      n_jobs=-1)
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
             ('dtc', DecisionTreeClassifier(random_state=1)),]
    pipeline = Pipeline(steps)
#
    parameters = {'dtc__max_depth': [3,4,5,6],
                  'dtc__min_samples_leaf': [0.04, 0.06, 0.08],
                  'dtc__max_features': [0.2, 0.4, 0.6, 0.8]}
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
def ModelMetrics(model, X_test, y_test):
    # Determine the metrics of the model, and print the results
    cvScore = model.best_score_
    bestModel = model.best_estimator_
    score = bestModel.score(X_test,
                            y_test)
    params = model.best_params_
#
    print("")
    print("Best CV Accuracy: {0}".format(cvScore))
    print("Model Accuracy: {0}".format(score))
    print("Best hyperparameters:\n{0}".format(params))
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
    print("")
    print("Mean Squared Error 'MSE': {:.2f}".format(MSE(y_test,
                                                        y_pred)))
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
               'NVM2 Task 2 - Predictive Analysis\\')
    outDir = (directory +
              'Output\\')
    dataset = (directory +
               'churn_clean.csv')
    removeFields = ['Customer_id',
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
    cm = BuildPAM(pipeline,
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
                 y_test)
#