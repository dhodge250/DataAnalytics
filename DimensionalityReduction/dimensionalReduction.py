# ---------------------------------------------------------------------
# --------------------Dimensional Reduction v.1.0.0--------------------
# Created by: David Hodge
# Created on: 2021-4-10
# Description: This script is used for performing a clustering analysis
#            on a given dataset utilizing Principal Component
#            Analysis (PCA)
# Version:
#       1.0.0 - Initial script created 
#
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
    #     The BuildCAM method builds the clustering analysis model
    # Output:
    #     Returns the optimized fitted model
#
def BuildCluster(pipeline, X_train, y_train):
#
    # Fit the training dataset to the pipeline object
    print("Fitting the clustering model...")
    pipeline.fit(X_train,
                 y_train)
    return pipeline
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
    # Delete any column from the dataframe that is not in the list of
    # fields to be kept
    for column_name in df_clean.columns:
        if column_name not in fields:
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
    #     Returns a KMeans pipeline object
#
def CreatePipeline():
    # Setup the Pipeline that will be used to cluster the dataset.
    print("Creating pipeline object...")
    steps = [('scaler', StandardScaler()),
             ('reduce', PCA(n_components = 4)),
             ('kmeans', KMeans(n_clusters = 3)),]
    pipeline = Pipeline(steps)
#
    return pipeline
#
# Method: ModelMetrics
    # Inputs:
    #     model = A clustering model object
    #     X_test = Matrix of predictor variables from a test dataframe
    #         object
    #     y_test = Matrix of target variable from a test dataframe
    #         object
    #     y_pred = Matrix of predicted target variable values from a
    #         test dataframe object
    # Processing:
    #     The ModelMetrics method prints the metrics from a
    #     clustering model
    # Outputs:
    #     None
#
def ModelMetrics(model, X_test, y_test, y_pred, output):
#
    # Determine the metrics of the model, and print the results
    score = accuracy_score(y_test,
                           y_pred)
    var = model.steps[1][1].explained_variance_ratio_.cumsum()
    tolvar = model.steps[1][1].explained_variance_ratio_
#
    print("")
    print("Model Accuracy: {0}".format(score))
    print("Number of Components: {0}".format(len(
                                            model.steps[1][1].components_)))
    print("Explained Variance for each Principal Component: {0}".format(
                                                            tolvar.round(3)))
    print("Total Variance for Principal Components: {0}".format(
                                                        sum(tolvar).round(2)))
    print("")
    print("Principal Component Matrix:\n{0}".format(
                                    model.steps[1][1].components_.round(3)))
#
    # Plot the PCA Variance to determine optimum components
    plt.plot(var)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Principal Component Variance')
    plt.savefig(output +
                '{}.png'.format('Principal Component Variance'))
    plt.close()
#
# Method: PredictModel
    # Inputs:
    #     model = A clustering model object
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
               'D212_Data_Mining_II\\' +
               'OFM2 Task 2 - Dimensionality Reduction Methods\\')
    outDir = (directory +
              'Output\\')
    dataset = (directory +
               'churn_clean.csv')
    keepFields = ['Contacts',
                  'Age',
                  'Yearly_equip_failure',
                  'Outage_sec_perweek',
                  'Email',
                  'Techie']
    target = 'Contacts'
#
    # Create a Pandas dataframe object from the .csv dataset
    Churn = pd.read_csv(dataset)
#
    # Clean the dataframe, summarize, and then standardize the data
    Churn_clean = CleanDF(Churn,
                          outDir,
                          keepFields)
    SummarizeData(Churn_clean,
                  outDir)
#
    # Set the feature and target variables
    X = Churn_clean.loc[:,Churn_clean.columns != target]
    y = Churn_clean[target]
#
    # Create a pipeline object and get model parameters
    pipeline = CreatePipeline()
#
    # Split the dataframe into training and testing datasets
    X_train, X_test, y_train, y_test = SplitDataset(X, y, outDir)
#
    # Build the clustering model
    cm = BuildCluster(pipeline,
                      X_train,
                      y_train)
#
    # Predict the target variable using the clustering model
    y_pred = PredictModel(cm,
                          X_test,
                          y_test,
                          outDir,
                          'clustering')
#
    # Print the model metrics to the console
    ModelMetrics(cm,
                 X_test,
                 y_test,
                 y_pred,
                 outDir)
#