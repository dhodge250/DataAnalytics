# ---------------------------------------------------------------------
# ---------------------Predictive Modeler v.2.0.0----------------------
# Created by: David Hodge
# Created on: 2020-1-09
# Description: This script is used for creating an optimized predictive
#            model from an inputted dataset.
# Version:
#       2.0.0 - Updated program to 
#
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------
#
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os.path
import seaborn as sns
#
from joblib import dump, load
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#
# Method: BivariateGraph
    # Inputs:
    #     df = A Pandas dataframe object
    #     target = Target variable as string object
    #     output = Folder directory as string object
    # Processing:
    #     The BivariateGraph method creates bivariate scatterplots of
    #     each column in the dataframe object compared to the target
    #     variable, and saves each plot to a .png file.
    # Output:
    #     None
#
def BivariateGraph(df, target, output):
#
    # Take a random samlple of n recordes from the dataframe
    print('Creating bivariate graphs...')
    df_sample = df.sample(n=100)
#
    # For each column in the dataframe, create a bivariate scatterplot
    for column in df_sample:
        if df[column].dtypes != 'object':
            try:
                sns.scatterplot(data = df_sample,
                                x = df_sample[column],
                                y = df_sample[target])
                plt.title('Bivariate Relationship Between' +
                          '{0} \n and {1}'.format(str(column),
                                                  target))
                plt.savefig(output +
                '{}_Fire.png'.format(str(column)))
                plt.close() 
            except FileNotFoundError:
                print("    - Skipping {0}...".format(str(column)))
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
    #     output = Folder directory as string object
    # Processing:
    #     The BuildCAM method builds the classification analysis model
    # Output:
    #     Returns the optimized fitted model
#
def BuildModel(pipeline, parameters, X_train, y_train, output, name):
    # Use grid search to identify the hyperparameters to optimize the
    # classification analysis
    print("Using grid search to tune the hyperparameters...")
    pipeline = GridSearchCV(pipeline,
                            param_grid = parameters,
                            n_jobs = -1,
                            cv = 3,
                            scoring = 'r2')
#
    # Fit the training dataset to the pipeline object
    print("Fitting the model...")
    pipeline.fit(X_train,
                 y_train)
#
    # Save the model to disk for later use
    dump(pipeline,
         output + name,
         compress = 1)
    return pipeline
#
# Method: CleanDF
    # Inputs:
    #     df = A Pandas dataframe object
    #     output = Folder directory as string object
    #     fields = A list object that contains field names to remove
    #     title = A string object for the name of the csv file
    # Processing:
    #     The csvWrite method writes the object values to a CSV file.
    # Output:
    #     Returns a new "cleaned" dataframe object
#
def CleanDF(df, output, fields, title):
#
    # View original dataframe
    print("Dataframe before cleaning:\n{0}".format(df.dtypes))
    print("")
#
    # Create a copy of the dataframe to be used for cleaning
    print("Cleaning the dataset...")
#
    # Delete any column from the dataframe that is in the list of
    # fields to be removed
    for column_name in df.columns:
        if column_name in fields:
            del df[column_name]
#
    # Fill any NaN records
    print("Filling in missing values...")
    df_new = df.interpolate()
#
    # Convert categorical variables into numeric
    print("Converting categorical variables...")
    df_clean = pd.get_dummies(df_new,
                              drop_first = True)
#
    # Save the dataframe to the output directory
    df_clean.to_csv(output +
                    title +
                    '.csv')
#
    # View reduced dataframe
    print("Dataframe after cleaning:\n{0}".format(df_clean.dtypes))
    print("")
#
    # Summarize the variables in the dataframe
    df_new.describe().to_csv(output +
                               'Fire_summary.csv')
    return df_new, df_clean
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
             ('pca', PCA()),
             ('linear', LinearRegression())]
    pipeline = Pipeline(steps)
#
    parameters = {'pca__n_components': np.arange(1, 95, dtype = int)}
    return pipeline, parameters
#
# Method: CorrelationMatrix
    # Inputs:
    #     df = A Pandas dataframe object
    #     X = Matrix of predictor variables
    #     output = Folder directory as string object
    # Processing:
    #     The CorrelationMatrix creates a correlation matrix, and
    #     heatmap of the correlation matrix.
    # Output:
    #     None
#
def CorrelationMatrix(df, X, output):
#
    print("Creating correlation matrix...")
    # Create a correlation matrix, and save to a .csv file
    X.corr().to_csv(output + 
                    'CorrelationMatrix.csv')
#
    # Create a heatmap of the correlation matrix
    heatmap = sns.heatmap(X.corr(),
                          cmap='BrBG')
    heatmap.set_title('Correlation Heatmap',
                      fontdict={'fontsize':18},
                      pad=12)
    plt.savefig(output +
                'CorrelationMatrix.png',
                dpi=600,
                bbox_inches='tight')
    plt.close()
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
def ModelMetrics(model, X, y_true, y_pred, output, title):
#
    # Determine the metrics of the model, and print the results
    score = r2_score(y_true,
                     y_pred)
    var = model.best_estimator_.steps[1][1].explained_variance_ratio_.cumsum()
    tolvar = model.best_estimator_.steps[1][1].explained_variance_ratio_
#
    print("")
    print("{1} Model R2: {0}".format(score,
                                     title))
    print("{1} Number of Components: {0}".format(len(
                            model.best_estimator_.steps[1][1].components_),
                            title))
    print("{1} Explained Variance for each Principal Component:\n{0}".format(
                                                            tolvar.round(3),
                                                            title))
    print("{1} Total Variance for Principal Components: {0}".format(
                                                        sum(tolvar).round(2),
                                                        title))
    print("")
    print("{1} Principal Component Matrix:\n{0}".format(
                    model.best_estimator_.steps[1][1].components_.round(3),
                    title))
    print("")
#
    # Write the components to a text file
    print('Writing component matrix to csv file...')
    matrix = model.best_estimator_.steps[1][1].components_.round(3)
    np.savetxt(output +
               title +
               ' Principal Component Matrix.csv',
               matrix,
               delimiter = ',')
#
    # Plot the PCA Variance to determine optimum components
    plt.plot(var)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Principal Component Variance')
    plt.savefig(output +
                '{}.png'.format(title + ' Principal Component Variance'))
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
def PredictModel(model, X_true, y_true, output, name):
#
    # Predict the target variable, and save the predictions to a .csv
    # file
    print("Running predictions for {}...".format(name))
    y_pred = model.predict(X_true)
    df = pd.DataFrame({'Actual': y_true,
                       'Predicted': y_pred})
    df.to_csv(output +
              '{}_PredictResults.csv'.format(name))
#
    # Take a random samlple of n recordes from the dataframe, and
    # create a scatterplot of the residual error
    df_sample = df.sample(100)
    plt.figure(figsize=(8,8))
    plt.scatter(df_sample.Predicted,
                (df_sample.Actual - df_sample.Predicted))
    plt.title('Residual Error Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.savefig(output +
                '{}_Residual_Plot.png'.format(name))
    plt.close()
#
    return y_pred
#
# Method: SNSPairPlot
    # Inputs:
    #     df = A Pandas dataframe object
    #     output = Folder directory as string object
    # Processing:
    #     The SNSPairPlot method creates a pairplot of the data in
    #     the dataframe object
    # Output:
    #     None
#
def SNSPairPlot(df, output):
#
    print("Creating pairplot...")
    # Take a random samlple of n recordes from the dataframe, and
    # create a pairplot of the data
    df_sample = df.sample(n=100)
    sns.pairplot(df_sample)
    plt.savefig(output +
                'Pairplot.png')
    plt.close()
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
def SplitDataset(X, y, output):
#
    # Split the dataframe into training and testing dataframes
    print("Splitting the dataframe into training and testing datasets...")
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=21)
    train = [X_train, y_train]
    trainDF = pd.concat(train, axis=1, join="inner")
    trainDF.to_csv(output +
                   'Fire_train.csv')
    test = [X_test, y_test]
    testDF = pd.concat(test, axis=1, join="inner")
    testDF.to_csv(output +
                   'Fire_test.csv')
    return X_train, X_test, y_train, y_test
#
# Method: UnivariateGraph
    # Inputs:
    #     df = A Pandas dataframe object
    #     output = Folder directory as string object
    # Processing:
    #     The UnivariateGraph method creates a histogram for each
    #     variable in the dataframe to visually show the distribution
    #     each variable
    # Output:
    #     None
#
def UnivariateGraph(df, output):
#
    # Create a histogram for each column in the dataframe object
    print("Creating univariate graphs...")
    for column_name in df:
        if df[column_name].dtypes != 'object':
            try:
                plt.figure(figsize=(5,5))
                df[column_name].plot(kind='hist')
                plt.title('Univariate Distribution of {}'.format(\
                    str(column_name)))
                plt.xlabel('{}'.format(str(column_name)))
                plt.ylabel('Density')
                plt.savefig(output +
                            '{}.png'.format(str(column_name)))
                plt.close()
            except FileNotFoundError:
                print("    - Skipping {0}...".format(str(column_name)))
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
                 'D214_Data_Analytics_Graduate_Capstone\\' +
                 'NKM2 Task 2 - Data Analytics Report and Executive Summary\\')
    outDir = (directory +
              'Output\\')
    dataset = (directory +
               'US_Fire_Precip.csv')
    removeFields = ['OBJECTID',
                    'FPA_ID',
                    'NWCG_REPORTING_UNIT_ID',
                    'SOURCE_REPORTING_UNIT',
                    'SOURCE_REPORTING_UNIT_NAME',
                    'LOCAL_FIRE_REPORT_ID',
                    'STAT_CAUSE_CODE',
                    'OWNER_CODE']
    modelName = 'mlModel.pkl'
#

    # Create a Pandas dataframe object from the .csv dataset
    Fire = pd.read_csv(dataset,
                       low_memory = False)
#
    # Clean the dataframe, summarize, and then standardize the data
    Fire, Fire_clean = CleanDF(Fire, 
                               outDir,
                               removeFields,
                               'Fire_Precip_Clean')
#
    # Set the predictor and target variables
    X = Fire_clean.loc[:,Fire_clean.columns != 'Acreage']
    y = Fire_clean['Acreage']
#
    if not os.path.isfile(outDir + 'CorrelationMatrix.png'):
        # Create visualizations of the variables in the dataframe
        SNSPairPlot(Fire,
                    outDir)
        UnivariateGraph(Fire,
                        outDir)
        BivariateGraph(Fire,
                        'Acreage',
                        outDir)
        CorrelationMatrix(Fire,
                          X,
                          outDir)
#
    # Create a pipeline object and get model parameters
    pipeline, parameters = CreatePipeline()
#
    # Split the dataframe into training and testing datasets
    X_train, X_test, y_train, y_test = SplitDataset(X, y, outDir)
#
    if not os.path.isfile(outDir + modelName):
        # Build the regression model
        model = BuildModel(pipeline,
                           parameters,
                           X_train,
                           y_train,
                           outDir,
                           modelName)
    else:
        print('...model already exists')
        model = load(outDir + modelName)
#
    # Predict the target variable using the model
    y_pred_train = PredictModel(model,
                                X_train,
                                y_train,
                                outDir,
                                'Train')
    y_pred_test = PredictModel(model,
                               X_test,
                               y_test,
                               outDir,
                               'Test')
#
    # Calculate the model metrics
    ModelMetrics(model,
                 X_train,
                 y_train,
                 y_pred_train,
                 outDir,
                 'Train')
    ModelMetrics(model,
                 X_test,
                 y_test,
                 y_pred_test,
                 outDir,
                 'Test')