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

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import seaborn as sns
from sklearn import preprocessing

from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
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
    df_sample = df.sample(n=100)
#
    # For each column in the dataframe, create a bivariate scatterplot
    for column in df_sample:
        if df[column].dtypes != 'object':
            sns.scatterplot(data = df_sample,
                            x = df_sample[column],
                            y = df_sample[target])
            plt.title('Bivariate Relationship Between {0} \n and {1}'.format(\
                str(column),
                target))
            plt.savefig(output +
            '{}_Churn.png'.format(str(column)))
            plt.close()    
#
# Method: BuildLM
    # Inputs:
    #     y = Matrix of target variable
    #     X = Matrix of predictor variables
    # Processing:
    #     The BuildLM creates a fitted linear regression model.
    # Output:
    #     Returns the fitted linear regression model as a statsmodels
    #     object.
#
def BuildLM(y,X):
#
    # Create fitted linear regression model
    lm = sm.OLS(y,X).fit()
    print(lm.summary())
    return lm
#
# Method: CleanDF
    # Inputs:
    #     df = A Pandas dataframe object
    #     fields = A list of columns to be "manually" removed from the
    #         dataframe
    #     output = Folder directory as string object
    # Processing:
    #     The csvWrite method writes the object values to a CSV file.
    # Output:
    #     Returns a new "cleaned" dataframe object
#
def CleanDF(df, fields, output, dummy = 'Yes'):
#
    # Create a copy of the dataframe to be used for cleaning
    df_clean = df.copy()
#
    # Delete any column from the dataframe that is in the list of
    # fields to be removed
    for column_name in df_clean.columns:
        if column_name in fields:
            del df_clean[column_name]
#
#   Dependent on the model type selected, remove unnecessary columns
    if dummy == 'No':
        # Delete any column from the dataframe that is an 'object' type
        for column in df_clean.columns:
            if df_clean[column].dtypes == 'object':
                del df_clean[column]
#
    elif dummy == 'Yes':
        # Create a dummy variable for any non-numeric column in the
        # dataframe
        df_clean = pd.get_dummies(df_clean,
                                  drop_first = True)
#
    # Fill any NaN records
    print("Filling in missing values...")
    df_clean = df_clean.interpolate()
#
    # Add constant variable to the dataframe, and save the dataframe
    # to the output directory
    df_clean = sm.add_constant(df_clean)
    df_clean.to_csv(output +
                    'Churn_dummy.csv')
    return df_clean
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
    # Create a correlation matrix, and save to a .csv file
    X.corr().to_csv(output + 
                    'CorrelationMatrix.csv')
#
    # Create a heatmap of the correlation matrix
    plt.figure(figsize=(47,47))
    heatmap = sns.heatmap(X.corr(),
                          vmin=-1,
                          vmax=1,
                          annot=True,
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
# Method: PredictModel
    # Inputs:
    #     model = A linear regression model object
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
def PredictModel(model, X_test, y_test, output, name, X_select = None):
#
    if X_select is not None:
        # Remove any column from the X_test matrix that is not in the
        # list of selected predictor variables
        for item in X_test:
            if item not in X_select:
                del X_test[item]
#
    # Predict the target variable, and save the predictions to a .csv
    # file
    y_pred = model.predict(X_test)
    df = pd.DataFrame({'Actual': y_test,
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
    plt.ylabel('Standardized Residuals')
    plt.savefig(output +
                '{}_Residual_Plot.png'.format(name))
    plt.close()
#
# Method: RecursiveFeatureSelection
    # Inputs:
    #     X_train = Matrix of predictor variables from a training
    #         dataframe object
    #     y_train = Matrix of target variable from a training dataframe
    #         object
    # Processing:
    #     The RecursiveFeatureSelection method uses recursive feature
    #     selection to select the best variables for a linear
    #     regression model
    # Output:
    #     A matrix of reduced predictor variables is returned
#
def RecursiveFeatureSelection(X_train, y_train, linear = 'yes'):
#
    if linear == 'yes':
        regres = LinearRegression()
    elif linear == 'no':
        regres = LogisticRegression()
#
    # Use recursive feature selection to select the best predictor
    # variables to build a linear regression model
    rfs = RFECV(estimator=regres)
    rfs.fit(X_train,
            y_train)
#
    # Create a new matrix of target variables built from the selected
    # variables, and add a constant variable to the matrix
    X_RFE = X_train[X_train.columns[rfs.support_]]
    X_RFE = sm.add_constant(X_RFE)
    return X_RFE
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
def SplitDataset(X,y):
#
    # Split the dataframe into training and testing dataframes
    X_train, X_test, y_train, y_test = train_test_split(\
        X,
        y,
        test_size=0.3,
        random_state=101)
    return X_train, X_test, y_train, y_test
#
# Method: StandardizeData
    # Inputs:
    #     df = A Pandas dataframe object
    # Processing:
    #     The StandardizeData method standardizes the data in the 
    #     dataframe object so all variables have a mean of 0 and a
    #     standard deviation of 1
    # Output:
    #     Returns a standardized version of the dataframe object
#
def StandardizeData(df):
#
    # Create a scaler object for the dataframe
    names = df.columns
    scaler = preprocessing.StandardScaler()
#
    # Scale the data in the dataframe using the scaler object, and
    # create a new dataframe object with the standardized variables
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled,
                             columns=names)
    return df_scaled
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
    Churn_clean.describe().to_csv(output +
                                  'Churn_summary.csv')
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
    for column_name in df:
        if df[column_name].dtypes != 'object':
            plt.figure(figsize=(5,5))
            df[column_name].plot(kind='hist')
            plt.title('Univariate Distribution of {}'.format(\
                str(column_name)))
            plt.xlabel('{}'.format(str(column_name)))
            plt.ylabel('Density')
            plt.savefig(output +
                        '{}.png'.format(str(column_name)))
            plt.close()
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
              'Output2\\')
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
#
    # Create a Pandas dataframe object from the .csv dataset
    Churn = pd.read_csv(dataset)
#
    # Clean the dataframe, summarize, and then standardize the data
    Churn_clean = CleanDF(Churn, 
                          removeFields,
                          outDir,
                          'Yes')
    SummarizeData(Churn_clean,
                  outDir)
    Churn_std = StandardizeData(Churn_clean)
#
    # Set the predictor and target variables
    X = Churn_std.loc[:,Churn_std.columns != 'Acreage']
    y = Churn_std['Acreage']
#
    # Create visualizations of the variables in the dataframe
    # SNSPairPlot(Churn_std,
    #             outDir)
    # UnivariateGraph(Churn_std,
    #                 outDir)
    # BivariateGraph(Churn_std,
    #                 'Tenure',
    #                 outDir)
    # CorrelationMatrix(Churn_std,
    #                   X,
    #                   outDir)
#
    # Split the dataframe into training and testing datasets, and
    # create an initial linear regression model with all predictor
    # variables
    X_train, X_test, y_train, y_test = SplitDataset(X,y)
    lm1 = BuildLM(y_train,
                  X_train)
#
    # Predict the target variable using the initial regression model
    PredictModel(lm1,
                 X_test,
                 y_test,
                 outDir,
                 'initial')
#
    # Use recursive feature selection to reduce the initial predictor
    # variables to the minimum number required for an optimized linear
    # regression model
    X_select = RecursiveFeatureSelection(X_train,
                                         y_train)
#
    # Create an optimized linear regression model using the reduced
    # predictor variables, and then predict the target variable using
    # the optimized regression model
    lm2 = BuildLM(y_train,
                  X_select)
#    
    PredictModel(lm2,
                 X_test,
                 y_test,
                 outDir,
                 'optimized',
                 X_select)