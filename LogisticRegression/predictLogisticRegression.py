# ---------------------------------------------------------------------
# ---------------------Predictive Modeler v.2.0.0----------------------
# Created by: David Hodge
# Created on: 2020-1-09
# Description: This script is used for creating an optimized predictive
#            model from an inputted dataset.
# Version:
#       2.0.0 - Updated program to create Logistic Regression model
#
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
#
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFECV, RFE
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
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
    #     The BuildLM creates a fitted logistic regression model.
    # Output:
    #     Returns the fitted logistic regression model as a statsmodels
    #     object.
#
def BuildLM(y, X, linear = 'yes'):
#
    # Create fitted linear regression model
    if linear == 'yes':
        lm = sm.OLS(y,X).fit()
        print(lm.summary())
#
    # Create fitted logistic regression model
    elif linear == 'no':
        lm = LogisticRegression(max_iter=5000).fit(X,y)
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
# Method: MergeDF
    # Inputs:
    #     y = Matrix of target variable
    #     X = Matrix of predictor variables
    # Processing:
    #     The MergeDF merges the X and y matricies into a single
    #     dataframe.
    # Output:
    #     Returns the merged dataframe object
#
def MergeDF(X, y):
    x_df = pd.DataFrame(data=X)
    y_df = pd.DataFrame(data=y)
    df = pd.merge(x_df,
                  y_df,
                  left_on = 'index1',
                  right_on = y_df.index)
    return df
#
# Method: PredictModel
    # Inputs:
    #     model = A logistic regression model object
    #     X_test = Matrix of predictor variables from a test dataframe
    #         object
    #     y_test = Matrix of target variable from a test dataframe
    #         object
    #     X_select = List of target variables selected to use in an
    #         optimized logistic regression model
    #     output = Folder directory as string object
    # Processing:
    #     The PredictModel method predicts the target variable for
    #     the test dataframe, and creates a plot of the residual error
    #     for the predictions.
    # Output:
    #     None
#
def PredictModel(model, X, y, output, name, X_select = None,
                 logit = 'no'):
#
    # if X_select is not None:
    #     # Remove any column from the X matrix that is not in the
    #     # list of selected predictor variables
    #     for item in X:
    #         if item not in X_select:
    #             del X[item]
#
    # Predict the target variable, and save the predictions to a .csv
    # file
    y_pred = model.predict(X)
    df = pd.DataFrame({'Actual': y,
                       'Predicted': y_pred})
    df.to_csv(output +
              '{}_PredictResults.csv'.format(name))
#
    if logit == 'yes':
        print(classification_report(y,
                                    y_pred))
        print(confusion_matrix(y,
                               y_pred))
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
    #     selection to select the best variables for a logistic
    #     regression model
    # Output:
    #     A matrix of reduced predictor variables is returned
#
def RecursiveFeatureSelection(X, y, linear = 'yes'):
#
    if linear == 'yes':
        regres = LinearRegression()
    elif linear == 'no':
        regres = LogisticRegression(solver='lbfgs',
                                    max_iter=5000)
#
    # Use recursive feature selection to select the best predictor
    # variables to build a logistic regression model
    rfs = RFE(regres,
              10)
    rfs.fit(X,
            y)
#
    # Create a new matrix of target variables built from the selected
    # variables, and add a constant variable to the matrix
    # X_RFE = rfs.transform(X.values)
    # XCopy[XCopy.columns] = X_RFE
    # print(X_RFE)
    return X[X.columns[rfs.get_support(indices = True)]]
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
def StandardizeData(output = None,
                    df = None,
                    log = 'no',
                    X_train = None,
                    X_test = None):
#
    if log == 'yes':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test
#
    else:
        # Create a scaler object for the dataframe
        df['index1'] = df.index
        dfCopy = df.copy()
        columns = df.loc[:,df.columns != 'index1']
        scaler = StandardScaler()
    #
        # Scale the data in the dataframe using the scaler object, and
        # create a new dataframe object with the standardized variables
        df_scaled = scaler.fit_transform(columns.values)
        dfCopy[columns.columns] = df_scaled
        return dfCopy
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
               'D208_Predictive_Modeling\\' +
               'NBM2 Task 2 - Logistic Regression for Predictive Modeling\\')
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
                    'Job']
#
    # Create a Pandas dataframe object from the .csv dataset
    Churn = pd.read_csv(dataset)
#
    # Clean the dataframe, and then summarize the data
    Churn_clean = CleanDF(df = Churn, 
                          fields = removeFields,
                          output = outDir,
                          dummy = 'Yes')
    SummarizeData(Churn_clean,
                  outDir)
#
    # Set the predictor and target variables
    yString = 'Churn_Yes'
    X = Churn_clean.loc[:,Churn_clean.columns != yString]
    y = Churn_clean[yString]
#
    # Create visualizations of the variables in the dataframe
    SNSPairPlot(df = Churn_clean,
                output = outDir)
    UnivariateGraph(df = Churn_clean,
                    output = outDir)
    BivariateGraph(df = Churn_clean,
                    target = yString,
                    output = outDir)
    CorrelationMatrix(df = Churn_clean,
                      X = X,
                      output = outDir)
#
    # Split the dataframe into training and testing datasets, and
    # standardize the data, and 
    X_train, X_test, y_train, y_test = SplitDataset(X, y)
#    
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    
    X_train_scl = StandardizeData(df = X_train_df,
                                  output = outDir)
    X_test_scl = StandardizeData(df = X_test_df,
                                 output = outDir)
#
    # Merge the X and y datasets back into single dataframes
    train_df = MergeDF(X_train_scl,
                        y_train)
    test_df = MergeDF(X_test_scl,
                      y_test)
#
    # Set the X and y variables
    Xtrain = train_df.loc[:,train_df.columns != yString]
    Xtrain = Xtrain.loc[:,Xtrain.columns != 'index1']
    ytrain = train_df[yString]
#
    Xtest = test_df.loc[:,test_df.columns != yString]
    Xtest = Xtest.loc[:,Xtest.columns != 'index1']
    ytest = test_df[yString]
#    
    # Create an initial logistic regression model with all predictor
    # variables and print the variable coefficients
    lm1 = BuildLM(y = ytrain,
                  X = Xtrain,
                  linear = 'no')
    coeff = pd.concat([pd.DataFrame(train_df.columns),
                       pd.DataFrame(np.transpose(lm1.coef_))],
                      axis = 1)
    print(coeff)
#
    # Predict the target variable using the initial regression model
    PredictModel(model = lm1,
                  X = Xtrain,
                  y = ytrain,
                  output = outDir,
                  name = 'initial',
                  logit = 'yes')
#
    # Use recursive feature selection to reduce the initial predictor
    # variables to the minimum number required for an optimized logistic
    # regression model.
    X_select = RecursiveFeatureSelection(X = Xtrain,
                                         y = ytrain,
                                         linear = 'no')
    X_select_test = RecursiveFeatureSelection(X = Xtest,
                                              y = ytest,
                                              linear = 'no')
#
    # Create an optimized logistic regression model using the reduced
    # predictor variables, print the coefficients, and then predict
    # the target variable using the optimized regression model
    lm2 = BuildLM(y = ytrain,
                  X = X_select,
                  linear = 'no')
    coeff = pd.concat([pd.DataFrame(X_select.columns),
                       pd.DataFrame(np.transpose(lm2.coef_))],
                      axis = 1)
    print(coeff)
#    
    PredictModel(model = lm2,
                 X = X_select_test,
                 y = y_test,
                 output = outDir,
                 name = 'optimized',
                 logit = 'yes')