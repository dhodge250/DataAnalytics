# ---------------------------------------------------------------------
# --------------------Time Series Modeling v.1.0.0---------------------
# Created by: David Hodge
# Created on: 2020-5-9
# Description: This script is used for performing a Time Series
#             Analysis on a given dataset
# Version:
#       1.0.0 - Initial script created 
#
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------
#
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
#
from math import sqrt
from pmdarima.arima import auto_arima
from scipy.signal import welch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
#
# Method: BuildARIMA
    # Inputs:
    #     df = A Pandas dataframe object
    # Processing:
    #     The BuildARIMA method fits the ARIMA model to the dataset
    #     using the determined parameters
    # Output:
    #     Returns the ARIMA model
#
def BuildARIMA(df):
#
    # Create the ARIMA model using the optimized parameters
    print("Fitting model to dataset...")
    model = auto_arima(df.Revenue,
                       seasonal = True,
                       d = None,
                       m = 1)
    return model
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
def CleanDF(df, idx, output, title):
#
    # Create a Pandas dataframe object from the .csv dataset
    df = pd.read_csv(df)
#
    # Create a copy of the dataframe to be used for cleaning
    print("Cleaning the dataset...")
    df_clean = df.copy()
#
    # Set the index of the dataframe, and convert it to datetime
    df_clean[str(idx)] = pd.to_datetime(df_clean[str(idx)])
    df_clean = df_clean.set_index(idx)
    df_clean.index = pd.DatetimeIndex(df_clean.index.values,
                                      freq=df_clean.index.inferred_freq)
#
    # Save the dataframe to the output directory
    df_clean.to_csv(output +
                    title +
                    '.csv')
    # Plot the dataframe
    plt.figure(figsize=(10,8))
    plt.plot(df_clean)
    plt.xlabel('Date')
    plt.ylabel('Revenue (in Millions)')
    plt.savefig(output +
                'Time Series Graph.png')
    plt.close()
#
    return df_clean   
#
# Method: DataStationary
    # Inputs:
    #     df = A Pandas dataframe object
    #     decomp = A seasonal_decomposition object
    #     output = Folder directory as string object
    # Processing:
    #     The DataStationary removes the trend from the dataset and
    #     makes the dataset stationary
    # Output:
    #     det = A stationary Pandas dataframe object
#
def DataStationary(df, decomp, output):
#
    # Detrend the dataset
    print('Detrending the dataset...')
    det = df.diff()
    det =det.dropna(how = 'any')
#
    # Plot the cleaned dataset
    plt.figure(figsize=(10,8))
    plt.plot(det)
    plt.title('Telco Dataset with the Trend Removed')
    plt.savefig(output +
                'Detrended Dataset.png')
    plt.close()
#
    return det
#
# Method: ExportSummary
    # Inputs:
    #     obj = The object to write to the text file
    #     output = Folder directory as string object
    #     title = A string object for the title of the file
    # Processing:
    #     The ExportSummary writes the summary of an object to a
    #     text file.
    # Output:
    #     None
#
def ExportSummary(obj, output, title):
#
    # Writes the summary to a text file
    print('Writing summary to text file...')
    file = open(output +
                title +
                '.txt',
                'w')
    file.write(str(obj.summary()))
#
# Method: ForecastModel
    # Inputs:
    #     model = A fitted ARIMA model object
    # Processing:
    #     The ForecastModel forecasts the dataset using the ARIMA model
    # Output:
    #     forc = A predicted model forecast object
#
def ForecastModel(df, model, days, output):
#
    # Forecasting the model into the future
    print('Forecasting the model by {0} days...'.format(days))
    forc, conint = model.predict(n_periods = days,
                                 return_conf_int = True)
    forc_idx = pd.date_range(df.index[df.shape[0]-1],
                             periods = days,
                             freq = 'D')
    forc_series = pd.Series(forc,
                            index = forc_idx)
    lwr_series = pd.Series(conint[:, 0],
                           index = forc_idx)
    upr_series = pd.Series(conint[:, 1],
                           index = forc_idx)
#
    # Plot the predictions
    plt.figure(figsize=(10,8))
    plt.plot(df)
    plt.plot(forc_series)
    plt.xlabel('Date')
    plt.ylabel('Revenue (in Millions)')
    plt.fill_between(lwr_series.index,
                     lwr_series,
                     upr_series,
                     color = 'k',
                     alpha = 0.25)
    plt.legend(('Actual Revenue',
                'Forecasted Revenue',
                '95% Confidence Intervel'),
               loc='upper left')
    plt.savefig(output +
                '{} Day Prediction.png'.format(days))
    plt.close()
#
    return forc
#
# Method: PlotCorrelation
    # Inputs:
    #     df = A pandas dataframe object
    #     output = Folder directory as string object 
    # Processing:
    #     The PlotCorrelation plots the autocorrelation of the
    #     dataframe
    # Output:
    #     None
#
def PlotCorrelation(df, output, title):
#
    # Plot the autocorrelation of the dataframe
    print('Plotting the autocorrelation of the dataset...')
    plt.rcParams.update({'figure.figsize':(10,8),
                         'figure.dpi':120})
    plot_acf(df.Revenue)
    plt.title(title)
    plt.savefig(output +
                '{}.png'.format(title))
    plt.close()
#
# Method: PlotDecomp
    # Inputs:
    #     df = A pandas dataframe object
    #     output = Folder directory as string object 
    # Processing:
    #     The PlotDecomp plots the autocorrelation of the
    #     dataframe
    # Output:
    #     None
#
def PlotDecomp(df, output, title):
#
    # Plot the decomposition of the dataframe
    print('Plotting the decomposition of the dataset...')
    plt.rcParams.update({'figure.figsize':(10,8),
                         'figure.dpi':120})
    dec = STL(df,
              robust = True).fit()
#
    # Plot the decomposition
    dec.plot()
    plt.savefig(output +
                '{}.png'.format(title))
    plt.close()
#
    return dec
#
# Method: PlotDensity
    # Inputs:
    #     df = A pandas dataframe object
    #     output = Folder directory as string object 
    # Processing:
    #     The PlotDensity plots the autocorrelation of the
    #     dataframe
    # Output:
    #     None
#
def PlotDensity(df, output, title):
#
    # Plot the decomposition of the dataframe
    print('Plotting the spectral density of the dataset...')
    plt.rcParams.update({'figure.figsize':(10,8),
                         'figure.dpi':120})
    f, den = welch(df.Revenue)
#
    # Plot the decomposition
    plt.semilogy(f, den)
    plt.savefig(output +
                '{}.png'.format(title))
    plt.close()
#
# Method: PredictAccuracy
    # Inputs:
    #     pred = A predicted model forecast object
    #     testDF = A Pandas dataframe object
    # Processing:
    #     The PredictAccuracy calculates the accuracy of the test model
    # Output:
    #     None
#
def PredictAccuracy(pred, testDF, model, output):
#
    # Determine model accuracy metrics
    print('Determining model accuracy...')
    print('')
    print('Initial Model Metrics:')
    print('Mean Absolute Error: {:.3f}'.format(mean_absolute_error(testDF,
                                                                   pred)))
    print('Mean Absolute Percent Error: {:.3f}'.format(
                                mean_absolute_percentage_error(testDF,
                                                               pred)*100))
    print('Mean Squared Error: {:.3f}'.format(mean_squared_error(testDF,
                                                                 pred)))
    print('Root Mean Squared Error: {:.3f}'.format(sqrt(mean_squared_error(
                                                                       testDF,
                                                                       pred))))
    print('')
#
    # Plot the models residuals
    residu = pd.DataFrame(model.resid())
    residu.plot()
    plt.savefig(output +
                'Test Model Residuals.png')
    plt.close()
#
    # Determine residual distribution
    residu.plot(kind='kde')
    plt.savefig(output +
                'Test Model Distribution.png')
    plt.close()
#
# Method: PredictModel
    # Inputs:
    #     model = A fitted ARIMA model object
    #     df = A pandas dataframe object
    #     start = Int object representing length of dataset
    #     test = Int object representing length of test dataset
    #     output = Folder directory as string object 
    # Processing:
    #     The PredictModel plots the predictions of the test dataset
    #     against the training dataset
    # Output:
    #     pred = A Pandas dataframe object
#
def PredictModel(model, trainDF, testDF, days, output):
#
    # Run prediction against the test dataset
    print('Running prediction against the test dataset...')
    pred = pd.DataFrame(model.predict(n_periods = days),
                        index=testDF.index)
    pred.columns = ['Predicted_Revenue']
#
    # Plot the predictions
    plt.figure(figsize=(10,8))
    plt.plot(trainDF,
             label='Training')
    plt.plot(testDF,
             label='Testing')
    plt.plot(pred,
             label='Predicted')
    plt.legend(loc='upper left')
    plt.savefig(output +
                '{} {}.png'.format(days,
                                   'Day Prediction'))
    plt.close()
#
    return pred
#
# Method: SplitDataset
    # Inputs:
    #     df = df = A Pandas dataframe object
    #     output = A string object used to specify the directory to
    #        save the dataframe
    # Processing:
    #     The SplitDataset method splits the input dataframe object
    #     into seperate training and testing dataframes
    # Output:
    #     trainDF = A pandas dataframe object
    #     testDF = A pandas dataframe object
#
def SplitDataset(df, split, output):
#
    # Split the dataframe into training and testing dataframes
    print("Splitting the dataframe into training and testing datasets...")
    trainDf = df.iloc[:split,:]
    testDf = df.iloc[split:,:]
    trainDf.to_csv(output +
                   'Revenue_train.csv')
    testDf.to_csv(output +
                   'Revenue_test.csv')
    return trainDf, testDf
#
# Method: TestStationary
    # Inputs:
    #     df = A pandas dataframe object
    # Processing:
    #     The TestStationary tests the stationary of the dataframe
    # Output:
    #     None
#
def TestStationary(df):
#
    # Test if the dataset is stationary
    print('Testing if the dataset is stationary...')
    stat = adfuller(df.Revenue.dropna())
#
    print('')
    print('Dataset Metrics:')
    print('ADF Statistic: %.3f' % stat[0])
    print('p-value: %.3f' % stat[1])
    print('')
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
               'D213_Advanced_Data_Analytics\\' +
               'NLM2 Task 1 - Time Series Modeling\\')
    outDir = (directory +
              'Output\\')
    dataset = (directory +
               'teleco_time_series.csv')
    index = 'Day'
#
    # Clean the dataframe and set the index
    telcoCln = CleanDF(dataset,
                       index,
                       outDir,
                       'Telco_Clean')
#
    # Determine if the dataset is stationary
    TestStationary(telcoCln)
#
    # Plot the decomposition of the dataframe
    dec = PlotDecomp(telcoCln,
                     outDir,
                     'Decomposition of Dataset')
#
    # Plot the autocorrelation of the dataframe to test for
    # seasonality
    PlotCorrelation(telcoCln,
                    outDir,
                    'Autocorrelation of Dataset')
#
    # Plot the spectral density of the dataset
    PlotDensity(telcoCln,
                outDir,
                'Spectral Density')
#
    # Make the dataframe stationary
    telcoStat = DataStationary(telcoCln,
                               dec,
                               outDir)
    TestStationary(telcoStat)
#
    # Split the dataset into training and testing datasets
    trainDF, testDF = SplitDataset(telcoCln,
                                   540,
                                   outDir)
#
    # Fit the ARIMA model to the training dataset
    model = BuildARIMA(trainDF)
#
    # Write the model summary to a text file
    ExportSummary(model,
                  outDir,
                  'Training_Model_Summary')
#
    # Verify the model against the testing dataset
    pred = PredictModel(model,
                        trainDF,
                        testDF,
                        len(testDF),
                        outDir)
#
    # Calculate the ARIMA model accuracy
    PredictAccuracy(pred,
                    testDF,
                    model,
                    outDir)
#
    # Fit the ARIMA model to the full dataset
    model = BuildARIMA(telcoCln)
#
    # Write the model summary to a text file
    ExportSummary(model,
                  outDir,
                  'Telco_Model_Summary')
#
    # Forecast the dataset using the ARIMA model
    forc = ForecastModel(telcoCln,
                         model,
                         365,
                         outDir)
# 