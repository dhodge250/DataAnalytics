# ---------------------------------------------------------------------
# -------------------Market Basket Analysis v.1.0.0--------------------
# Created by: David Hodge
# Created on: 2021-4-19
# Description: This script is used for performing a market basket
#            analysis on a given dataset using the Apriori algorithm
# Version:
#       1.0.0 - Initial script created 
#
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------

# Import libraries
import pandas as pd
#
from mlxtend.frequent_patterns import association_rules, apriori
#
# Method: BuildApriori
    # Inputs:
    #     df = A Pandas dataframe object
    # Processing:
    #     The BuildApriori method creates the association rules for
    #     the dataframe using the Apriori algorithm
    # Output:
    #     Returns the frequent itemsets and the association rules to
    #     the calling object
#
def BuildApriori(df):
#
    # Create frequent itemsets using Apriori algorithm
    print("Creating frequent itemsets...")
    fI = apriori(df,
                 use_colnames=True,
                 min_support=0.001)
#
    # Calculate the association rules
    print("Calculating the association rules...")
    aR = association_rules(fI,
                           metric = "support",
                           min_threshold = 0.0)
    return aR
#
# Method: CleanDF
    # Inputs:
    #     df = A Pandas dataframe object
    #     output = Folder directory as string object
    # Processing:
    #     The CleanDF method cleans unnecessary columns from the
    #     dataframe, and creates dummy variables for the remaining
    #     columns.
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
                    'Churn_Clean.csv')
    return df_clean
#
# Method: PrintMetrics
    # Inputs:
    #     rules = A pandas dataframe object of association rules
    #     output = Folder directory as string object
    # Processing:
    #     The PrintMetrics method prints the metrics from an
    #     Apriori algorithm
    # Outputs:
    #     None
#
def PrintMetrics(rules, output):
#
    # Determine top 3 rules for each measure (support, confidence, and
    # lift)
    # support = rules.sort_values('support', ascending=False).head(3)
    # confidence = rules.sort_values('confidence', ascending=False).head(3)
    lift = rules.sort_values('lift', ascending=False).head(3)
#
    # Print metrics for each measure
    # print("Support:\n{0}".format(support))
    # print("")
    # print("Confidence:\n{0}".format(confidence))
    print("")
    print("Lift:\n{0}".format(lift))
    
    # Convert rules and methods to CSV
    rules.to_csv(output +
                 'AssociationRules.csv')
    # support.to_csv(output +
    #                'SupportOutput.csv')
    # confidence.to_csv(output +
    #                   'ConfidenceOutput.csv')
    lift.to_csv(output +
                'LiftOutput.csv')
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
               'OFM2 Task 3 - Association Rules and Lift Analysis\\')
    outDir = (directory +
              'Output\\')
    dataset = (directory +
               'churn_clean.csv')
    keepFields = ['Phone',
                  'InternetService',
                  'Multiple',
                  'OnlineSecurity',
                  'OnlineBackup',
                  'DeviceProtection',
                  'TechSupport',
                  'StreamingTV',
                  'StreamingMovies']
#
    # Create a Pandas dataframe object from the .csv dataset
    Churn = pd.read_csv(dataset)
#
    # Clean the dataframe
    Churn_clean = CleanDF(Churn,
                          outDir,
                          keepFields)
#
    # Create the association rules using Apriori
    aR = BuildApriori(Churn_clean)
#
    # Print the metrics of the generated rules
    PrintMetrics(aR,
                 outDir)
#
    