# ---------------------------------------------------------------------
# -------------------Hierarchical Clustering v.1.0.0-------------------
# Created by: David Hodge
# Created on: 2020-4-3
# Description: This script is used for performing Hierarchical
#             clustering on a given dataset
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
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import v_measure_score
from sklearn.preprocessing import StandardScaler
#
# Method: BuildCAM
    # Inputs:
    #     df = A pandas dataframe object
    #     itr = A string object used to label iteration
    # Processing:
    #     The BuildCluster method builds the hierarchical clusters
    # Output:
    #     Returns the predicted hierarchical clusters
#
def BuildCluster(df, clusters, itr):
#
    # Build the hierarchical cluster model
    print("Creating hiearchical clusters for {0}...".format(itr))
    v_scores = []
    predList = []
    for cl in clusters:
        cluster = AgglomerativeClustering(n_clusters = cl,
                                          affinity = 'euclidean',
                                          linkage = 'ward')
        pred = cluster.fit_predict(df)
        predList.append(pred)
        data = pd.DataFrame(pred)
        v_scores.append(v_measure_score(df.iloc[:, 0].values,
                                        data.iloc[:, 0].values))
    return predList, v_scores
#
# Method: CreateDendro
    # Inputs:
    #     df = A Pandas dataframe object
    #     itr = A string object used to label the iteration
    #     output = A string object used to specify the directory to
    #        save the plot
    # Processing:
    #     The CreateDendro method creates a dendrogram from a dataset.
    # Output:
    #     None
#
def CreateDendro(df, itr, output):
#
    # Create a plot to visualize dendrogram
    print("Creating dendrogram for {0}...".format(itr))
    # plt.figure(figsize=(10,7))
    plt.title('Dendrogram using Ward ({0})'.format(itr))
    plt.xlabel('Customers')
    plt.ylabel('Euclidean distances')
    link = linkage(df,
                   'ward')
    dendrogram(link)
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(output + 
                '{0}_{1}.png'.format('Dendrogram',
                                     itr))
#
# Method: NormalizeData
    # Inputs:
    #     df = A Pandas dataframe object
    #     col1 = A string object to label the new dataframe column
    #     col2 = A string object to label the new dataframe column
    #     output = A string object used to specify the directory to
    #        save the dataframe
    # Processing:
    #     The NormalizeData method normalizes a dataframe object
    # Output:
    #     Returns a normalized dataframe object
#
def NormalizeData(df,
                  col1,
                  col2,
                  output):
#
    # Scale the dataset
    print("Scalling the dataset...")
    scale = StandardScaler()
    scaled = scale.fit_transform(df)
#
    # Save the dataframe to the output directory
    dfScaled = pd.DataFrame(scaled)
    dfScaled.rename(columns = {0:col1,
                               1:col2},
                    inplace = True)
    dfScaled.to_csv(output +
                    'Churn_Normalized.csv')
    return dfScaled
#
# Method: PlotClusters
    # Inputs:
    #     df = A Pandas dataframe object
    #     clusters = A sklearn cluster object
    #     itr = A string object used to label iteration
    #     output = The output directory to save files
    # Processing:
    #     The PlotClusters method creates a scatterplot of the
    #     calculated hierarchical clusters
    # Output:
    #     None
#
def PlotClusters(target,
                 clusters,
                 itr,
                 N_clusters,
                 v_scores,
                 output):
#
    # Create plot of calculated clusters
    print("Plotting clusters for {0}...".format(itr))
    cluster = 2
    for cl in clusters:
        plt.scatter(target.iloc[:,0],
                    target.iloc[:,1],
                    c=cl,
                    cmap='rainbow')
        plt.title('Income with {0} Clusters ({1})'.format(str(cluster),
                                                          itr))
        plt.xlabel('Annual Income')
        plt.ylabel('Monthly Charge')
        fig1 = plt.gcf()
        plt.show()
        fig1.savefig(output + 
                    '{0}_{1}_{2}.png'.format(str(cluster),
                                             'Cluster_Plot',
                                             itr))
        cluster = (cluster + 1)
#        
    # Create a bar chart to compare the scores of the different models
    plt.bar(N_clusters,
            v_scores)
    plt.xlabel('Number of Clusters')
    plt.ylabel('V-Measure Score')
    plt.title('Model Score')
    fig2 = plt.gcf()
    plt.show()
    fig2.savefig(output + 
                '{0}_{1}.png'.format('Cluster_Score',
                                     itr))
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
def SplitDataset(df, output):
#
    # Split the dataframe into training and testing dataframes
    print("Splitting the dataframe into training and testing datasets...")
    trainDf = df.iloc[:7000,:]
    testDf = df.iloc[7001:,:]
    trainDf.to_csv(output +
                   'Income_train.csv')
    testDf.to_csv(output +
                   'Income_test.csv')
    return trainDf, testDf
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
               'OFM2 Task 1 - Clustering Techniques\\')
    outDir = (directory +
              'Output\\')
    dataset = (directory +
               'churn_clean.csv')
    col1 = 'Income'
    col2 = 'Bandwidth_GDB_Year'
    itr = 0
    N_clusters = [2,3,6]
    v_scores = []

#
    # Create a Pandas dataframe object from the .csv dataset
    Churn = pd.read_csv(dataset)
    target = Churn.iloc[:, [16, 41]].values
#
    # Normalize the dataframe
    Churn_scaled = NormalizeData(target,
                                 col1,
                                 col2,
                                 outDir)
#
    # Split the dataframe into training and testing datasets
    trainDf, testDf = SplitDataset(Churn_scaled,
                                   outDir)
    dfList = [trainDf,
              testDf]
#
    # Create clusters for the training and testing datasets
    for df in dfList:
        if itr == 0:
            dfType = 'Train'
        else:
            dfType = 'Test'
        # Create dendrogram from training dataset
        CreateDendro(df,
                     dfType,
                     outDir)
#
        # Build the hierarchical clusters
        predList, v_scores = BuildCluster(df,
                                          N_clusters,
                                          dfType)
#
        # Plot the hierarchical clusters
        PlotClusters(df,
                     predList,
                     dfType,
                     N_clusters,
                     v_scores,
                     outDir)
#
        # Increase iteration
        itr = (itr + 1)
#