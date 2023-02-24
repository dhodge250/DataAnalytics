# ---------------------------------------------------------------------
# -----------------------Neural Networks v.1.0.0-----------------------
# Created by: David Hodge
# Created on: 2021-5-25
# Description: This script is used for building a neural network using
#            natural language processing (NLP) to predict the sentiment
#            of a review from an IMDB review dataset.
# Version:
#       1.0.0 - Initial script created 
#
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------

# Import libraries
import matplotlib.pyplot as plt
import pandas as pd
#
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Activation, Dense, Embedding, LSTM
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#
# Method: BuildModel
    # Inputs:
    #     wordnum = An int type object
    #     vectLen = An int type object
    #     maxLen = An int type object
    # Processing:
    #     The BuildModel method creates an LSTM model using the input
    #     arguments
    # Output:
    #     model = A TensorFlow model object
#
def BuildModel(wordnum, vectLen, maxLen):
#
    # Create the NLP model
    print("Building LSTM model...")
    model = Sequential()
    model.add(Embedding(wordnum,
                        vectLen,
                        input_length = maxLen))
    model.add(SpatialDropout1D(0.25))
    model.add(LSTM(10))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss = BinaryCrossentropy(),
                  optimizer = Adam(),
                  metrics = ['accuracy'])
#
    return model
#
# Method: CleanDF
    # Inputs:
    #     df = A string type object
    #     predict = A string type object
    #     char = A list type object
    #     output = Folder directory as string object
    # Processing:
    #     The CleanDF method converts the dataset into a Pandas
    #     dataframe object, and then removes any characters from the
    #     review column text.
    # Output:
    #     Returns a new "cleaned" dataframe object
#
def CleanDF(df, predict, char, output):
#
    # Create a Pandas dataframe object from the .csv dataset
    print('Creating Pandas dataframe...')
    df_new = pd.read_csv(df)
#
    # Clean the Pandas dataframe
    print('Removing characters from the review column...')
    # Remove characters from review
    for item in char:
        df_new[predict] = df_new[predict].str.replace(item,
                                                  '',
                                                  regex=False).astype('str')
#
    # Write the cleaned dataframe to a new .csv file
    df_new.to_csv(output +
                  'imdbClean.csv')
#
    # Plot the dataset in a histogram
    print('Plotting the dataset...')
    wordNum = [len(x.split()) for x in df_new[predict].tolist()]
    plt.hist(wordNum)
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.savefig(output +
                'Dataset Sequence.png')
    plt.close()
#
    return df_new
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
    with open(output +
              title +
              '.txt',
              'w') as file:
        obj.summary(print_fn=lambda x: file.write(x + '\n'))
#
# Method: PrepDF
    # Inputs:
    #     df = A Pandas dataframe object
    #     df = An int type object
    #     df = An int type object
    #     title = A string object for the title of the file
    #     output = Folder directory as string object
    # Processing:
    #     The PrepDF method tokenizes the dataframe, and converts the
    #     text values into numeric values.
    # Output:
    #     Returns the length of the tokens, and the padded sequence
#
def PrepDF(df, wordLen, maxLen, title, output):
#
    # Determine numeric values for each word in the dataframe
    print('Preparing the dataframe for NLP...')
    # Use tokens to create maxtrix of all unique words
    values = df.values
    token = Tokenizer(num_words = wordLen)
    token.fit_on_texts(values)
    wordnum = len(token.word_index) + 1
#
    # Replace text values with numeric values
    repl = token.texts_to_sequences(values)
#
    # Append zeros to each sequence to make each sequence equal length
    pads = pad_sequences(repl,
                         maxlen = maxLen)
    df = pd.DataFrame(pads)
    df.to_csv(output +
              title)
#
    return wordnum, pads   
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
def SplitDataset(X, y):
#
    # Split the dataframe into training and testing dataframes
    print("Splitting the dataframe into training and testing datasets...")
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=21)
    return X_train, X_test, y_train, y_test
#
# Method: TestModel
    # Inputs:
    #     model = A TensorFlow model object
    #     pads = An array type object
    #     df = An array type object
    # Processing:
    #     The TestModel method tests a TensorFlow model against a
    #     test dataset.
    # Output:
    #     None
#
def TestModel(model, pads, df):
#
    # Test the NLP model
    print("Testing LSTM model...")
    test_results = model.evaluate(pads,
                                  df)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {100*test_results[1]}%')
#
# Method: TrainModel
    # Inputs:
    #     pads = An array type object
    #     df = An array type object
    #     model = A TensorFlow model object
    #     batch = An int type object
    #     epoch = An int type object
    #     output = Folder directory as string object
    # Processing:
    #     The TrainModel method fits the NLP model to the dataset
    #     using the determined parameters
    # Output:
    #     None
#
def TrainModel(pads, df, model, batch, epoch, output):
#
    # Train the NLP model
    print("Training LSTM model...")
    callback = EarlyStopping(monitor = 'loss',
                             patience = 3)
    model = model.fit(pads,
                      df,
                      validation_split = 0.2,
                      epochs = epoch,
                      batch_size = batch,
                      callbacks = [callback])
#
    # Plot the loss on a graph
    loss = model.history['loss']
    epochs = range(1,
                   len(loss) + 1)
    plt.plot(epochs,
             loss,
             label = 'Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(output +
                'Training Loss.png')
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
               'D213_Advanced_Data_Analytics\\' +
               'NLM2 Task 2 - Sentiment Analysis Using Neural Networks\\')
    outDir = (directory +
              'Output\\')
    dataset = (directory +
               'imdb_labelled.csv')
    predict = 'reviewText'
    target = 'rating'
    char = ['.', ';', ':', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')',
            '-', '_', '=', '+', '`', '~', '{', '[', '}', ']', '|', '<', '>',
            '/', '?', '"', "'", '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '0', ',']
    wordLen = 200
    maxLen = 150
    vectLen = 10
    batch = 150
    epoch = 150
#
    # Clean the dataset
    df = CleanDF(dataset,
                 predict,
                 char,
                 outDir)
#
    # Split the dataframe into training and testing datasets
    x = df[predict]
    y = df[target]
    X_train, X_test, y_train, y_test = SplitDataset(x,
                                                    y)
#
    # Replace the text values in the datasets with numeric values
    trainWord, trainPads = PrepDF(X_train,
                                  wordLen,
                                  maxLen,
                                  'Train Dataset Sequence.csv',
                                  outDir)
    testWord, testPads = PrepDF(X_test,
                                wordLen,
                                maxLen,
                                'Test Dataset Sequence.csv',
                                outDir)
#
    # Build the LSTM model
    model = BuildModel(trainWord,
                       vectLen,
                       maxLen)
#
    # Export the model's summary to a text file
    ExportSummary(model,
                  outDir,
                  'Training_Model_Summary')
#
    # Train the LSTM model
    TrainModel(trainPads,
               y_train,
               model,
               batch,
               epoch,
               outDir)
#
    # Test the LSTM model
    TestModel(model,
              testPads,
              y_test)
#