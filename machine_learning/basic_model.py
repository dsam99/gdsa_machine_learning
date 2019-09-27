import sys
import os
import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.utils import plot_model

# setting environment to allow duplicated KMP library
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Basic_Model():
    '''
    Simple model of a neural network to try classifying differnet passes
    of Mars rover data
    '''

    def __init__(self, input_size, alpha=0.0003):
        self.model = Sequential()
        self.learning_rate = alpha
    
        # really simple architecture
        self.model.add(Dense(64, input_shape=(input_size,), activation='relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(32, activation="relu"))

        self.model.add(Dense(1, activation='sigmoid', name="output_layer"))

        # using adam optimizer
        optimizer = Adam(lr=self.learning_rate)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    def train(self, X, y, validation_x, validation_y):
        '''
        Function to train a feed-forward neural network on input data

        Keyword Args:
        X - input data
        y - labels
        '''

        result = self.model.fit(X, y, epochs=400, batch_size=20, validation_data=(validation_x, validation_y))            
        # function for plotting training and validation accuracy for training
        plot_training(result)

        return result

    def predict(self, X):
        '''
        Function to apply a model to predict based on input data

        Keyword Args:
        X - data examples

        Returns:
        a numpy array of predicted labels
        '''

        return self.model.predict(X)
    
    def test(self, X, y):
        '''
        Function to test the model on held-out test data

        Keyword Args:
        X - test examples
        y - the true label for the test examples

        Returns:
        a float representing the loss of the model
        '''

        return self.model.test_on_batch(X, y)
    
    def display(self):
        '''
        Function to visualize neural network
        '''

        plot_model(self.model, to_file="ffnn.png")


def create_dataset():
    '''
    Function to set up the processing of data before passing it through the model for training
    '''

    # df = pd.read_csv("../data/processed_data/full_normalized_labelled_zscore.csv")
    # df = pd.read_csv("../data/processed_data/full_normalized_labelled_minmax.csv")
    df = pd.read_csv("../data/processed_data/898-2450_ids_normalized_zscore.csv")

    labels = np.array(df["GroundTruth"].tolist())
    df.drop(columns=["DylanLabel", "GDS__id",
                     "MiguelLabel", "Unnamed: 0_y", "GDSLabel", "BinaryGDSLabel", "GroundTruth"
                    ], inplace=True)

    data = df.values

    # randomly shuffling data
    random_order = np.random.permutation(len(data))
    data, labels = data[random_order], labels[random_order]

    # splitting data train (70%), validation (20%), test (10%)
    total = len(data)
    
    train_data = data[:int(round(7 / 10 * total))]
    train_labels = labels[:int(round(7 / 10 * total))]

    validation_data = data[int(round(7 / 10 * total)): int(round(9 / 10 * total))]
    validation_labels = labels[int(round(7 / 10 * total)): int(round(9 / 10 * total))]

    test_data = data[int(round(9 / 10 * total)):]
    test_labels = labels[int(round(9 / 10 * total)):]

    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

def create_id_dataset(remove_categorical=False):
    '''
    Function to set up pprocessing of data and keep ID for later seeing percentage success
    based on different types of passes (MRO, TGO, MVN, etc.)

    Returns:
    data - the full dataset to predict on
    labels - the labels of the data points
    orbiters_dict - a dictionary from (data point -> GDS__id)
    '''

    # df = pd.read_csv("../data/processed_data/full_normalized_ids_labelled_zscore.csv")
    # df = pd.read_csv("../data/processed_data/full_normalized_337_2450_ids_labelled_zscore.csv")
    # df = pd.read_csv("../data/processed_data/898-2450_ids_normalized_zscore.csv")
    df = pd.read_csv("../data/processed_data/337-2450_ids_normalized_zscore.csv")

    labels = np.array(df["GroundTruth"].tolist())
    df.drop(columns=["DylanLabel", "MiguelLabel", "Unnamed: 0_y", "GDSLabel", "BinaryGDSLabel", "GroundTruth"
                    ], inplace=True)

    # option to remove categorical vars
    if remove_categorical:
        df.drop(columns=["orbiter_MEX", "orbiter_nan", "orbiter_MVN",	"orbiter_DTE",	"orbiter_TGO",	"orbiter_ODY",	"orbiter_MRO",	"TDS_dssId_64",	"TDS_dssId_15",	"TDS_dssId_45",	"TDS_dssId_14",	"TDS_dssId_43",	"TDS_dssId_36",	'TDS_dssId_63',	"TDS_dssId_24", "TDS_dssId_34",	'TDS_dssId_0',	"TDS_dssId_26",	"TDS_dssId_65",	"TDS_dssId_54",	"TDS_dssId_25",	"TDS_dssId_35",	"TDS_dssId_55",	'TDS_dssId_50'], inplace=True)

    data = df.values

    # randomly shuffling data
    random_order = np.random.permutation(len(data))
    data, labels = data[random_order], labels[random_order]

    # splitting data train (70%), validation (20%), test (10%)
    total = len(data)
    
    train_data = data[:int(round(7 / 10 * total))]
    train_labels = labels[:int(round(7 / 10 * total))]

    validation_data = data[int(round(7 / 10 * total)): int(round(9 / 10 * total))]
    validation_labels = labels[int(round(7 / 10 * total)): int(round(9 / 10 * total))]

    # dropping GDS__id from training and validation
    train_data = np.array([data[1:] for data in train_data])
    validation_data = np.array([data[1:] for data in validation_data])

    test_data = data[int(round(9 / 10 * total)):]
    test_labels = labels[int(round(9 / 10 * total)):]

    # grabbing ids from test data and dropping
    test_ids = np.array([x[0] for x in test_data])
    test_data = np.array([data[1:] for data in test_data])

    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels, test_ids

def plot_training(result):
    '''
    Function to plot the result of training

    Keyword Args:
    result - the result of training the keras model
    '''
    # Plot training & validation accuracy values
    plt.plot(result.history['accuracy'])
    plt.plot(result.history['val_accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train Dataset', 'Validation Dataset'], loc='upper left')
    plt.show()


def main_ids():
    X, y, validation_x, validation_y, test_x, test_y, test_id= create_id_dataset()

    input_size = np.shape(X)[1]

    # creating and training model
    model = Basic_Model(input_size)
    model.train(X, y, validation_x, validation_y)

    # splitting dataset into different orbiters
    MRO_data = []
    MRO_labels = []
    MRO_ids = []
    TGO_data = []
    TGO_labels = []
    TGO_ids = []
    ODY_data = []
    ODY_labels = []
    ODY_ids = []

    for i in range(len(test_x)):
        if test_id[i].startswith("MRO"):
            MRO_data.append(test_x[i])
            MRO_labels.append(test_y[i])
            MRO_ids.append(test_id[i])
        elif test_id[i].startswith("TGO"):
            TGO_data.append(test_x[i])
            TGO_labels.append(test_y[i])
            TGO_ids.append(test_id[i])

        elif test_id[i].startswith("ODY"):
            ODY_data.append(test_x[i])
            ODY_labels.append(test_y[i])
            ODY_ids.append(test_id[i])

    # converting to numpy arrays    
    MRO_data = np.array(MRO_data)
    MRO_labels = np.array(MRO_labels)
    TGO_data = np.array(TGO_data)
    TGO_labels = np.array(TGO_labels)
    ODY_data = np.array(ODY_data)
    ODY_labels = np.array(ODY_labels)

    print("Overall Test Accuracy: " + str(model.test(test_x, test_y)))

    print("MRO Dataset Results: " + str(len(MRO_data)) + " data points")
    print(model.test(MRO_data, MRO_labels))

    print("TGO Dataset Results:" + str(len(TGO_data)) + " data points")
    print(model.test(TGO_data, TGO_labels))

    print("ODY Dataset Results: " + str(len(ODY_data)) + " data points")
    print(model.test(ODY_data, ODY_labels))

    model.model.save('ffnn_0729.h5')  # creates a HDF5 file 'my_model.h5'
    bad_MRO_pass = []
    bad_TGO_pass = []
    bad_ODY_pass = []

    MRO_preds = model.predict(MRO_data)

    # looping through and getting error passes
    for i in range(len(MRO_preds)):
        pred = MRO_preds[i]
        if pred >= 0.5:
            pred = 1
        else:
            pred = 0
        if pred == MRO_labels[i]:
            continue
        else:
            print(MRO_ids[i])
            bad_MRO_pass.append((MRO_ids[i], MRO_data[i], MRO_labels[i]))
    
    # looping through and getting error passes
    TGO_preds = model.predict(TGO_data)

    # looping through and getting error passes
    for i in range(len(TGO_preds)):
        pred = TGO_preds[i]
        if pred >= 0.5:
            pred = 1
        else:
            pred = 0
        if pred == TGO_labels[i]:
            continue
        else:
            print(TGO_ids[i])
            bad_TGO_pass.append((TGO_ids[i], TGO_data[i], TGO_labels[i]))
    
    # looping through and getting error passes
    ODY_preds = model.predict(ODY_data)

    # looping through and getting error passes
    for i in range(len(ODY_preds)):
        pred = ODY_preds[i]
        if pred >= 0.5:
            pred = 1
        else:
            pred = 0
        if pred == ODY_labels[i]:
            continue
        else:
            print(ODY_ids[i])
            bad_ODY_pass.append((ODY_ids[i], ODY_data[i], ODY_labels[i]))
        
    
    print(len(bad_MRO_pass), len(bad_TGO_pass), len(bad_ODY_pass))
    # model.display()

def main():
    X, y, validation_x, validation_y, test_x, test_y = create_dataset()

    input_size = np.shape(X)[1]

    # creating and training model
    model = Basic_Model(input_size)
    model.train(X, y, validation_x, validation_y)

    # final test loss
    model.test(test_x, test_y)

def main_ids_no_categorical():
    X, y, validation_x, validation_y, test_x, test_y, test_id = create_id_dataset(remove_categorical=True)

    input_size = np.shape(X)[1]

    # creating and training model
    model = Basic_Model(input_size)
    model.train(X, y, validation_x, validation_y)

    # splitting dataset into different orbiters
    MRO_data = []
    MRO_labels = []
    MRO_ids = []
    TGO_data = []
    TGO_labels = []
    TGO_ids = []
    ODY_data = []
    ODY_labels = []
    ODY_ids = []

    for i in range(len(test_x)):
        if test_id[i].startswith("MRO"):
            MRO_data.append(test_x[i])
            MRO_labels.append(test_y[i])
            MRO_ids.append(test_id[i])
        elif test_id[i].startswith("TGO"):
            TGO_data.append(test_x[i])
            TGO_labels.append(test_y[i])
            TGO_ids.append(test_id[i])

        elif test_id[i].startswith("ODY"):
            ODY_data.append(test_x[i])
            ODY_labels.append(test_y[i])
            ODY_ids.append(test_id[i])

    # converting to numpy arrays    
    MRO_data = np.array(MRO_data)
    MRO_labels = np.array(MRO_labels)
    TGO_data = np.array(TGO_data)
    TGO_labels = np.array(TGO_labels)
    ODY_data = np.array(ODY_data)
    ODY_labels = np.array(ODY_labels)

    print("Overall Test Accuracy: " + str(model.test(test_x, test_y)))

    print("MRO Dataset Results: " + str(len(MRO_data)) + " data points")
    print(model.test(MRO_data, MRO_labels))

    print("TGO Dataset Results:" + str(len(TGO_data)) + " data points")
    print(model.test(TGO_data, TGO_labels))

    print("ODY Dataset Results: " + str(len(ODY_data)) + " data points")
    print(model.test(ODY_data, ODY_labels))

    model.model.save('ffnn_729_nocat.h5')  # creates a HDF5 file 'my_model.h5'

    bad_MRO_pass = []
    bad_TGO_pass = []
    bad_ODY_pass = []

    MRO_preds = model.predict(MRO_data)

    # looping through and getting error passes
    for i in range(len(MRO_preds)):
        pred = MRO_preds[i]
        if pred >= 0.5:
            pred = 1
        else:
            pred = 0
        if pred == MRO_labels[i]:
            continue
        else:
            print(MRO_ids[i])
            bad_MRO_pass.append((MRO_ids[i], MRO_data[i], MRO_labels[i]))
    
    # looping through and getting error passes
    TGO_preds = model.predict(TGO_data)

    # looping through and getting error passes
    for i in range(len(TGO_preds)):
        pred = TGO_preds[i]
        if pred >= 0.5:
            pred = 1
        else:
            pred = 0
        if pred == TGO_labels[i]:
            continue
        else:
            print(TGO_ids[i])
            bad_TGO_pass.append((TGO_ids[i], TGO_data[i], TGO_labels[i]))
    
    # looping through and getting error passes
    ODY_preds = model.predict(ODY_data)

    # looping through and getting error passes
    for i in range(len(ODY_preds)):
        pred = ODY_preds[i]
        if pred >= 0.5:
            pred = 1
        else:
            pred = 0
        if pred == ODY_labels[i]:
            continue
        else:
            print(ODY_ids[i])
            bad_ODY_pass.append((ODY_ids[i], ODY_data[i], ODY_labels[i]))
        
    
    print(len(bad_MRO_pass), len(bad_TGO_pass), len(bad_ODY_pass))


if __name__ == "__main__":
    # main_ids()
    main_ids_no_categorical()



