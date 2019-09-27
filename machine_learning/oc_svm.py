import numpy as np 
import pandas as pd
from sklearn.svm import OneClassSVM

def svm_anomalies(train_data, train_oids, test_data, test_oids):
    '''
    Function to detect anomalies given training data

    Keyword Args:
    train_data - training data
    train_oids - overflight ids for training data
    test_data - testing data
    test_oids - overflight ids for testing data

    Returns:
    two dictionaries of oid -> anomaly (-1 or 1)

    '''

    OC_SVM = OneClassSVM(kernel="rbf", degree=20)
    train_anomalies = OC_SVM.fit_predict(train_data)
    test_anomalies = OC_SVM.predict(test_data)

    train_dict = {}
    test_dict = {}

    for i in range(len(train_oids)):
        train_dict[train_oids[i]] = train_anomalies[i]

    for i in range(len(test_oids)):
        test_dict[test_oids[i]] = test_anomalies[i]

    return train_dict, test_dict

def create_dataset(remove_categorical=False):
    '''
    Function to set up the processing of data before passing it through the model for training
    '''

    df = pd.read_csv("../data/processed_data/337-2450_ids_normalized_zscore.csv")

    if remove_categorical:
        df.drop(columns=["orbiter_MEX", "orbiter_nan", "orbiter_MVN",	"orbiter_DTE",	"orbiter_TGO",	"orbiter_ODY",	"orbiter_MRO",	    
                         "TDS_dssId_64",	"TDS_dssId_15",	"TDS_dssId_45",	"TDS_dssId_14",	"TDS_dssId_43",	
                         "TDS_dssId_36",	'TDS_dssId_63',	"TDS_dssId_24", "TDS_dssId_34",	'TDS_dssId_0',	
                         "TDS_dssId_26",	"TDS_dssId_65",	"TDS_dssId_54",	"TDS_dssId_25",	"TDS_dssId_35",	
                         "TDS_dssId_55",	'TDS_dssId_50'], inplace=True)

    labels = np.array(df["GroundTruth"].tolist())

    df.drop(columns=["DylanLabel", "GDS__id", "MiguelLabel", "Unnamed: 0_y", "GDSLabel", 
                     "BinaryGDSLabel", "GroundTruth"], inplace=True)

    data = df.values
    features = df.keys().tolist()

    # randomly shuffling data
    random_order = np.random.permutation(len(data))
    data, labels = data[random_order], labels[random_order]

    # splitting data train (70%), validation (20%), test (10%)
    total = len(data)
    
    # randomly shuffling data
    train_data = data[:int(round(8 / 10 * total))]
    train_labels = labels[:int(round(8 / 10 * total))]

    test_data = data[int(round(8 / 10 * total)):]
    test_labels = labels[int(round(8 / 10 * total)):]

    return train_data, train_labels, test_data, test_labels, features

def correct(result_1, result_2):
    '''
    Function to check if anomaly prediction and label match
    '''

    if result_1 == -1:
        if result_2 == 0:
            return True
    else:
        if result_2 == 1:
            return True
    
    return False

def main():
    train_data, train_labels, test_data, test_labels, features = create_dataset(remove_categorical=True)
    train_anoms, test_anoms = svm_anomalies(train_data, test_data)

    for i in range(len(train_labels)):
        if correct(train_anoms[i], train_labels[i]):
            print("Anomaly: %d | Label: %d | Correct" % (train_anoms[i], train_labels[i]))
        else:
            print("Anomaly: %d | Label: %d | Incorrect" % (train_anoms[i], train_labels[i]))

    for i in range(len(test_labels)):
        if correct(test_anoms[i], test_labels[i]):
            print("Anomaly: %d | Label: %d | Correct" % (test_anoms[i], test_labels[i]))
        else:
            print("Anomaly: %d | Label: %d | Incorrect" % (test_anoms[i], test_labels[i]))
main()