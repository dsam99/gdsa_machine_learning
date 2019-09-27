import sys
# adding other modules to path
sys.path.append("../signal_processing")
sys.path.append("../data_collection")

import numpy as np
import pandas as pd

from basic_model import Basic_Model
from signal_processor import process_dataframe, normalize
from label import labeller
from get_data import get_data_from_sol
from automate import automate_data_collection

def demo(sol_start, sol_end):
    '''
    Function to run the demo for completely automating data collection, processing, and classification
    '''

    data = []

    for sol in range(sol_start, sol_end):
        if sol % 10 == 0:
            print("Sol " + str(sol))
        data.append(get_data_from_sol(str(sol)))
    
    print(data)
    
    full_dataset = pd.concat(data, ignore_index=True)
    print(full_dataset)
    processed_df = process_dataframe(full_dataset)
    print(processed_df)
    labelled_df = labeller(processed_df, 5)
    labelled_df.to_csv("demo_file.csv", index=False)
    # print(labelled_df.keys().tolist())
    normalized_df = normalize(labelled_df, min_max=False, z_score=True)
    print(normalized_df.keys().tolist())

    # processing data and removing categoricals
    labels = normalized_df["Label (%)"].tolist()
    ids = normalized_df["GDS__id"].tolist()

    df = normalized_df.drop(columns=["Label (%)", "GDS__id"])
    df = drop_cat(df)

    data_points = df.values

    # print(data_points)

    # loading and using model
    model = load("./saved_models/ffnn_0806_nocat.h5")
    preds = []

    for i in range(len(data_points)):
        q = model.predict(np.array([data_points[i],]))[0]
        preds.append(q)


    # flattening predictions list
    preds = np.array(preds).flatten()

    num_correct = 0

    # print(preds, labels)

    for i in range(len(labels)):
        print("%s | Predicted: %f | Actual: %d" % (ids[i], preds[i], labels[i]))

        if preds[i] >= 0.5:
            if labels[i] == 1:
                num_correct += 1
        else:
            if labels[i] == 0:
                num_correct += 1
    
    print("Correct: %d/%d" % (num_correct, len(labels)))

def drop_cat(df):
    '''
    Function to remove categorical variables from dataset
    '''
    
    keys = df.keys().tolist()

    cat_list = [
        "orbiter_MEX", "orbiter_nan", "orbiter_MVN","orbiter_DTE",	"orbiter_TGO",	"orbiter_ODY",	"orbiter_MRO",	
        "TDS_dssId_64",	"TDS_dssId_15",	"TDS_dssId_45",	"TDS_dssId_14",	"TDS_dssId_43",	"TDS_dssId_36",	'TDS_dssId_63',	
        "TDS_dssId_24", "TDS_dssId_34",	'TDS_dssId_0',	"TDS_dssId_26",	"TDS_dssId_65",	"TDS_dssId_54",	"TDS_dssId_25",	
        "TDS_dssId_35",	"TDS_dssId_55",	'TDS_dssId_50', "TDS_relayProductID"
    ]

    to_remove = list(filter(lambda x: x in keys, cat_list))
    df.drop(columns=to_remove, inplace=True)
    

    return df

def load(filename, model_params={"input_size": 18}):
    '''
    Function to load saved FFNN model
    '''

    model = Basic_Model(model_params["input_size"])
    model.model.load_weights(filename)
    return model

demo(2480, 2482)
