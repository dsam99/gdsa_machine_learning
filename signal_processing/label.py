import pandas as pd

'''
Changes in data volumes 
rover (lander_return _value) -> orbiter (orbiter_return_value) -> tds (convert from frames) -> gds (dataActual)

Final pass in gds_es database -> figure out how to implement
'''


def label_percentage(dataframe, percent, save_to_csv=False, outfile=None):
    '''
    Function to label passes based on a certain percentage lost in the downlink process

    Keyword Args:
    dataframe - a dataframe containing the passes of data
    percent - the percentage threshold for labelling as bad

    Return:
    None
    '''

    labels = []
    percent /= 100

    # labelling based on percentage of data lost
    for _, row in dataframe.iterrows():

        if row["MAROS_orbiter_return_value"] == 0:
            labels.append(0)
            continue

        if row["MAROS_lander_return_value"] == 0:
            labels.append(0)
            continue

        if row["rover_to_orbiter_delta"] / row["MAROS_orbiter_return_value"] > percent:
            labels.append(0)
            continue
        
        if row["orbiter_to_TDS_delta"] / row["MAROS_lander_return_value"] > percent:
            labels.append(0)
            continue

        if row["TDS_insync_tf_32_frames"] != 0:
            if row["TDS_to_GDS_delta"] / row["TDS_insync_tf_32_frames"] > percent:
                labels.append(0)
                continue
            else:
                labels.append(1)
                continue

        elif row["TDS_insync_tf_0_frames"] != 0:
            if row["TDS_to_GDS_delta"] / row["TDS_insync_tf_0_frames"] > percent:
                labels.append(0)
                continue
            else:
                labels.append(1)
                continue
        else:
            labels.append(1)


    print(dataframe.shape)
    print(len(labels))

    dataframe.insert(0, "Label (%)", labels)

    post_process(dataframe)

    if save_to_csv:
        dataframe.to_csv(outfile, index=False)

    return dataframe


def label_mbs(dataframe, mbs):
    '''
    Function to label passes based on a certain amount of mbs lost in the downlink process

    Keyword Args:
    dataframe - a dataframe containing the passes of data
    mbs - the # of megabytes threshold for labelling as bad

    Return:
    a modified dataframe containing labelled data
    '''

    labels = []

    # labelling based on percentage of data lost
    for _, row in dataframe.iterrows():
        labelled = False

        if row["MAROS_orbiter_return_value"] == 0:
            labels.append(0)
            continue

        if row["MAROS_lander_return_value"] == 0:
            labels.append(0)
            continue

        if row["rover_to_orbiter_delta"] > mbs:
            labels.append(0)
            continue
        
        if row["orbiter_to_TDS_delta"]> mbs:
            labels.append(0)
            continue

        if row["TDS_insync_tf_32_frames"] != 0:
            if row["TDS_to_GDS_delta"] > mbs:
                labels.append(0)
                continue
            else:
                labels.append(1)
                continue

        elif row["TDS_insync_tf_0_frames"] != 0:
            if row["TDS_to_GDS_delta"] > mbs:
                labels.append(0)
                continue
            else:
                labels.append(1)
                continue
        else:
            labels.append(1)

    dataframe.insert(0, "Label (mbs)", labels)
    return dataframe

def label_last_pass(dataframe):
    '''
    Function to label passes as bad if they aren't the last pass within the range
    to check for incomplete re-passes

    Keyword Args:
    dataframe - a dataframe containing the passes of data

    Return:
    a modified dataframe containing labelled data
    '''

    # how to implement?


def post_process(df):
    '''
    Final processing for a dataframe before it is passed to ML algorithms
    '''

    # dropping unnecessary columns
    df.drop(columns=["GDS__id", "windowID", "TDS_endScet", "TDS_endRct", "TDS_insync_startErt",
                     "TDS_relayProductID", "Unnamed: 0", "TDS_expirationTime", "TDS_outasync_startErt",
                     "TDS_dataType", "TDS_creationTime", "GDS_dataActual", "MAROS_lander_return_value",
                     "MAROS_orbiter_return_value", "relayProductId", "TDS_insync_megabits"], inplace=True)

if __name__ == "__main__":
    x = pd.read_csv("../data/processed_data/898_1500_processed.csv")
    label_percentage(x, 5, True, "../data/processed_data/898_1500_labelled.csv")
    

