import sklearn
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from scipy.stats import pearsonr

def p_score(df):
    '''
    Function to use p-score (correlation) for feature selection

    Keyword Args:
    df - dataframe of dataset
    '''

    labels = df["Label (%)"].tolist()
    df.drop(columns=["Label (%)", "GDS__id"], inplace=True)

    features = np.array(df.keys().tolist())
    data_points = df.values

    p_dict = {}

    for i in range(len(features)):
        feature = data_points.T[i]
        p_val, tails_p = pearsonr(feature, labels)
        # print(features[i] + " : p-value = " + str(p_val) + " | 2-tailed p-value = " + str(tails_p))
        p_dict[features[i]] = p_val

    sorted_d = sorted(p_dict.items(), key=lambda x:abs(x[1]), reverse=True)
    for i in sorted_d:
        print(i[0] + " : " + str(i[1]))
    
    return p_dict

def pca(df, num_components):
    '''
    Function to use principal component analysis

    Keyword Args:
    df - dataframe of dataset
    '''

    df.drop(columns=["Label (%)", "GDS__id"], inplace=True)

    features = np.array(df.keys().tolist())
    data_points = df.values

    # pca = PCA(n_components=num_components)
    # components = pca.fit_transform(data_points)
    # print(components)

    means = np.mean(data_points.T, axis=1)
    centered_data = data_points - means

    covariances = np.cov(centered_data.T)
    eigenvals = np.linalg.eigvals(covariances) 

    for i in range(len(eigenvals)):
        print(features[i] + " : " + str(eigenvals[i]))

def variance_selection(df):
    '''
    Function to display features with low variance in the dataframe

    Keyword Args:
    df - dataframe of dataset

    Returns: 
    df - dataframe with least features removed
    '''

    df.drop(columns=["Label (%)", "GDS__id"], inplace=True)

    features = np.array(df.keys().tolist())
    data_points = df.values

    v_dict = {}

    for i in range(len(features)):
        feature = data_points.T[i]
        variance = np.var(feature)
        v_dict[features[i]] = variance

    sorted_d = sorted(v_dict.items(), key=lambda x:abs(x[1]), reverse=True)
    for i in sorted_d:
        print(i[0] + " : " + str(i[1]))

    # implement removing of any features that are desired

    return v_dict
    
def f_classification_test(df):
    '''
    Function to display the results of features and the chi-squared test

    Keyword Args:
    df - dataframe of dataset
    '''

    labels = df["Label (%)"].tolist()
    df.drop(columns=["Label (%)", "GDS__id"], inplace=True)

    features = np.array(df.keys().tolist())
    data_points = df.values

    f_vals, _ = f_classif(data_points, labels)
    f_dict = {}

    for i in range(len(features)):
        f_dict[features[i]] = f_vals[i]

    sorted_d = sorted(f_dict.items(), key=lambda x:abs(x[1]), reverse=True)
    for i in sorted_d:
        print(i[0] + " : " + str(i[1]))

if __name__ == "__main__":
    df = pd.read_csv("../data/processed_data/full_normalized_ids_labelled_zscore.csv")
    p = p_score(df)
    # pca(df, 10)
    df = pd.read_csv("../data/processed_data/full_normalized_ids_labelled_zscore.csv")
    v = variance_selection(df)
    # f_classification_test(df)

    # creating csv file 
    df = pd.DataFrame.from_dict([p, v])

    features = np.array(df.keys())
    data = df.values
    new_df = pd.DataFrame(data.transpose(), columns=["p-score", "variance"])

    new_df.insert(0, "feature name", features)
    print(new_df.head())
    new_df.to_csv("feature_information.csv", index=False)
    print("made csv")