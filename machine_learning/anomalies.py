import numpy as np 
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from adversarial_autoencoder import AAE_model, get_model, create_dataset
from oc_svm import svm_anomalies


def compare_anomalies():
    '''
    Method to use the adversarial autoencoder and the one-class svm to compare anomaly detection
    '''

    # loading model and dataset
    X, y, _, _, test_set, test_labels, train_ids, test_ids = create_dataset(remove_categorical=True)
    aae = get_model()

    # computing anomalies
    aae_train_anoms, aae_test_anoms = aae.detect_anomalies(X, train_ids, test_set, test_ids)
    svm_train_anoms, svm_test_anoms = svm_anomalies(X, train_ids, test_set, test_ids)

    for t_id in train_ids:
        print("ID: %s | AAE: %d | SVM: %d" % (t_id, aae_train_anoms[t_id], svm_train_anoms[t_id]))

    for t_id in test_ids:
        print("ID: %s | AAE: %d | SVM: %d" % (t_id, aae_test_anoms[t_id], svm_test_anoms[t_id]))
    

def get_anomalies():
    '''
    Function to extract anomalies and their corresponding data points
    '''
    
    # loading model and dataset
    X, y, _, _, test_set, test_labels, train_ids, test_ids = create_dataset(remove_categorical=True)
    aae = get_model()

    aae_train_anoms, aae_test_anoms = aae.detect_anomalies(X, train_ids, test_set, test_ids)
    svm_train_anoms, svm_test_anoms = svm_anomalies(X, train_ids, test_set, test_ids)

    # computing anomalies in latent space
    latent_train = aae.EncoderGenerator.call(X)
    latent_test = aae.EncoderGenerator.call(test_set)
    latent_svm_train, latent_svm_test = svm_anomalies(latent_train, train_ids, latent_test, test_ids)
    
    return aae_train_anoms, aae_test_anoms, svm_train_anoms, svm_test_anoms, latent_svm_train, latent_svm_test, X, test_set, y, test_labels, train_ids, test_ids


def convert_tsne(data):
    '''
    Function to convert 
    '''

    tsne = TSNE(n_components=2)
    embedded = tsne.fit_transform(data)
    return embedded

def plot_anomalies(aae_anom, svm_anom, t_data, ids, plot_latent=False, latent=None, labels=None):
    '''
    Function to visualize anomalies in 2D using TSNE

    Keyword Args:
    anomalies - a list of [-1 or 1] indicating anomalies or not (-1 anomaly) (1 not anomaly)
    data - the values of the data points
    '''

    x, y = zip(*t_data)
    
    aae_colors = []
    svm_colors = []

    # assigning colors for visualization
    for i in range(len(ids)):
        if aae_anom[ids[i]] == 1:
            aae_colors.append("blue")
        else:
            aae_colors.append("red")

        if svm_anom[ids[i]] == 1:
            svm_colors.append("blue")
        else:
            svm_colors.append("red")    

    aae_colors = np.array(aae_colors)
    svm_colors = np.array(svm_colors)

    if plot_latent:

        latent_colors = []
        gt_colors = []

        # assigning colors for visualization
        for i in range(len(ids)):
            if latent[ids[i]] == 1:
                latent_colors.append("blue")
            else:
                latent_colors.append("red") 
            
            if labels[i] == 1:
                gt_colors.append("blue")
            else:
                gt_colors.append("red")

        aae_colors = np.array(aae_colors)
        svm_colors = np.array(svm_colors)

        plt.subplot(2, 2, 1)
        plt.scatter(x, y, c=aae_colors, s=5)

        # plotting legend
        red_patch = mpatches.Patch(color='red', label='Anomalies')
        blue_patch = mpatches.Patch(color='blue', label='Normal')
        plt.legend(handles=[red_patch, blue_patch], loc="upper left")
        plt.title("Adversarial Autoencoder")

        plt.subplot(2, 2, 2)
        plt.scatter(x, y, c=svm_colors, s=5)

        # plotting legend
        red_patch = mpatches.Patch(color='red', label='Anomalies')
        blue_patch = mpatches.Patch(color='blue', label='Normal')
        plt.legend(handles=[red_patch, blue_patch], loc="upper left")
        plt.title("One-Class SVM")
    
        plt.subplot(2, 2, 3)
        plt.scatter(x, y, c=latent_colors, s=5)

        # plotting legend
        red_patch = mpatches.Patch(color='red', label='Anomalies')
        blue_patch = mpatches.Patch(color='blue', label='Normal')
        plt.legend(handles=[red_patch, blue_patch], loc="upper left")
        plt.title("One-Class SVM in Latent Space")

        plt.subplot(2, 2, 4)
        plt.scatter(x, y, c=gt_colors, s=5)

        # plotting legend
        red_patch = mpatches.Patch(color='red', label='Successful')
        blue_patch = mpatches.Patch(color='blue', label='Unsuccessful')
        plt.legend(handles=[red_patch, blue_patch], loc="upper left")
        plt.title("Ground Truth Labels")

    else:
        plt.subplot(1, 2, 1)
        plt.scatter(x, y, c=aae_colors, s=5)

        # plotting legend
        red_patch = mpatches.Patch(color='red', label='Anomalies')
        blue_patch = mpatches.Patch(color='blue', label='Normal')
        plt.legend(handles=[red_patch, blue_patch], loc="upper left")
        plt.title("Adversarial Autoencoder")

        plt.subplot(1, 2, 2)
        plt.scatter(x, y, c=svm_colors, s=5)

        # plotting legend
        red_patch = mpatches.Patch(color='red', label='Anomalies')
        blue_patch = mpatches.Patch(color='blue', label='Normal')
        plt.legend(handles=[red_patch, blue_patch], loc="upper left")
        plt.title("One-Class SVM")

    # plt.tight_layout()
    plt.show()

def main():
    aae_train, aae_test, svm_train, svm_test, latent_train, latent_test, train_data, test_data, train_labels, test_labels, train_ids, test_ids = get_anomalies()

    t_train = convert_tsne(train_data)
    t_test = convert_tsne(test_data)

    # plot_anomalies(aae_train, svm_train, t_train, train_ids)
    # plot_anomalies(aae_test, svm_test, t_test, test_ids)

    plot_anomalies(aae_train, svm_train, t_train, train_ids, plot_latent=True, latent=latent_train, labels=train_labels)
    plot_anomalies(aae_test, svm_test, t_test, test_ids, plot_latent=True, latent=latent_test, labels=test_labels)

main()
