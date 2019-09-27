import tensorflow as tf
import numpy as np 
import pandas as pd
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy


class Autoencoder(Model):
    def __init__(self, input_size, latent_size, rate, learning_rate,
                 discrim_weight=1, gen_weight=1, decode_weight=1):
        '''
        Constructor for an autoencoder
        '''

        super().__init__()

        # hyperparameters
        self.rate = rate
        self.learning_rate = learning_rate
        self.latent_size = latent_size
        self.input_size = input_size

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.aenc_model = tf.keras.Sequential([
            #encoder framework
            Dense(256, activation=tf.nn.sigmoid), Dropout(rate),
            Dense(128, activation=tf.nn.sigmoid), Dropout(rate), 
            Dense(64, activation=tf.nn.sigmoid),Dropout(rate),
            Dense(self.latent_size, activation=tf.nn.relu),
            # decoder framework
            Dense(64, activation=tf.nn.sigmoid), Dropout(rate),
            Dense(128, activation=tf.nn.sigmoid), Dropout(rate),
            Dense(256, activation=tf.nn.sigmoid), Dropout(rate),
            Dense(self.input_size, activation=tf.nn.tanh)
        ])    

    def train(self, X, y, validation_set, validation_labels, num_epochs=10, batch_size=50, latent_size=16):
        self.aenc_model.compile(optimizer='adam', loss='binary_crossentropy')
        self.aenc_model.fit(X, y,
                epochs=num_epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(validation_set, validation_labels))

def create_dataset():
    '''
    Function to set up the processing of data before passing it through the model for training
    '''

    df = pd.read_csv("../data/processed_data/full_normalized_labelled_zscore.csv")
    labels = np.array(df["Label (%)"].tolist())

    df.drop(columns=["Label (%)"], inplace=True)
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

def main():

    # hyperparameters
    latent_size = 16
    batch_size = 100
    rate = 0.2
    learning_rate = 0.001

    X, y, validation_set, validation_labels, test_set, test_labels = create_dataset()
    pass_size = np.shape(X)[1]

    # creating model and training
    model = Autoencoder(pass_size, latent_size, rate, learning_rate)
    model.train(X, y, validation_set, validation_labels, 
                batch_size=batch_size, latent_size=latent_size)


main()