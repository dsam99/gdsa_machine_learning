import tensorflow as tf
import numpy as np 
import pandas as pd
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy

class ks_AAE_model():

    '''
    Implementation of an Adversarial Autoencoder using tensorflow
    '''

    def __init__(self, input_size, latent_size, 
                 rate, learning_rate,
                 discrim_weight=1, gen_weight=1, decode_weight=1):

        # hyperparameters
        self.rate = rate
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.latent_size = latent_size

        self.Encoder = self.encoder()
        self.Decoder = self.decoder()
        self.Discriminator = self.discriminator()

        # optimizer for all networks
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # building autoencoder framework
        input_data = Input(shape=(self.input_size,), name="aenc_input")
        latent_data = self.Encoder(input_data)
        reconstructed_input = self.Decoder(latent_data)

        prob_fake = self.Discriminator(latent_data)

        # compiling models and constructing autoencoder
        self.Discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer, 
                                   metrics=['accuracy'])
        self.AAE = Model(input_data, [reconstructed_input, prob_fake])
        self.AAE.compile(loss=['mse', 'binary_crossentropy'], optimizer=self.optimizer)


    def encoder(self):
        '''
        Function to create the encoder network
        '''
        
        # creating generator network and layers
        input_layer = Input(shape=self.input_size, name="enc_input")
        layer_1 = Dense(128, activation=tf.nn.sigmoid)(input_layer)
        dp_1 = Dropout(self.rate)(layer_1)

        # second dense layer with sigmoid and dropout
        layer_2 = Dense(64, activation=tf.nn.sigmoid)(dp_1)
        dp_2 = Dropout(self.rate)(layer_2)

        # third dense layer with sigmoid and dropout
        layer_3 = Dense(32, activation=tf.nn.sigmoid)(dp_2)
        dp_3 = Dropout(self.rate)(layer_3)

        # final layer
        out_layer = Dense(self.latent_size, activation=tf.nn.sigmoid)(dp_3)

        return Model(input_layer, out_layer)
    
    def decoder(self):
        '''
        Function to create decoder network
        '''

        # creating decoder network and layers
        input_layer = Input(shape=self.latent_size, name="dec_input")
        layer_1 = Dense(32, activation=tf.nn.sigmoid)(input_layer)
        dp_1 = Dropout(self.rate)(layer_1)

        # second dense layer with sigmoid and dropout
        layer_2 = Dense(64, activation=tf.nn.sigmoid)(dp_1)
        dp_2 = Dropout(self.rate)(layer_2)

        # third dense layer with sigmoid and dropout
        layer_3 = Dense(128, activation=tf.nn.sigmoid)(dp_2)
        dp_3 = Dropout(self.rate)(layer_3)

        # final layer using tanh to make [-1, 1]
        out_layer = Dense(self.input_size, activation=tf.nn.tanh)(dp_3)

        return Model(input_layer, out_layer)
    
    def discriminator(self):
        '''
        Function to create discriminator network
        '''

        # creating decoder network and layers
        input_layer = Input(shape=self.latent_size, name="dis_input")
        layer_1 = Dense(32, activation=tf.nn.sigmoid)(input_layer)
        dp_1 = Dropout(self.rate)(layer_1)

        # second dense layer with sigmoid and dropout
        layer_2 = Dense(64, activation=tf.nn.sigmoid)(dp_1)
        dp_2 = Dropout(self.rate)(layer_2)

        # third dense layer with sigmoid and dropout
        layer_3 = Dense(32, activation=tf.nn.sigmoid)(dp_2)
        dp_3 = Dropout(self.rate)(layer_3)

        # final layer using sigmoid to make probability of fake/real
        out_layer = Dense(1, activation=tf.nn.sigmoid)(dp_3)

        return Model(input_layer, out_layer)
    

    def train(self, X, y, validation_set, validation_labels, 
              num_epochs=200, batch_size=50, latent_size=16):
        '''
        Function to train keras implementation of AAE model
        '''

        for ep in range(num_epochs):

            # randomly shuffling data
            random_order = np.random.permutation(len(X))
            X, y = X[random_order], y[random_order]

            random_order_validation = np.random.permutation(len(validation_set))
            validation_set, validaton_labels = validation_set[random_order_validation], validation_labels[random_order_validation]

            avg_dis_loss = 0
            avg_aenc_loss = 0

            for batch in range(0, len(X), batch_size):
                # making sure not going out of bounds
                if batch + batch_size > len(X):
                    continue
                else:
                    x_input = X[batch:batch+batch_size] 
                    y_input = y[batch:batch+batch_size] 
            
                # training autoencoder
                aenc_loss = self.AAE.train_on_batch(x_input, [x_input, tf.ones_like(x_input)])

                # training discriminator

                # getting latent space inputs
                latent_input = self.Encoder.predict(x_input)
                latent_noise = gen_noise(batch_size, latent_size)

                real_loss = self.Discriminator.train_on_batch(latent_noise, tf.ones_like(latent_noise))
                fake_loss = self.Discriminator.train_on_batch(latent_input, tf.zeros_like(latent_noise))

                discrim_loss = 0.5 * tf.math.add(real_loss, fake_loss)

                avg_aenc_loss += np.mean(aenc_loss) * 0.001
                avg_dis_loss += np.mean(discrim_loss)

            avg_aenc_loss /= batch_size
            avg_dis_loss /= batch_size

            print("[Epoch " + str(ep) + " of " + str(num_epochs) + "]")
            print("--------------------------------------------------")
            print("Average Discriminator Loss: " + str(avg_dis_loss))
            print("Average Autoencoder Loss: " + str(avg_aenc_loss))

def gen_noise(batch_size, latent_size):
    '''
    Function to generate noise from -1 to 1 from a Gaussian distribution

    Keywords:
    batch_size - the size of the output batch
    latent_size - the number of dimensiosn of each data point
    '''

    return np.random.normal(loc=0, scale=1, size=(batch_size, latent_size))


def plot_training(training_acc, validation_acc):
    '''
    Function to plot the results of training (training accuracy, validation accuracy)

    Keyword Args:
    training_acc - a list of the training accuracies
    validation_acc - a list of the valdiation accuracies
    '''

    plt.plot(training_acc, np.range(len(training_acc)), linewidth=4)
    plt.plot(validation_acc, np.range(len(validation_acc)), linewidth=4)

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")

    plt.legend(["Average Training Accuracy", "Validation Accuracy"], loc='upper left')
    plt.show()

def create_dataset():
    '''
    Function to set up the processing of data before passing it through the model for training
    '''

    df = pd.read_csv("../data/processed_data/full_normalized_labelled_zscore.csv")
    labels = np.array(df["Label (%)"].tolist())

    df.drop(columns=["Label (%)"], inplace=True)
    data = df.values

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
    learning_rate = 0.01

    X, y, validation_set, validation_labels, test_set, test_labels = create_dataset()
    pass_size = np.shape(X)[1]

    model = ks_AAE_model(pass_size, latent_size, rate, learning_rate)
    model.train(X, y, validation_set, validation_labels, 
                batch_size=batch_size, latent_size=latent_size)

if __name__ == "__main__":
    main()