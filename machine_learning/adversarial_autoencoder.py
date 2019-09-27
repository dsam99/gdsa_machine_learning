import tensorflow as tf
import numpy as np 
import pandas as pd
import math

from tensorflow.keras import Model
import matplotlib.pyplot as plt
from tensorflow.keras.losses import mse
from tensorflow.keras.layers import Input
from scipy.stats import entropy
from numpy.linalg import norm
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
import pickle

# pulling models from other models.py file
from models import EncoderGenerator, Decoder, Discrminator, Discriminator_Multiclass

class AAE_model(Model):
    '''
    Implementation of an Adversarial Autoencoder using tensorflow
    '''

    def __init__(self, input_size, latent_size, rate,
                 discrim_weight=1, gen_weight=1, decode_weight=1):
        '''
        Constructor for an AAE

        Keyword Args:
        input_dimensions - the default image size is 28x28
        '''

        super().__init__()

        # hyperparameters
        self.rate = rate
        self.input_size = input_size
        self.latent_size = latent_size

        # separate networks in model
        self.EncoderGenerator = EncoderGenerator(self.input_size, self.latent_size, 
                                                 self.rate, gen_weight, 0.0001)
        self.Decoder = Decoder(self.input_size, self.latent_size,
                               self.rate, decode_weight, 0.0001)
        self.Discriminator = Discrminator(self.latent_size, self.rate, 
                                          discrim_weight, 0.01)
        
        # for multiclass case
        self.Discriminator_Multiclass = Discriminator_Multiclass(self.latent_size, self.rate, 
                                          discrim_weight, 0.01)
        
        # creating autoencoder and adversarial learning
        AE_input = Input(shape=(self.input_size,), name="AE_input")
        latent_data, reconstructed_input = self.call(AE_input)
        self.Autoencoder = Model(AE_input, reconstructed_input)

        # compiling models
        self.Autoencoder.compile(loss="mse", optimizer=self.EncoderGenerator.optimizer)
        # self.save_model(self.EncoderGenerator.layer_1, "saved_models/aae_computationmodel")

        self.threshold = None

    def call(self, input_tensor):
        '''
        Function to apply the adversarial autoencoder model
        '''

        # stacking encoder and decoder for training
        enc_output = self.EncoderGenerator.call(input_tensor)
        dec_output = self.Decoder.call(enc_output)

        return enc_output, dec_output

    def train(self, X, y, validation_set=None, validation_labels=None, 
              num_epochs=400, batch_size=50, latent_size=16, save=False):
        '''
        Training the adversarial autoencoder

        Keyword Args;
        X - input data
        y - labels
        '''

        # using validation data for anomaly detection
        v_loss_good = []
        v_loss_bad = []

        if not validation_labels is None and not validation_set is None:
            v_l_g, v_l_b = self.calc_validation(validation_set, validation_labels)
            v_loss_good.append(v_l_g)                
            v_loss_bad.append(v_l_b)
            print(v_l_g, v_l_b)

        for ep in range(num_epochs):

            # randomly shuffling data
            random_order = np.random.permutation(len(X))
            X, y = X[random_order], y[random_order]

            avg_dis_loss = 0
            avg_aenc_loss = 0
            avg_gen_loss = 0

            for batch in range(0, len(X), batch_size):
                # making sure not going out of bounds
                if batch + batch_size > len(X):
                    continue
                else:
                    x_input = X[batch:batch+batch_size] 
                    y_input = y[batch:batch+batch_size] 
             
                # RECONSTRUCTION PHASE (updating autoencoder - encoder and decoder)

                # training based on mse loss on reconstructing nput
                dec_loss = self.Autoencoder.train_on_batch(x_input, x_input)

                # REGULARIZATION PHASE (updating adversarial learning - discriminator and generator)
                noise = gen_noise(batch_size, latent_size)
                
                # updating discriminator weights
                with tf.GradientTape() as tape:
                    gen_output = self.EncoderGenerator.call(x_input)
                    dis_output_fake = self.Discriminator.call(gen_output)
                    dis_output_real = self.Discriminator.call(noise)
                    # Add extra losses created during this forward pass:
                    dis_loss = self.Discriminator.loss(dis_output_fake, dis_output_real)

                    grads = tape.gradient(dis_loss, self.Discriminator.trainable_weights)
                    self.Discriminator.optimizer.apply_gradients(zip(grads, self.Discriminator.trainable_weights))

                # updating generator weights           
                with tf.GradientTape() as tape:
                    gen_output = self.EncoderGenerator.call(x_input)
                    dis_output = self.Discriminator.call(gen_output)
                    # Add extra losses created during this forward pass:
                    gen_loss = self.EncoderGenerator.loss(dis_output, tf.ones_like(dis_output))

                    grads = tape.gradient(gen_loss, self.EncoderGenerator.trainable_weights)
                    self.EncoderGenerator.optimizer.apply_gradients(zip(grads, self.EncoderGenerator.trainable_weights))


                # summing all losses
                avg_dis_loss += dis_loss.numpy()
                avg_aenc_loss += dec_loss
                avg_gen_loss += gen_loss.numpy()

            # averaging loss values over epoch
            avg_dis_loss /= (len(X) / batch_size)
            avg_aenc_loss /= (len(X) / batch_size)
            avg_gen_loss /= (len(X) / batch_size)

            if not validation_labels is None and not validation_set is None:
                v_l_g, v_l_b = self.calc_validation(validation_set, validation_labels)
                v_loss_good.append(v_l_g)                
                v_loss_bad.append(v_l_b)


            print("[Epoch " + str(ep) + " of " + str(num_epochs) + "]")
            print("--------------------------------------------------")
            print("Average Discriminator Loss: " + str(avg_dis_loss))
            print("Average Autoencoder Loss: " + str(avg_aenc_loss))
            print("Average Generator Loss: " + str(avg_gen_loss))

            # printing validation results
            if not validation_labels is None and not validation_set is None:
                print("Validation Loss Good: " + str(v_l_g))
                print("Validation Loss Bad: "  + str(v_l_b))

            print("")

        # plot_validation(v_loss_good, v_loss_bad)
        
        # CALCULATE THRESHOLD HERE -> currently selecting 0.8 or if training was higher
        self.threshold = max(0.8, avg_aenc_loss)
    
    def train_multiclass(self, X, y, num_epochs=150, batch_size=50, latent_size=16, save=False):
        '''
        Training the adversarial autoencoder incorporating supervised labels

        Keyword Args;
        X - input data
        y - labels
        '''

        for ep in range(num_epochs):

            # randomly shuffling data
            random_order = np.random.permutation(len(X))
            X, y = X[random_order], y[random_order]

            avg_dis_loss = 0
            avg_aenc_loss = 0
            avg_gen_loss = 0

            for batch in range(0, len(X), batch_size):

                # making sure not going out of bounds
                if batch + batch_size > len(X):
                    continue
                else:
                    x_input = X[batch:batch+batch_size] 
                    y_input = y[batch:batch+batch_size] 
             
                # RECONSTRUCTION PHASE (updating autoencoder - encoder and decoder)

                # training based on mse loss on reconstructing nput
                dec_loss = self.Autoencoder.train_on_batch(x_input, x_input)

                # REGULARIZATION PHASE (updating adversarial learning - discriminator and generator)
                noise = gen_noise(batch_size, latent_size)
                
                # splitting data into successful and unsuccessful
                x_success = []
                x_unsuccess = []

                for i in range(len(y_input)):
                    if y_input[i] == 1:
                        x_success.append(x_input[i])
                    else:
                        x_unsuccess.append(x_input[i])
               
                # updating discriminator weights
                with tf.GradientTape() as tape:

                    gen_output_success = self.EncoderGenerator.call(x_success)
                    gen_output_unsuccess = self.EncoderGenerator.call(x_unsuccess)

                    # three outputs for different cases from discriminator
                    dis_output_success = self.Discriminator_Multiclass.call(gen_output_success)
                    dis_output_unsuccess = self.Discriminator_Multiclass.call(gen_output_unsuccess)
                    dis_output_dist = self.Discriminator_Multiclass.call(noise)

                    # Add extra losses created during this forward pass:
                    dis_loss = self.Discriminator_Multiclass.loss(dis_output_success, dis_output_unsuccess, dis_output_dist)
                    grads = tape.gradient(dis_loss, self.Discriminator_Multiclass.trainable_weights)
                    self.Discriminator_Multiclass.optimizer.apply_gradients(zip(grads, 
                                                                            self.Discriminator_Multiclass.trainable_weights))

                # updating generator weights           
                with tf.GradientTape() as tape:
                    gen_output_success = self.EncoderGenerator.call(x_success)
                    gen_output_unsuccess = self.EncoderGenerator.call(x_unsuccess)

                    # three outputs for different cases from discriminator
                    dis_output_success = self.Discriminator_Multiclass.call(gen_output_success)
                    dis_output_unsuccess = self.Discriminator_Multiclass.call(gen_output_unsuccess)
                    
                    # Add extra losses created during this forward pass:
                    gen_loss = self.EncoderGenerator.mc_loss(dis_output_success, dis_output_unsuccess)

                    grads = tape.gradient(gen_loss, self.EncoderGenerator.trainable_weights)
                    self.EncoderGenerator.optimizer.apply_gradients(zip(grads, self.EncoderGenerator.trainable_weights))


                # summing all losses
                avg_dis_loss += dis_loss.numpy()
                avg_aenc_loss += dec_loss
                avg_gen_loss += gen_loss.numpy()

            # averaging loss values over epoch
            avg_dis_loss /= (len(X) / batch_size)
            avg_aenc_loss /= (len(X) / batch_size)
            avg_gen_loss /= (len(X) / batch_size)

            print("[Epoch " + str(ep) + " of " + str(num_epochs) + "]")
            print("--------------------------------------------------")
            print("Average Discriminator Loss: " + str(avg_dis_loss))
            print("Average Autoencoder Loss: " + str(avg_aenc_loss))
            print("Average Generator Loss: " + str(avg_gen_loss))
            # print("Validation Accuracy: " + str(self.test(validation_set, validation_labels)))
            print("")

    def calc_validation(self, v_data, v_labels):
        '''
        Method to calculate the validation loss on good data and on bad data
        '''

        # randomly shuffling data
        random_order = np.random.permutation(len(v_data))
        v_data, v_labels = v_data[random_order], v_labels[random_order]

        # filtering out bad passes
        v_good, v_bad = [], []

        for i in range(len(v_labels)):
            if v_labels[i] == 1:
                v_good.append(v_data[i])
            else:
                v_bad.append(v_data[i])
        
        # making even number of data points per validation set
        min_length = min(len(v_bad), len(v_good))

        v_good = v_good[:min_length]
        v_bad = v_bad[:min_length]

        v_good = np.array(v_good)
        v_bad = np.array(v_bad)

        v_l_g = self.test_autoencoder(v_good)
        v_l_b = self.test_autoencoder(v_bad)

        return v_l_g, v_l_b

    def test_discriminator(self, test_X, noise):
        '''
        Function to test the discriminator 

        Keyword Args:
        test_X - data points to put through testing
        noise - noise generated to test the model
        '''

        correct_real = 0
        correct_fake = 0

        latent_input = self.EncoderGenerator.call(test_X, training=False)
        fake_preds = self.Discriminator.call(latent_input, training=False)
        real_preds = self.Discriminator.call(noise, training=False)

        for i in fake_preds:
            if i < 0.5:
                correct_real += 1
        
        print(str(correct_real) + " from fake")

        for i in real_preds:
            if i > 0.5:
                correct_fake += 1

        print(str(correct_fake) + " from real")
        return (correct_real + correct_fake) / (len(test_X) + len(noise))
    
    def test_generator(self, test_X):
        '''
        Function to test generator

        Keyword Args:
        test_X - data points to put through testing
        '''

        success = 0

        latent_input = self.EncoderGenerator.call(test_X, training=False)
        fake_preds = self.Discriminator.call(latent_input, training=False)

        for i in fake_preds:
            if i > 0.5:
                success += 1
        
        return success / len(test_X)
    
    def test_autoencoder(self, test_X):
        '''
        Function to calculate the loss of the autoencoder on test data points 

        Keyword Args:
        test_X - data points to put through testing
        '''

        latent_input = self.EncoderGenerator.call(test_X, training=False)
        reconstructed_X = self.Decoder.call(latent_input, training=False)

        mse_losses = []

        for i in range(len(latent_input)):
            mse = tf.reduce_sum(tf.pow(test_X[i] - reconstructed_X[i], 2))
            mse_losses.append(mse.numpy())

        return self.Decoder.loss(reconstructed_X, test_X).numpy()
    
    def visualize_latent(self, test_X, test_labels):
        '''
        Function to visualize the latent space projection of the test data
        '''

        latent_X = self.EncoderGenerator.call(test_X).numpy()
        embedded_space = convert_tsne(latent_X)

        x, y = zip(*embedded_space)

        color = ["red" if x == 1 else "blue" for x in test_labels]

        plt.scatter(x, y, c=color, s=25)
        red_patch = mpatches.Patch(color='red', label='Successful Passes')
        blue_patch = mpatches.Patch(color='blue', label='Unsuccessful Passes')

        plt.legend(handles=[red_patch, blue_patch], loc="upper left")
        plt.show()
    
    def visualize_loss(self, test_X, test_labels):
        '''
        Function to visualize the reconstructed loss on the held out test set
        '''

        reconstructed_X = self.Autoencoder(test_X)
        losses = mse(test_X, reconstructed_X).numpy()

        # calculating statistics about losses
        # print(sorted(losses))

        # visualizing losses
        color = ["red" if x == 1 else "blue" for x in test_labels]

        plt.scatter(range(len(losses)), losses, c=color, s=5)

        red_patch = mpatches.Patch(color='red', label='Successful Passes')
        blue_patch = mpatches.Patch(color='blue', label='Unsuccessful Passes')
        plt.legend(handles=[red_patch, blue_patch], loc="upper left")

        plt.ylabel('Loss')
        plt.xlabel('Example #')

        plt.show()
    
    def generate_examples(self, num_examples):
        '''
        Function to generate synthetic data examples by sampling from the imposed distribution

        Keyword Args:
        num_examples - the number of examples to generate

        Returns:
        a list of input examples of length num_examples
        '''

        noise_input = gen_noise(num_examples, self.latent_size)
        reconstructed_output = self.Decoder.call(noise_input)
        return reconstructed_output

    def train_criterion(self, X):
        '''
        Criterion for GAN training from toronto presentation
        
        Keyword Args:
        X - data input into the autoencoder to recreate the latent space distribution

        Returns:
        the training criterion C(G) = -log(4) + 2 * JSD(p, q), where p is the latent distribution
        and q is the distribution learned by the AAE model
        '''

        latent_dist = self.EncoderGenerator.call(X).numpy()
        noise_dist = gen_noise(np.shape(latent_dist)[0], np.shape(latent_dist)[1])

        # computing probabilities of real and fake -> not sure if this is what goes into criterion
        p_real = self.Discriminator(latent_dist)
        p_fake = self.Discriminator(noise_dist)
        c_g = -1 * math.log(4) + 2 * jensen_shannon_divergence(p_real, p_fake)
        return c_g
    
    def calculate_anomalies(self, losses):
        '''
        Function to calculate the anomalies from the input losses

        Returns:
        array of (-1 or 1) -> 1 meaning in distribution, -1 meaning out of distribution
        '''

        to_return = []

        for i in losses:
            if i >= self.threshold:

                to_return.append(-1)
            else:
                to_return.append(1)

        return to_return

    def detect_anomalies(self, train_data, train_oids, test_data, test_oids):
        '''
        Function to detect anomalies from train and test data

        Keyword Args:
        train_data - training data
        train_oids - overflight ids for training data
        test_data - testing data
        test_oids - overflight ids for testing data

        Returns:
        two dictionaries of oid -> anomaly (-1 or 1)
        '''

        reconstructed_train = self.Autoencoder(train_data)
        train_losses = mse(train_data, reconstructed_train).numpy()

        reconstructed_test = self.Autoencoder(test_data)
        test_losses = mse(test_data, reconstructed_test).numpy()     

        train_anomalies = self.calculate_anomalies(train_losses)
        test_anomalies = self.calculate_anomalies(test_losses)

        train_dict = {}
        test_dict = {}

        for i in range(len(train_oids)):
            train_dict[train_oids[i]] = train_anomalies[i]

        for i in range(len(test_oids)):
            test_dict[test_oids[i]] = test_anomalies[i]

        return train_dict, test_dict 
    
    def test_mc_discriminator(self, test_x, test_y):
        '''
        Function to test the multiclass discriminator
        '''

        s_acc = 0
        s_tot = 0
        u_acc = 0
        u_tot = 0

        enc_out = self.EncoderGenerator.call(test_x)
        dis_out = self.Discriminator_Multiclass.call(enc_out).numpy()

        for i in range(len(test_y)):
            max_ind = np.argmax(dis_out[i])
            if test_y[i] == 1:
                s_tot += 1
                if max_ind == 0:
                    s_acc += 1
            else:
                u_tot += 1
                if max_ind == 1:
                    u_acc += 1

        s_acc /= s_tot
        u_acc /= u_tot

        print("Accuracy on Successful Passes: " + str(s_acc))
        print("Accuracy on Unsuccessful Passes: " + str(u_acc))

    def save_model(self):
        '''
        Method to save model 
        '''

        # tf.saved_model.save(input_var, output_var)

        # trying pickling model
        # pickle.dump(self, open("saved_models/aae.p", "wb"))

        # Saver([self]).save("/saved_models/aae/ckpt")
        self.save_weights("saved_models/aae_weights.h5")


    def restore_model(self):
        '''
        Method to restore model
        '''
        self.load_weights("saved_models/aae_weights.h5")

def jensen_shannon_divergence(p, q):
    '''
    Function to compute the Jenson-Shannon divergence between two distributions p and q
    '''

    p = p / norm(p)
    q = q / norm(q)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def create_dataset(remove_categorical=False):
    '''
    Function to set up the processing of data before passing it through the model for training
    '''

    # df = pd.read_csv("../data/processed_data/full_normalized_ids_labelled_zscore.csv")
    # df = pd.read_csv("../data/processed_data/full_normalized_337_2450_ids_labelled_zscore.csv")
    # df = pd.read_csv("../data/processed_data/898-2450_ids_normalized_zscore.csv")
    df = pd.read_csv("../data/processed_data/337-2450_ids_normalized_zscore.csv")

    if remove_categorical:
        df.drop(columns=["orbiter_MEX", "orbiter_nan", "orbiter_MVN",	"orbiter_DTE",	"orbiter_TGO",	"orbiter_ODY",	"orbiter_MRO",	    
                         "TDS_dssId_64",	"TDS_dssId_15",	"TDS_dssId_45",	"TDS_dssId_14",	"TDS_dssId_43",	
                         "TDS_dssId_36",	'TDS_dssId_63',	"TDS_dssId_24", "TDS_dssId_34",	'TDS_dssId_0',	
                         "TDS_dssId_26",	"TDS_dssId_65",	"TDS_dssId_54",	"TDS_dssId_25",	"TDS_dssId_35",	
                         "TDS_dssId_55",	'TDS_dssId_50'], inplace=True)

    labels = np.array(df["GroundTruth"].tolist())
    ids = np.array(df["GDS__id"].tolist())

    df.drop(columns=["DylanLabel", "GDS__id", "MiguelLabel", "Unnamed: 0_y", "GDSLabel", 
                     "BinaryGDSLabel", "GroundTruth"], inplace=True)

    data = df.values
    features = df.keys().tolist()

    # randomly shuffling data
    random_order = np.random.permutation(len(data))
    data, labels, ids = data[random_order], labels[random_order], ids[random_order]

    # splitting data train (70%), validation (20%), test (10%)
    total = len(data)
    
    # randomly shuffling data
    train_data = data[:int(round(7 / 10 * total))]
    train_labels = labels[:int(round(7 / 10 * total))]
    train_ids = ids[:int(round(7 / 10 * total))]


    validation_data = data[int(round(7 / 10 * total)): int(round(9 / 10 * total))]
    validation_labels = labels[int(round(7 / 10 * total)): int(round(9 / 10 * total))]

    test_data = data[int(round(9 / 10 * total)):]
    test_labels = labels[int(round(9 / 10 * total)):]
    test_ids = ids[int(round(9 / 10 * total)):]

    # # dropping GDS__id from training and validation
    # train_data = np.array([data[1:] for data in train_data])
    # validation_data = np.array([data[1:] for data in validation_data])
    # test_data = np.array([data[1:] for data in test_data])

    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels, train_ids, test_ids

def gen_noise(batch_size, latent_size, labels=False):
    '''
    Function to generate noise from -1 to 1 from a Gaussian distribution

    Keywords:
    batch_size - the size of the output batch
    latent_size - the number of dimensiosn of each data point
    '''

    # generating noise from multiple class of Gaussians
    if labels:
        classes = np.random.randint(0, 2, size=batch_size)
        
        noise = []
        for c in classes:
            to_add = np.array([0, 0, 0])
            to_add[c] = 1
            
            # 3 different Gaussians for classes
            if c == 0:
                n = np.random.normal(loc=-0.33, scale=0.33, size=latent_size)
            elif c == 1:
                n = np.random.normal(loc=0.33, scale=0.33, size=latent_size)
            elif c == 2:
                n = np.random.normal(loc=0, scale=0.33, size=latent_size)

            noise.append(np.concatenate((to_add, n)))
        
        noise = np.array(noise)

    # generating noise from Gaussian
    else:
        noise = np.random.normal(loc=0, scale=0.5, size=(batch_size, latent_size))
    
    return noise
    
    # return np.random.uniform(-1, 1, size=(batch_size, latent_size))


def plot_validation(v_good, v_bad):
    '''
    Function to plot the results of evaluation on validation set for each epoch
    '''

    plt.plot(range(len(v_good)), v_good, linewidth=3)
    plt.plot(range(len(v_bad)), v_bad, linewidth=3)

    plt.xlabel("Epochs")
    plt.ylabel("Loss (mse)")

    plt.legend(["Good Pass", "Bad Pass"], loc='upper right')
    plt.show()

def main_only_good():
    '''
    Method to only train the Adversarial Au
    '''

    # hyperparameters
    latent_size = 12
    batch_size = 100
    rate = 0.3
    learning_rate = 0.01

    X, y, validation_set, validation_labels, test_set, test_labels, train_ids, test_ids = create_dataset(remove_categorical=True)
    pass_size = np.shape(X)[1]

    model = AAE_model(pass_size, latent_size, rate)

    # filtering out bad passes
    X_good, y_good = [], []
    X_bad, y_bad = [], []

    for i in range(len(y)):
        if y[i] == 1:
            X_good.append(X[i])
            y_good.append(y[i])
        else:
            X_bad.append(X[i])
            y_bad.append(y[i])
    
    X_good = np.array(X_good)
    y_good = np.array(y_good)
    X_bad = np.array(X_bad)
    y_bad = np.array(y_bad)

    # adding bad passes not used in training data to validation set
    validation_set = np.concatenate((validation_set, X_bad))
    validation_labels = np.concatenate((validation_labels, y_bad))
        
    model.train(X_good, y_good, validation_set=validation_set, validation_labels=validation_labels, 
                num_epochs=20, batch_size=50, latent_size=latent_size, save=True)

    # converting test set to good and bad passes
    test_good, test_good_l = [], []
    test_bad, test_bad_l, = [], []

    for i in range(len(test_labels)):
        if test_labels[i] == 1:
            test_good.append(test_set[i])
            test_good_l.append(test_labels[i])
        else:
            test_bad.append(test_set[i])
            test_bad_l.append(test_labels[i])

    t_g, t_b = model.calc_validation(test_set, test_labels)

    # displaying results
    print("Results on Good Dataset")
    print("---------------------------------------------")
    gen_acc_good = model.test_generator(test_good)
    print("Generator Test Accuracy: " + str(gen_acc_good))

    # aenc_loss_good = model.test_autoencoder(test_good)
    aenc_loss_good = [model.test_autoencoder([x]) for x in test_good]
    print("Autoencoder Reconstruction Test Loss: " + str(t_g))

    print("")

    print("Results on Bad Dataset")
    print("---------------------------------------------")
    gen_acc_bad = model.test_generator(test_bad)
    print("Generator Test Accuracy: " + str(gen_acc_bad))

    # aenc_loss_bad = model.test_autoencoder(test_bad)
    aenc_loss_bad = [model.test_autoencoder([x]) for x in test_bad]
    print("Autoencoder Reconstruction Test Loss: " + str(t_b))

    print(aenc_loss_good)
    print(aenc_loss_bad)

    plt.plot(aenc_loss_good)
    plt.show()

def main_multiclass():
    '''
    Function to train an AAE to a multiclass discriminator setting
    '''

    # hyperparameters
    latent_size = 12
    batch_size = 100
    rate = 0.3
    learning_rate = 0.01

    X, y, _, _, test_set, test_labels, train_ids, test_ids = create_dataset(remove_categorical=True)
    pass_size = np.shape(X)[1]

    model = AAE_model(pass_size, latent_size, rate)
    model.train_multiclass(X, y, batch_size=batch_size, latent_size=latent_size, save=True)
    model.test_mc_discriminator(test_set, test_labels)


def convert_tsne(latent_space):
    '''
    Function to visualize the latent space of the autoencoder in 2D
    '''

    tsne = TSNE(n_components=2)
    embedded = tsne.fit_transform(latent_space)
    return embedded

def main():
    # hyperparameters
    latent_size = 12
    batch_size = 100
    rate = 0.3
    learning_rate = 0.01

    X, y, _, _, test_set, test_labels, train_ids, test_ids = create_dataset(remove_categorical=True)
    pass_size = np.shape(X)[1]

    model = AAE_model(pass_size, latent_size, rate)
    model.train(X, y, num_epochs=1, batch_size=batch_size, latent_size=latent_size, save=True)
   
    # need to train model before restoring?
    # model.save_model()
    model.restore_model()

    # training results
    # gen_acc = model.test_generator(test_set)
    # print("Generator Test Accuracy: " + str(gen_acc))
    # aenc_loss = model.test_autoencoder(test_set)
    # print("Autoencoder Reconstruction Test Loss: " + str(aenc_loss))
    # print("Training criterion based on toronto paper C(g):" + str(model.train_criterion(X)))

    # model.visualize_latent(test_set, test_labels)
    # model.visualize_loss(test_set, test_labels)
    train_out, test_out = model.detect_anomalies(X, train_ids, test_set, test_ids)
    print(train_out)
    print(test_out)

if __name__ == "__main__":
    main()
    # train_models()
    # main_only_good()
    # main_multiclass()





