import tensorflow as tf
import numpy as np 
import pandas as pd
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
import math


class EncoderGenerator(Model):
    '''
    Class for a Generative Model used in the Adversarial Autoencoder
    '''

    def __init__(self, input_size, latent_size, rate, loss_weight, learning_rate):
        super().__init__()

        # hyperparameters
        self.latent_size = latent_size
        self.loss_weight = loss_weight

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # creating generator network and layers
        self.layer_1 = Dense(256, activation=tf.nn.leaky_relu, input_shape=(input_size,))
        self.dp_1 = Dropout(rate)

        # second dense layer with leaky_relu and dropout
        self.layer_2 = Dense(128, activation=tf.nn.leaky_relu)
        self.dp_2 = Dropout(rate)

        # third dense layer with leaky_relu and dropout
        self.layer_3 = Dense(64, activation=tf.nn.leaky_relu)
        self.dp_3 = Dropout(rate)

        # final layer
        self.out_layer = Dense(self.latent_size, activation=tf.nn.tanh)


    def call(self, X, training=True):
        '''
        Function to call the encoder/generator model to generate a latent space representation
        of the model to find distribution q(z|x)

        Keyword Args:
        X - the data points to convert into a latent space

        Returns:
        z - the latent representation of X
        '''

        # first layer
        output_1 = self.layer_1(X)
        if training:
            output_1 = self.dp_1(output_1)

        # passing into second layer
        output_2 = self.layer_2(output_1)
        if training:
            output_2 = self.dp_2(output_2)

        # passing into third layer
        output_3 = self.layer_3(output_2)
        if training:
            output_3 = self.dp_3(output_3)

        #final output
        z = self.out_layer(output_3)

        return z

    def loss(self, fake_data, ones): 
        '''
        Funciton to calculate the loss of the generator/encoder 

        Keyword Args:
        fake_data - the result of the discrminator on evaluating the generator data
        ones - a numpy array of ones - meaning discriminator was fooled

        Returns:
        - the loss of the generator/encoder
        '''

        loss_function = BinaryCrossentropy(name="generator_loss")
        g_loss = loss_function(fake_data, ones) * self.loss_weight
        return g_loss

    def train_custom(self, dis_output):
        '''
        Function to train the encoder/generator network and apply the optimizer

        Keyword Args:
        dis_output - the output of the discriminator on the generator's latent space representation

        Returns:
        - the loss of the decoder (binary cross entropy)
        '''

        with tf.GradientTape() as tape:
            loss = BinaryCrossentropy()
            loss = loss(tf.ones_like(dis_output), dis_output) * self.loss_weight
            grads = tape.gradient(loss, self.trainable_variables)
            print(grads, "enc")
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

class Decoder(Model):
    '''
    Class for a Decoder Model used in the Adversarial Autoencoder
    '''

    def __init__(self, input_size, latent_size, rate, loss_weight, learning_rate):
        super().__init__()

        # hyperparameters
        self.input_size = input_size
        self.loss_weight = loss_weight

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # creating generator network and layers
        self.layer_1 = Dense(64, activation=tf.nn.leaky_relu, input_shape=(latent_size,))
        self.dp_1 = Dropout(rate)

        # second dense layer with leaky_relu and dropout
        self.layer_2 = Dense(128, activation=tf.nn.leaky_relu)
        self.dp_2 = Dropout(rate)

        # third dense layer with leaky_relu and dropout
        self.layer_3 = Dense(256, activation=tf.nn.leaky_relu)
        self.dp_3 = Dropout(rate)

        # final layer -> back to normalized as [-1, 1]
        self.out_layer = Dense(self.input_size, activation=tf.nn.tanh)
    
    def call(self, z, training=True):
        '''
        Function to convert the latent space representation back to the original image
        to find distribution p(x|z)

        Keyword Args:
        z - the latent representation of X

        Returns:
        reconstructed_image - the reconstructed X
        '''

        # first layer
        output_1 = self.layer_1(z)
        if training:
            output_1 = self.dp_1(output_1)

        # passing into second layer
        output_2 = self.layer_2(output_1)
        if training:
            output_2 = self.dp_2(output_2)

        # passing into third layer
        output_3 = self.layer_3(output_2)
        if training:
            output_3 = self.dp_3(output_3)

        #final output
        reconstructed_image = self.out_layer(output_3)
        return reconstructed_image

    def loss(self, reconstructed_image, image): 
        '''
        Function to calculate the loss of the autoencoder model

        Keyword Args:
        image - the original image
        reconstructed_image - the result of applying the encoder and decoder to the image

        Returns:
        - the loss of the decoder (binary cross entropy)
        '''


        loss_function = MeanSquaredError(name="decoder_loss")
        dec_loss = loss_function(image, reconstructed_image) * self.loss_weight
        print(dec_loss)
        return dec_loss


    def train_custom(self, fake_image, image):
        '''
        Function to train the decoder model and apply the optimizer

        Keyword Args:
        image - the original image
        reconstructed_image - the result of applying the encoder and decoder to the image

        Returns:
        - the loss of the decoder (binary cross entropy)
        '''
        
        with tf.GradientTape() as tape:
            loss = self.loss(fake_image, image)
            grads = tape.gradient(loss, self.trainable_variables)
            print(grads, "decode")
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss
    
class Discrminator(Model):
    '''
    Class for a Discriminator Model used in the Adversarial Autoencoder
    '''

    def __init__(self, latent_size, rate, loss_weight, learning_rate):
        super().__init__()

        # hyperparameters
        self.loss_weight = loss_weight

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # creating generator network and layers
        self.layer_1 = Dense(64, activation=tf.nn.leaky_relu)
        self.dp_1 = Dropout(rate)

        # second dense layer with leaky_relu and dropout
        self.layer_2 = Dense(128, activation=tf.nn.leaky_relu)
        self.dp_2 = Dropout(rate)

        # third dense layer with leaky_relu and dropout
        self.layer_3 = Dense(256, activation=tf.nn.leaky_relu)
        self.dp_3 = Dropout(rate)

        # final layer shrinking to 0-1 output
        self.out_layer = Dense(1, activation=tf.nn.sigmoid)
    
    def call(self, z, training=True):
        '''
        Function to compare the latent space representation to the imposed input distribution

        Keyword Args:
        z - the data points sampled from the imposing distribution p(z) or from the latent space

        Returns:
        a scalar representing if the data is from p(z) or from the latent space 
        '''

        # first layer
        output_1 = self.layer_1(z)
        if training:
            output_1 = self.dp_1(output_1)

        # passing into second layer
        output_2 = self.layer_2(output_1)
        if training:
            output_2 = self.dp_2(output_2)

        # passing into third layer
        output_3 = self.layer_3(output_2)
        if training:
            output_3 = self.dp_3(output_3)

        #final output
        out = self.out_layer(output_3)
        return out

    def loss(self, fake_data, real_data): 
        '''
        Funciton to calculate the loss of the discrminator model

        Keyword Args:
        real_data - the output of the discrminator on actual data
        fake_data - the output of the discrminator on fake data

        Returns:
        - the loss of the discrmininator
        '''

        loss_function = BinaryCrossentropy(name="discrim_loss")
        fake_loss = loss_function(tf.zeros_like(fake_data), fake_data)
        
        # maximizing log(D(G(z))) instead of min of log(1-D(G(z))) for better convergence
        
        # real_loss = loss_function(tf.ones_like(real_data), real_data)
        real_loss = -1 * loss_function(tf.zeros_like(real_data), real_data)
        
        d_loss = (real_loss + fake_loss) * self.loss_weight
        return d_loss
    
    def train_custom(self, fake_latent, real_latent):
        '''
        Funciton to apply the optimizer to update network parameters

        Keyword Args:
        real_data - the output of the discrminator on actual data
        fake_data - the output of the discrminator on fake data

        Returns:
        - the loss of the discrmininator
        '''

        with tf.GradientTape() as tape:
            loss = self.loss(fake_latent, real_latent)
            grads = tape.gradient(loss, self.trainable_variables)
            print(grads, "discrim")
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss


