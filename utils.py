from tensorflow.keras import layers,Model,activations
import tensorflow as tf
import numpy as np
from config import *


def dec_conv2d_block(x,n,kernel_size=(3,3),strides=(1,1),padding='same'):
    x=layers.Conv2D(n,kernel_size=kernel_size,strides=strides,padding=padding)(x)
    x=layers.BatchNormalization()(x)
    x=activations.elu(x)
    x=layers.Conv2D(n,kernel_size=kernel_size,strides=strides,padding=padding)(x)
    x=layers.BatchNormalization()(x)
    x=activations.elu(x)
    return x
def enc_conv2d_block(x,n,kernel_size=(3,3),strides=(1,1),padding='same',downsampling=True):
    x=layers.Conv2D(n,kernel_size=kernel_size,strides=strides,padding=padding)(x)
    x=layers.BatchNormalization()(x)
    x=activations.elu(x)
    if downsampling:
        strides=(2,2)
    x=layers.Conv2D(n,kernel_size=kernel_size,strides=strides,padding=padding)(x)
    x=layers.BatchNormalization()(x)
    x=activations.elu(x)
    return x



def generator():
    input=layers.Input(shape=(h,))
    x=layers.Dense(8*8*n)(input)
    x=layers.Reshape((8,8,n))(x)
    x=activations.elu(x)
    x=dec_conv2d_block(x,n)
    x=layers.UpSampling2D(size=(2,2))(x)
    x=dec_conv2d_block(x,n)
    x=layers.UpSampling2D(size=(2,2))(x)
    x=dec_conv2d_block(x,n)
    x=layers.UpSampling2D(size=(2,2))(x)
    x=dec_conv2d_block(x,n)
    x=layers.Conv2D(channels,(3,3),strides=(1,1),padding="same",activation="tanh")(x)
    return Model(input,x,name="generator") 


def discriminator():

    ##############
    #Encoding part
    ##############
    input=layers.Input(shape=(img_size,img_size,channels))
    x=layers.Conv2D(n,(3,3),strides=(1,1),padding="same",activation="elu")(input)
    x=enc_conv2d_block(x,n)
    x=enc_conv2d_block(x,2*n)
    x=enc_conv2d_block(x,3*n)
    x=enc_conv2d_block(x,4*n,downsampling=False)

    #Flatten part
    x=layers.Reshape((np.prod(x.get_shape().as_list()[1:]),))(x)
    x=layers.Dense(h,)(x)

    ##############
    #Decoding part
    ##############
    x=layers.Dense(8*8*n)(x)
    x=layers.Reshape((8,8,n))(x)
    x=activations.elu(x)
    x=dec_conv2d_block(x,n)
    x=layers.UpSampling2D(size=(2,2))(x)
    x=dec_conv2d_block(x,n)
    x=layers.UpSampling2D(size=(2,2))(x)
    x=dec_conv2d_block(x,n)
    x=layers.UpSampling2D(size=(2,2))(x)
    x=dec_conv2d_block(x,n)
    x=layers.Conv2D(channels,(3,3),strides=(1,1),padding="same",activation="tanh")(x)
    return Model(input,x,name="discriminator")




def loss_fn(input,output):
    return tf.math.reduce_mean(tf.math.abs(output-input))

