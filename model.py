import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
import numpy as np
import keras
import cv2
from config import *
import os
class BEGAN(Model):
    def __init__(self, generator, discriminator, h, learning_rate, lamda_k=0.001, gamma=0.5):
        super(BEGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.h = h
        self.kt = tf.Variable(0.0,trainable=False)
        self.lr = BEGANLRSchedule(learning_rate)
        self.lamda_k = lamda_k
        self.gamma = gamma

    def compile(self, loss_fn):
        super(BEGAN, self).compile()
        self.loss_fn = loss_fn
        self.gen_opt = optimizers.Adam(learning_rate=self.lr)
        self.disc_opt = optimizers.Adam(learning_rate=self.lr)

    @tf.function
    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        batch_size = tf.shape(real_images)[0]
        random_latent_vector = tf.random.uniform(shape=[batch_size, self.h], minval=-1., maxval=1.)
        # Update BEGAN
        with tf.GradientTape(persistent=True) as tape:
            fake_images = self.generator(random_latent_vector, training=True)
            rec_real_images = self.discriminator(real_images, training=True)
            rec_fake_images = self.discriminator(fake_images, training=True)
            
            loss_real = self.loss_fn(real_images, rec_real_images)
            loss_fake = self.loss_fn(fake_images, rec_fake_images)
            D_loss = loss_real - self.kt * loss_fake
            G_loss = loss_fake
            Convergence = loss_real + tf.math.abs(self.gamma * loss_real - loss_fake)

        grad_d = tape.gradient(D_loss, self.discriminator.trainable_variables)
        grad_g = tape.gradient(G_loss, self.generator.trainable_variables)

        self.disc_opt.apply_gradients(zip(grad_d, self.discriminator.trainable_variables))
        self.gen_opt.apply_gradients(zip(grad_g, self.generator.trainable_variables))


        self.kt.assign_add(self.lamda_k * (self.gamma * loss_real - loss_fake))

        return {'D_loss': D_loss, 'G_loss': G_loss, 'Convergence': Convergence}


class BEGANLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, initial_learning_rate):
    super(BEGANLRSchedule,self).__init__()
    self.initial_learning_rate = initial_learning_rate

  def __call__(self, step):
     self.initial_learning_rate=np.maximum(1e-6,self.initial_learning_rate/2)
     return self.initial_learning_rate

class BEGANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img, latent_dim):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images = (generated_images * 127.5) + 127.5

        for i in range(self.num_img):
            img = generated_images[i].numpy().astype(int)  # Convert to uint8
            if  not os.path.exists("/images"):    
                os.makedirs("/images")
            image_path = f"/images/generated_image_{i}_{epoch}.png"
            cv2.imwrite(image_path, img)
