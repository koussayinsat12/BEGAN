import tensorflow as tf
from utils import *
from model import*
import pathlib
from tensorflow.keras import preprocessing
import warnings
warnings.filterwarnings("ignore")
import cv2
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
DATADIR="pokemon" #your dataset
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [img_size, img_size]) 
    return image
image_paths = [os.path.join(DATADIR, img) for img in os.listdir(DATADIR)]
dataset = tf.data.Dataset.from_tensor_slices(image_paths)
dataset = dataset.map(tf.io.read_file)
dataset = dataset.map(preprocess_image)
dataset=dataset.map(lambda x:(x-127.5)/127.5)
dataset = dataset.batch(BATCH_SIZE)
dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
gen=generator()
disc=discriminator()
cbk=BEGANMonitor(num_img=8,latent_dim=h)
began=BEGAN(generator=gen,discriminator=disc,h=h,learning_rate=lr)
began.compile(loss_fn=loss_fn)
began.fit(dataset,epochs=20,callbacks=[cbk])

