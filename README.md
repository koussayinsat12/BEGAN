# BEGAN-Pokemon-Image-Generation

This repository contains an implementation of a **Boundary Equilibrium Generative Adversarial Network (BEGAN)** for generating Pokémon images. The project uses TensorFlow and Keras for building the generator and discriminator networks.

---

## Repository Structure

```
.
├── config.py           # Configuration parameters
├── model.py            # BEGAN model and training loop
├── utils.py            # Generator, discriminator, and utility functions
├── train.py            # Training script
├── pokemon/            # Dataset directory (Pokémon images)
└── images/             # Generated images will be saved here
```

---

## Configuration (`config.py`)

```python
h = 128          # Latent vector size
img_size = 64    # Image height and width
epochs = 100     # Number of training epochs
channels = 3     # Number of image channels
BATCH_SIZE = 16  # Batch size
n = 64           # Base number of convolution filters
lr = 0.0001      # Learning rate
```

---

## Training (`train.py`)

1. Set the dataset directory:
```python
DATADIR = "pokemon"  # Path to your dataset
```

2. Preprocess images:
```python
- Resize images to 64x64
- Normalize pixel values to [-1, 1]
```

3. Create TensorFlow dataset and batch:
```python
dataset = tf.data.Dataset.from_tensor_slices(image_paths)
dataset = dataset.map(tf.io.read_file)
dataset = dataset.map(preprocess_image)
dataset = dataset.map(lambda x: (x-127.5)/127.5)
dataset = dataset.batch(BATCH_SIZE)
dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
```

4. Initialize networks:
```python
gen = generator()
disc = discriminator()
cbk = BEGANMonitor(num_img=8, latent_dim=h)
began = BEGAN(generator=gen, discriminator=disc, h=h, learning_rate=lr)
began.compile(loss_fn=loss_fn)
```

5. Train the BEGAN:
```python
began.fit(dataset, epochs=20, callbacks=[cbk])
```

---

## Monitoring Generated Images

- `BEGANMonitor` callback generates `num_img` images at the end of each epoch.
- Images are saved in the `/images` directory:
```
/images/generated_image_{i}_{epoch}.png
```

---

## How to Run

1. Make sure you have all dependencies installed:
```bash
pip install tensorflow matplotlib opencv-python tqdm
```

2. Place your Pokémon images in a folder named `pokemon`.

3. Run the training script:
```bash
python -m train
```

4. Check `/images` directory for generated images after each epoch.

---

## ⚡ Notes

- This implementation uses a **dynamic learning rate schedule** (`BEGANLRSchedule`) and the **kt parameter** for balancing generator/discriminator losses.  
- The discriminator is an autoencoder; the generator tries to fool it by producing realistic images.  
- Loss function is Mean Absolute Error (L1 loss) between inputs and reconstructed images.

---

## 🔗 References

- [BEGAN Paper](https://arxiv.org/abs/1703.10717) – Berthelot et al., 2017
