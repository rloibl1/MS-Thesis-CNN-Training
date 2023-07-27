import numpy as np
import os
import pickle
import timeit
from random import shuffle
from keras import utils

# Open dict listing
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


# Randomize exploration order
def randomize(dataset_len):
    indices = np.arange(dataset_len)
    shuffle(indices)
    return indices


# Create generator
def npyGen(dataset_path, batch_size=16, img_size=32, channels=3):

    dataset = load_obj(dataset_path)

    image_paths = list(dataset.keys())
    labels = list(dataset.values())

    # While loop so generator resets
    while 1:
        # Init return arrays
        X = np.empty((batch_size, img_size, img_size, channels), dtype=np.float32)
        y = np.empty(batch_size, dtype=np.int8)

        # Determine number of batches that can be pulled per epoch
        num_steps = np.int(len(image_paths) / batch_size)

        # Randomize Index order
        rand_indices = randomize(len(image_paths))

        # Reduce number of total number of indices to match batch size interval
        rand_indices = rand_indices[0: num_steps * batch_size]

        for i, index in enumerate(rand_indices):
            img = np.load(image_paths[index])
            X[i % batch_size, :, :, :] = img
            y[i % batch_size] = labels[index]

            if i % batch_size == batch_size - 1:
                # Yield batch of data
                yield X / 65536, utils.to_categorical(y, 2)
