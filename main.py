import os
import numpy as np
import cv2 as cv
import caer
import gc
import canaro
import matplotlib.pyplot as plt
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import LearningRateScheduler

# dataset source: https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset

IMG_SIZE = (80,80)
channels = 1
char_path = r'simpsons_dataset'

# Creating a character dictionary, sorting it in descending order
char_dict = {}
for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path,char)))

# Sort in descending order
char_dict = caer.sort_dict(char_dict, descending=True)

#  Getting the first 10 categories with the most number of images
characters = []
count = 0
for i in char_dict:
    characters.append(i[0])
    count += 1
    if count >= 10:
        break


# Create the training data
train = caer.preprocess_from_dir(char_path, characters, channels=channels, IMG_SIZE=IMG_SIZE, isShuffle=True)

# Separating the array and corresponding labels
featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)

# Normalize the featureSet ==> (0,1)
featureSet = caer.normalize(featureSet)

# Converting numerical labels to binary class vectors
labels = to_categorical(labels, len(characters))

# Creating train and validation data
x_train, x_val, y_train, y_val = caer.train_test_split(featureSet, labels, val_ratio=.2)

# Deleting variables to save memory
del train
del featureSet
del labels
gc.collect()

# Useful variables when training
BATCH_SIZE = 32
EPOCHS = 10

# Image data generator (introduces randomness in network ==> better accuracy)
datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

# Create our model (returns the compiled model)
model = canaro.models.createSimpsonsModel(IMG_SIZE=IMG_SIZE, channels=channels, output_dim=len(characters),
                                         loss='binary_crossentropy', decay=1e-7, learning_rate=0.001, momentum=0.9,
                                         nesterov=True)

model.summary()




