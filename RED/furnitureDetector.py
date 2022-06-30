# IMPORTS
import numpy as np
import pandas as pd
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras import models
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, GlobalAveragePooling2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
import os
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
# from google.colab.patches import cv2_imshow


# VARIABLES GLOABLES
img_width, img_height = 224, 224
batchsize = 10
NUM_EPOCHS = 5

# DIRECTORIOS
for dirname, _, filenames in os.walk('datasets'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# DIRECTORIOES DE IMAGENES DE ENTRENAMIENTO E IMAGENES DE VALIDIACION
train_dir = os.path.abspath('datasets/train')
val_dir = os.path.abspath('datasets/valid')

# DIRECTORIOS DE ENTRENAMIENTO DE  CADA CLASE

train_bed_dir = os.path.join(train_dir, 'bed')
train_chair_dir = os.path.join(train_dir, 'chair')
train_sofa_dir = os.path.join(train_dir, 'sofa')
train_swivelchair_dir = os.path.join(train_dir, 'swivelchair')
train_table_dir = os.path.join(train_dir, 'table')

# DIRECTORIOS DE VALIDACION DE CADA CLASE

val_bed_dir = os.path.join(val_dir, 'bed')
val_chair_dir = os.path.join(val_dir, 'chair')
val_sofa_dir = os.path.join(val_dir, 'sofa')
val_swivelchair_dir = os.path.join(val_dir, 'swivelchair')
val_table_dir = os.path.join(val_dir, 'table')

# CANTIDAD DE IMAGENES DE CADA DIRECTORIO

num_bed_train = len(os.listdir(train_bed_dir))
num_chair_train = len(os.listdir(train_chair_dir))
num_sofa_train = len(os.listdir(train_sofa_dir))
num_swivelchair_train = len(os.listdir(train_swivelchair_dir))
num_table_train = len(os.listdir(train_table_dir))

num_bed_val = len(os.listdir(val_bed_dir))
num_chair_val = len(os.listdir(val_chair_dir))
num_sofa_val = len(os.listdir(val_sofa_dir))
num_swivelchair_val = len(os.listdir(val_swivelchair_dir))
num_table_val = len(os.listdir(val_table_dir))

num_train_images = num_bed_train + num_chair_train + \
    num_sofa_train + num_swivelchair_train + num_table_train
num_val_images = num_bed_val + num_chair_val + \
    num_sofa_val + num_swivelchair_val + num_table_val
print(num_train_images, num_val_images)


# data generator for train dataset
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True
)

# data generator for validation  dataset
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True
)


# apply train data generator on train data
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(img_height, img_width), batch_size=batchsize)

# apply val data generator on validation data
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(img_height, img_width), batch_size=batchsize)


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()



base_model = VGG16(weights='imagenet',
                   include_top=False,
                   input_shape = (img_height, img_width,3)
                   )


len(base_model.layers)


def build_model(base_model, dropout, fc_layers, num_classes):
  for each_layer in base_model.layers:
    each_layer.trainable = True
  
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Flatten()(x)

  # Fine-tune from this layer onwards
  fine_tune_at = 100

  ## freeze the bafore layers of fine tune at number in base model
  for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False

  for fc in fc_layers:
    x = Dense(units=fc, activation='relu')(x)
    x = Dropout(dropout)(x)

  ## output layer with softmax activation function
  predictions = Dense(num_classes, activation='softmax')(x)

  final_model = Model(inputs = base_model.input, outputs = predictions)
  return final_model


class_list = ['bed', 'chair', 'sofa', 'swivelchair', 'table']
## units for fully connected layers 
FC_LAYERS = [1024, 1024]
dropout = 0.3

model = build_model(base_model, dropout, FC_LAYERS, len(class_list))


## compilation
adam = Adam(learning_rate=0.00001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


model.summary()


# ## train model
# history = model.fit(train_generator, 
#                     epochs=NUM_EPOCHS, 
#                     steps_per_epoch= num_train_images // batchsize, 
#                     validation_data=val_generator,
#                     validation_steps = num_val_images // batchsize)



# model.save_weights('./model')

# Restore the weights
model.load_weights('./model')


# test with the generated 3D model of the table

img_path = os.path.abspath('TEST3.jpg')
img = image.image_utils.load_img(img_path, target_size=(img_height, img_width))
img_tensor = image.image_utils.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis =0)
img_tensor = preprocess_input(img_tensor)

featuremap = model.predict(img_tensor)
index = np.argmax(featuremap)
print(class_list[index])
plt.imshow(img)


