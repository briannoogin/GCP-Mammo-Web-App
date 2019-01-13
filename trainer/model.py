import keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D, Flatten, MaxPooling2D, Dropout, Activation
from keras import regularizers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from tensorflow.python.lib.io import file_io

import argparse
import pandas 
import os
import sys 
import datetime

import numpy as np

def define_model(num_classes):
    # set rng seed
    np.random.seed(0)
    model = Sequential()
    # one input layer
    # Convolution layers
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=(500,500,1)))

    model.add(Conv2D(64, (5, 5)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (5, 5)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Dense Layers
    model.add(Flatten())
    model.add(Dense(100,activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(100,activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(100,activation = 'relu'))
    model.add(Dropout(0.1))

    # one output layer
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def train_model(model,job_dir,**args):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

    # load files from Google Cloud Storage
    path = "data"
    # make a folder in the VM so that files can be copied from GCS
    try:  
        os.mkdir(path)
    except OSError:  
        print ("Creation of the directory %s failed" % path)
    else:  
        print ("Successfully created the directory %s " % path)

    # run google cloud command to copy images from GCS to local VM storage
    os.system('gsutil -m cp -r gs://cbis-ddsm-cnn/data/train %s' % path)

    # flow from VM directory
    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(500,500),
        color_mode='grayscale',
        class_mode='sparse',
        seed = 7,
        batch_size = 32)

    # used to calculate number of steps per epoch
    num_examples = 1318
    steps = num_examples/train_generator.batch_size
    model.fit_generator(train_generator,steps_per_epoch=steps,epochs = 10)

    # save model locally
    currentDT = str(datetime.datetime.now())
    model_name = "model_%s.h5" % currentDT
    model.save(model_name)
    gc_model_name = "models/" + model_name
    # save the model file to GCloud
    with file_io.FileIO(model_name, mode='rb') as input_f:
        with file_io.FileIO(job_dir + gc_model_name, mode='w+') as output_f:
            output_f.write(input_f.read())
    #saved_model_path = tf.contrib.saved_model.save_keras_model(model, "./saved_models")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__
    model = define_model(3)
    train_model(model,**arguments)