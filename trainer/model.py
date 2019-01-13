import keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D, Flatten, MaxPooling2D
from keras import regularizers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from tensorflow.python.lib.io import file_io
import json
import _pickle as pickle
from google.cloud import storage

import argparse
import pandas 

import multiprocessing.pool
from functools import partial
from keras.preprocessing.image import Iterator
import warnings
import numpy as np
import keras.backend as K
import os
import sys 
def define_model(num_classes):
    model = Sequential()
    # one input layer
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=(500,500,1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
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
    os.system('gsutil -m cp -r gs://cbis-ddsm-cnn/data/flow %s' % path)
    #train_generator = train_datagen.flow_from_dataframe(img_df,job_dir + '/data/train',x_col='img_file',y_col='class_label',has_ext=False)
    train_generator = train_datagen.flow_from_directory(
        'data/flow',
        target_size=(500,500),
        color_mode='grayscale',
        class_mode='sparse',
        seed = 7)

    num_examples = 1318
    steps = num_examples/train_generator.batch_size
    model.fit_generator(train_generator,steps_per_epoch=steps,epochs = 10)
    # save model locally
    model.save('model.h5')
    # save the model file to GCloud
    with file_io.FileIO('model.h5', mode='rb') as input_f:
        with file_io.FileIO(job_dir + '/models/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())
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
    #load_data()
    #load_json()