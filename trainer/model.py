import keras
from keras.models import Sequential, Model
from keras.layers import Dense,Conv2D, Flatten, MaxPooling2D, Dropout, Activation, BatchNormalization, GlobalAveragePooling2D
from keras import regularizers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.nasnet import NASNetLarge
from keras.models import load_model
import tensorflow as tf
from keras.callbacks import TensorBoard
from tensorflow.python.lib.io import file_io

import argparse
import pandas 
import os
import sys 
import datetime

import numpy as np

def define_model(num_classes):
    model = Sequential()
    # one input layer
    # Convolution layers
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=(250,250,3)))

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

# define a NASNetLarge model from Google Brain
def define_pretrained_NASNet_model(num_classes):
    nas_net = NASNetLarge(input_shape = (250,250,3),weights = 'imagenet',classes=num_classes, include_top = False)
    # freeze the pre-trained layers
    for layer in nas_net.layers:
        layer.trainable=False
    # add additional layers 
    model = nas_net.output
    model = GlobalAveragePooling2D()(model)
    model = Dense(100,activation='relu')(model)
    model = BatchNormalization()(model)
    model = Dense(100,activation='relu')(model)
    model = BatchNormalization()(model)
    model = Dense(100,activation='relu')(model)
    model = BatchNormalization()(model)
    preds = Dense(num_classes,activation='softmax')(model)
    model = Model(inputs = nas_net.input, outputs = preds)
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    #model.summary()
    return model

# train from scratch 
def define_NASNet_model(num_classes):
    model = NASNetLarge(input_shape = (250,250,1),weights = None,classes=num_classes, include_top = True)
    # one output layer
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# trains model in batches 
def train_model(model,job_dir,mode,**args):
    # image data generator for data augmentation for training data
    train_datagen = ImageDataGenerator(
        rescale = 1./255, 
        rotation_range=30,
        horizontal_flip=True)

    # image data generator for validation data
    validation_datagen = ImageDataGenerator(
        rescale = 1./255, 
        )
    
    # only run if mode option is cloud, so that files are not being downloaded
    if(mode == 'CLOUD'):
        print("Loading train and validation files from GCS")
        # run google cloud command to copy images from GCS to local VM storage
        path = 'data'
        os.system('gsutil -m -q cp -r gs://cbis-ddsm-cnn/data/train %s' % path)
        os.system('gsutil -m -q cp -r gs://cbis-ddsm-cnn/data/test %s' % path)
        print("Loading train and validation files from GCS complete")

    # flow from directory
    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(250,250),
        color_mode='rgb',
        class_mode='sparse',
        seed = 7,
        batch_size = 16)

    validation_generator = validation_datagen.flow_from_directory(
        'data/test',
        target_size=(250,250),
        color_mode='rgb',
        class_mode='sparse',
        seed = 7,
        batch_size = 16,
        shuffle = True)

    # used to calculate number of steps per epoch
    num_train_examples = 1318
    num_val_examples = 378
    train_steps = num_train_examples / train_generator.batch_size
    val_steps = num_val_examples / validation_generator.batch_size

    # add tensorboard
    if mode == 'LOCAL':
        log_path = job_dir + '/logs/'
    elif mode == 'CLOUD':
        log_path = job_dir + 'logs/'
    else:
        raise ValueError("Incorrect mode argument")

    tensorboard = TensorBoard(log_dir = log_path,histogram_freq=0, write_images=True)
    model.fit_generator(train_generator,steps_per_epoch=train_steps,epochs = 100,callbacks=[tensorboard],validation_data = validation_generator,validation_steps=val_steps)

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

# evaluates model from test performance 
def evaluate_model(model,mode, **args):

    if(mode == 'CLOUD'):
        print("Loading test files from GCS")
        # run google cloud command to copy images from GCS to local VM storage
        path = 'data'
        os.system('gsutil -m -q cp -r gs://cbis-ddsm-cnn/data/test %s' % path)
        print("Loading test files from GCS complete")

    # image data generator for data augmentation
    test_datagen = ImageDataGenerator(
        rescale = 1./255, 
        )
    # flow from irectory
    test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(250,250),
        color_mode='rgb',
        class_mode='sparse',
        seed = 7,
        batch_size = 2,
        shuffle = False)
    num_examples = 378
    steps = num_examples/test_generator.batch_size
    result = model.evaluate_generator(test_generator,steps=steps)
    print("Test Error",result[0])
    print("Test Accuracy:", result[1])

if __name__ == "__main__":
    # set rng seed
    np.random.seed(0)
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
      )
    parser.add_argument(
      '--mode',
      help='Used to configure run settings for local or on cloud',
      required=True
    )
    parser.add_argument(
        '--train',
        help = 'Used to train model or load trained model from memory',
        required = True
    )
    args = parser.parse_args()
    arguments = args.__dict__
    
    if(arguments['mode'] == 'CLOUD'):
        print('hi')
        # make a folder in the VM so that files can be copied from GCS
        path = "data"
        try:  
            os.mkdir(path)
        except OSError:  
            print ("Creation of the directory %s failed" % path)
        else:  
            print ("Successfully created the directory %s " % path)

    train = arguments['train']
    # train phase
    if train == 'TRUE':
        # define model
        model = define_pretrained_NASNet_model(3)
        model = train_model(model,**arguments)
    # load model
    else:
        model_name = 'pretrained_nasnet.h5'
        print("Loading model from Google Cloud")	
        os.system('gsutil -q cp gs://cbis-ddsm-cnn/models/%s data/%s' % (model_name, model_name))
        print("Finished loading model from Google Cloud")	
        model = load_model('data/' + model_name)
        # evaluate model on test data
        evaluate_model(model,**arguments)