import keras
from keras.models import Sequential, Model
from keras.layers import Dense,Conv2D, Flatten, MaxPooling2D, Dropout, Activation, BatchNormalization, GlobalAveragePooling2D, Input,Concatenate
from keras import regularizers
from keras import optimizers
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.nasnet import NASNetLarge
from keras.applications.xception import Xception
from keras.applications.inception_v3  import InceptionV3
from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import model_from_config

import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import builder as saved_model_builder
#from tensorflow_serving.session_bundle import exporter

import numpy as np
import argparse
import pandas 
import os
import sys 
import datetime


# defines an InceptionNet model
def define_pretrained_InceptionNet_model(num_classes):
    input = Input(shape = (1,250,250))
    concat_input = Concatenate(axis = 1)([input,input,input])
    inception_net = InceptionV3(input_tensor = concat_input,weights = 'imagenet',classes=num_classes, include_top = False)
    layers = inception_net.layers
    # freeze the first 150 pre-trained layers
    for layer in layers[0:150]:
        layer.trainable=False
    dropout = .50
    #add additional layers 
    model = inception_net.output
    model = GlobalAveragePooling2D()(model)
    model = Dense(100,kernel_regularizer=regularizers.l2(0.1))(model)
    model = BatchNormalization()(model)
    model = Activation(activation='relu')(model)
    model = Dropout(dropout,seed = 7)(model)

    model = Dense(100,kernel_regularizer=regularizers.l2(0.1))(model)
    model = BatchNormalization()(model)
    model = Activation(activation='relu')(model)
    model = Dropout(dropout,seed = 7)(model)

    model = Dense(100,kernel_regularizer=regularizers.l2(0.1))(model)
    model = BatchNormalization()(model)
    model = Activation(activation='relu')(model)
    model = Dropout(dropout, seed = 7)(model)

    preds = Dense(num_classes,activation='softmax')(model)
    model = Model(inputs = input, outputs = preds)
    sgd = optimizers.SGD(lr=0.0025, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #model.summary()
    return model

# trains model in batches 
def train_model(model,job_dir,mode,model_name,**args):
    # image data generator for data augmentation for training data
    # may add more data augmentation to reduce overfitting
    train_datagen = ImageDataGenerator(
        rescale = 1./255, 
        rotation_range=30,
        horizontal_flip=True,
        width_shift_range=.2,
        height_shift_range=.2,
        shear_range=0.2,
        zoom_range=0.2,)

    # image data generator for validation data
    validation_datagen = ImageDataGenerator(
        rescale = 1./255, 
        )
    
    # only run if mode option is cloud, so that files are not being downloaded
    if(mode == 'CLOUD'):
        print("Loading train and validation files from GCS")
        # run google cloud command to copy images from GCS to local VM storage
        path = 'data'
        os.system('gsutil -m -q cp -r gs://cbis-ddsm-cnn/data/combined_train %s' % path)
        os.system('gsutil -m -q cp -r gs://cbis-ddsm-cnn/data/combined_test %s' % path)
        print("Loading train and validation files from GCS complete")

    # flow from directory
    train_generator = train_datagen.flow_from_directory(
        'data/combined_train',
        target_size=(250,250),
        color_mode='grayscale',
        class_mode='sparse',
        seed = 7,
        batch_size = 16)

    validation_generator = validation_datagen.flow_from_directory(
        'data/combined_test',
        target_size=(250,250),
        color_mode='grayscale',
        class_mode='sparse',
        seed = 7,
        batch_size = 16,
        shuffle = True)

    # used to calculate number of steps per epoch
    num_train_examples = 2864
    num_val_examples = 704
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
    checkpoint = ModelCheckpoint(model_name, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    model.fit_generator(train_generator,steps_per_epoch=train_steps,epochs = 100,callbacks=[tensorboard,checkpoint],validation_data = validation_generator,validation_steps=val_steps)
    
    #model.save(model_name)
    gc_model_name = "models/" + model_name
    # save the model file to GCloud
    with file_io.FileIO(model_name, mode='rb') as input_f:
        with file_io.FileIO(job_dir + gc_model_name, mode='w+') as output_f:
            output_f.write(input_f.read())
    return model

# evaluates model from test performance 
def evaluate_model(model,mode, **args):

    if(mode == 'CLOUD'):
        print("Loading test files from GCS")
        # run google cloud command to copy images from GCS to local VM storage
        path = 'data'
        os.system('gsutil -m -q cp -r gs://cbis-ddsm-cnn/data/combined_test %s' % path)
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

# exports a keras model into a tensorflow model so it can be hosted on Google ML REST API
def export_model(model,job_dir,model_name):
    # define model input and output
    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={"input": model.input},
        outputs={"output": model.output})
    builder = tf.saved_model.builder.SavedModelBuilder(job_dir + "/models/" + model_name)
    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={
            "predict": signature, "serving_default": signature})
    # save tensorflow model
    builder.save()

if __name__ == "__main__":

    # used to make formatting image in Google Cloud easier 
    K.set_image_data_format('channels_first')

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
    parser.add_argument(
        '--model_name',
        help = 'Name is used to save model to file',
        required = True
    )
    args = parser.parse_args()
    arguments = args.__dict__
    
    if(arguments['mode'] == 'CLOUD'):
        # make a folder in the VM so that files can be copied from GCS
        path = "data"
        try:  
            os.mkdir(path)
        except OSError:  
            print ("Creation of the directory %s failed" % path)
        else:  
            print ("Successfully created the directory %s " % path)

    train = arguments['train']
    export = True
    # train phase
    if train == 'TRUE':
        # define model
        model = define_pretrained_InceptionNet_model(3)
        model = train_model(model,**arguments)

    # load model
    else:
        model_name = arguments['model_name']
        print("Loading model from Google Cloud")	
        os.system('gsutil -q cp gs://cbis-ddsm-cnn/models/%s data/%s' % (model_name, model_name))
        print("Finished loading model from Google Cloud")	
        model = load_model('data/' + model_name)
        # evaluate model on test data
        evaluate_model(model,**arguments)
       
    if export == True:
        export_model(model,arguments['job_dir'],arguments['model_name'])
