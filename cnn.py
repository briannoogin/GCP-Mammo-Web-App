import keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D, Flatten, MaxPooling2D
from keras import regularizers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

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
    return model

def train_model(model):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        'train',
        target_size=(500, 500),
        batch_size=32,
        class_mode='sparse',
        color_mode = 'grayscale')
    model.fit_generator(train_generator,steps_per_epoch=32,epochs = 10)
    return model

if __name__ == "__main__":
    model = define_model(3)
    train_model(model)