from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential, Model, model_from_json
from keras import applications
from keras.optimizers import rmsprop, SGD
from datagen import npyGen
import os

batch_size = 32
num_classes = 2
img_size = 256
channels = 6
epochs = 100
num_training = 0
num_validation = 0
data_augmentation = False
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# Model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(img_size, img_size, channels)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

train_path = '/home/afit/Desktop/dataset_test/train_dict'
val_path = '/home/afit/Desktop/dataset_test/val_dict'

train_datagen = npyGen(train_path, batch_size=batch_size, img_size=img_size, channels=channels)
val_datagen = npyGen(val_path, batch_size=batch_size, img_size=img_size, channels=channels)

model.fit_generator(train_datagen,
                    epochs=epochs,
                    steps_per_epoch=num_training / batch_size,
                    validation_data=val_datagen,
                    validation_steps=num_validation / batch_size)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
