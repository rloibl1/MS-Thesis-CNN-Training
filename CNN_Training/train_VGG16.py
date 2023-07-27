from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, GlobalMaxPooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.models import Sequential, Model, model_from_json
from keras.utils import layer_utils
from keras.engine.topology import get_source_inputs
from keras import applications
from keras import backend as K
from keras.optimizers import rmsprop, SGD
from datagen import npyGen
from make_parallel import make_parallel
import os

batch_size = 64
num_classes = 2
img_size = 256
channels = 6
epochs = 100
num_training = 5675
num_validation = 3783
data_augmentation = False
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# Model
# Define Input Shape
input_shape = (img_size, img_size, channels)

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=input_shape))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(.5))
model.add(Dense(1024, activation="relu"))
model.add(Dropout(.5))
model.add(Dense(2, activation="softmax"))

print(model.summary())

# Parallel Computation
# model = make_parallel(model, gpu_count=2, batch_size=batch_size)

# initiate RMSprop optimizer
# opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
opt = keras.optimizers.SGD(lr=.001, momentum=.9)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

train_path = '/home/afit/Desktop/Dataset/train_dict'
val_path = '/home/afit/Desktop/Dataset/val_dict'

train_datagen = npyGen(train_path, batch_size=batch_size, img_size=img_size, channels=channels)
val_datagen = npyGen(val_path, batch_size=batch_size, img_size=img_size, channels=channels)

model.fit_generator(train_datagen,
                    epochs=epochs,
                    steps_per_epoch=num_training // batch_size,
                    validation_data=val_datagen,
                    validation_steps=num_validation // batch_size)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
