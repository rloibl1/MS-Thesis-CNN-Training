from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, GlobalMaxPooling2D, concatenate
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.models import Sequential, Model, model_from_json
from keras.callbacks import History, ModelCheckpoint
from keras.utils import layer_utils
from keras.engine.topology import get_source_inputs
from keras import applications
from keras import backend as K
from keras.optimizers import rmsprop, SGD, adam
from datagen import npyGen
from make_parallel import make_parallel
import os
import plotter as plt
import simplejson


def squeezeNet_Train(datasetPath, fileName, batch_size, epochs, input_shape, num_classes,
                     num_training, num_validation, num_test):

    # Saves the model with the highest validation accuracy during training
    class saveBestModel(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.current_val_acc = 0.0
            return

        def on_epoch_end(self, epoch, logs=None):
            new_val_acc = logs.get('val_acc')
            if new_val_acc > self.current_val_acc:
                print('New best model with', new_val_acc, 'val accuracy at epoch', epoch)
                self.current_val_acc = new_val_acc

                # Save Model
                # serialize model to JSON
                model_json = model.to_json()
                with open(os.path.join(save_dir, fileName + '_final_model.json'), 'w') as json_file:
                    json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))

                # serialize weights to HDF5
                model.save_weights(os.path.join(save_dir, fileName + '_final_model.h5'))
                print("Saved model to disk")

            return

    sq1x1 = "squeeze1x1"
    exp1x1 = "expand1x1"
    exp3x3 = "expand3x3"
    relu = "relu_"

    # Initialization Stuff
    save_dir = os.path.join(os.getcwd(), 'saved_models', fileName)
    print('Training starting for model', fileName)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    history = History()  # Enable history callback
    bestModel = saveBestModel() # Save best validation model

    # Dataset Location
    train_path = datasetPath + '/train_dict'
    val_path = datasetPath + '/val_dict'
    test_path = datasetPath + '/test_dict'

    # Model

    # Modular function for Fire Node
    def fire_module(x, fire_id, squeeze=16, expand=64):
        s_id = 'fire' + str(fire_id) + '/'

        channel_axis = 3

        x = Conv2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
        x = Activation('relu', name=s_id + relu + sq1x1)(x)

        left = Conv2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
        left = Activation('relu', name=s_id + relu + exp1x1)(left)

        right = Conv2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
        right = Activation('relu', name=s_id + relu + exp3x3)(right)

        x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
        return x

    # Define Input Shape
    input_img = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(input_img)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)

    x = Dropout(0.5, name='drop9')(x)

    x = Conv2D(num_classes, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax', name='loss')(x)

    model = Model(input_img, x, name='squeezenet')

    # print(model.summary())

    # Parallel Computation
    # model = make_parallel(model, gpu_count=2, batch_size=batch_size)

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    # opt = keras.optimizers.SGD(lr=.075, momentum=.90)

    # Compile Model
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    train_datagen = npyGen(train_path, batch_size=batch_size, img_size=input_shape[0], channels=input_shape[2])
    val_datagen = npyGen(val_path, batch_size=batch_size, img_size=input_shape[0], channels=input_shape[2])
    test_datagen = npyGen(test_path, batch_size=1, img_size=input_shape[0], channels=input_shape[2])

    model.fit_generator(train_datagen,
                        epochs=epochs,
                        steps_per_epoch=num_training // batch_size,
                        validation_data=val_datagen,
                        validation_steps=num_validation // batch_size,
                        callbacks= [history, bestModel],
                        verbose=0)

    # Create Accuracy and Loss Graphs
    plt.plot(history, 'plots/' + fileName + '_squeezeNet')

    # load json and create model
    json_file = open(os.path.join(save_dir, fileName + '_final_model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(os.path.join(save_dir, fileName + '_final_model.h5'))
    print("Loaded model from disk")

    # evaluate loaded model on test data
    # Define X_test & Y_test data first
    loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    score = model.evaluate_generator(test_datagen, steps=num_test)
    print("Test %s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

    return history
