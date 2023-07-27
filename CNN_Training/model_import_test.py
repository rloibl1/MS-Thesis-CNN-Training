# Code to import VGG19 and add fully connected layer retrieved from:
# https://medium.com/towards-data-science/transfer-learning-using-keras-d804b2e04ef8
import plotter as plt
import simplejson
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.models import Sequential, Model, model_from_json
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import rmsprop, SGD
from keras.callbacks import History
from keras import backend as K

def runCNN(datasetPath, fileName):
    # Creates a Data Generator From Specified Directory
    def createDataGenerator(dir, augment=False):
        if augment:
            datagen = ImageDataGenerator(
                rescale=1. / 255,
                horizontal_flip=True,
                fill_mode="nearest",
                zoom_range=0.3,
                width_shift_range=0.3,
                height_shift_range=0.3,
                rotation_range=30)
        else:
            datagen = ImageDataGenerator()

        generator = datagen.flow_from_directory(
            dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')

        return generator

    # Initialization Stuff
    history = History()  # Enable history callback
    img_width, img_height = 256, 256

    # Dataset Location
    train_data_dir = datasetPath + '/train'
    val_data_dir = datasetPath + '/val'
    nb_train_samples = 699
    nb_val_samples = 308

    # Epochs & Batch
    epochs = 100
    batch_size = 64

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    # Load Pre-Training & Pre-Validation Images
    train_generator = createDataGenerator(train_data_dir, augment=True)
    val_generator = createDataGenerator(val_data_dir, augment=True)

    # Import pre-trained VGG19 model
    model = applications.VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

    # Add a new fully connected layer
    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(2, activation="softmax")(x)

    # Final Model
    model_final = Model(input=model.input, output=predictions)

    # Print Model Summary
    # Feature extraction layers
    print(model.summary())
    # Complete Network
    print(model_final.summary())

    # Freeze some layers of the network (lock all feature extraction layers)
    for layer in model_final.layers[:19]:
        layer.trainable = False

    # Define Optimizer
    opt = SGD(lr=0.0001, momentum=0.9)

    # Compile Model
    model_final.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # Re-Train Model for New Classes
    model_final.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        callbacks=[history],
        validation_data=val_generator,
        validation_steps=nb_val_samples // batch_size)

    # Create Accuracy and Loss Graphs
    plt.plot(history, 'plots/' + fileName + '_xfer_model')

    # Save Model
    # serialize model to JSON
    model_json = model_final.to_json()
    with open('models/' + fileName + '_xfer_model.json', 'w') as json_file:
        json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))

    # serialize weights to HDF5
    model_final.save_weights('models/' + fileName + '_xfer_model.h5')
    print("Saved model to disk")

    # load json and create model
    json_file = open('models/' + fileName + '_xfer_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights('models/' + fileName + '_xfer_model.h5')
    print("Loaded model from disk")

    # evaluate loaded model on test data
    # Define X_test & Y_test data first
    loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    score = model_final.evaluate_generator(val_generator, steps=10)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

    return history

runCNN("/media/afit/HDD03/final_datasets/RGB_dataset", "test")