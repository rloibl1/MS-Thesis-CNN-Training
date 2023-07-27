import plotter as plt
import simplejson
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.models import Sequential, Model, model_from_json
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import rmsprop, SGD
from keras.callbacks import History
from keras import backend as K
from sklearn.metrics import confusion_matrix, roc_curve
from PIL import Image
from datagen import npyGen

def createDir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return None

# def showIncorrectImgs(results, fileList, fileName):
#     savePath = 'incorrect_images/' + fileName
#     createDir(savePath)
#
#     num_incorrect = 0
#     for i in range(results.shape[0]):
#         if results[i][0] != results[i][1]:
#             num_incorrect += 1
#             fileName = fileList[i].split('/')[1]
#             if results[i][0] == 0:
#                 true_label = 'airport'
#                 imgPath = dataset + '/airports/' + fileName
#             else:
#                 true_label = 'non_airport'
#                 imgPath = dataset + '/non_airports/' + fileName
#
#             print('Misclassified ' + true_label + ': ' + fileName)
#             img = Image.open(imgPath)
#             img.save(savePath + '/' + fileName)

def loadModel(fileName):
    # load json and create model
    json_file = open(fileName + '_final_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(fileName + '_final_model.h5')
    print("Loaded model from disk")
    return loaded_model

# evaluate loaded model on test data
def modelAccuracy(model, input_shape, data_dir, num_imgs):
    test_generator = npyGen(data_dir, batch_size=1, img_size=input_shape[0], channels=input_shape[2])
    opt = rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    score = model.evaluate_generator(test_generator, steps=num_imgs)
    print("Test %s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

def modelPredict(model, input_shape, data_dir, fileName):
    test_generator = npyGen(data_dir, batch_size=1, img_size=input_shape[0], channels=input_shape[2])
    opt = rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    probabilities = model.predict_generator(test_generator, len(test_generator.filenames))

    y_true = np.array([0] * 308 + [1] * 308).reshape((616, 1))
    y_pred = np.array(probabilities[:, 1]).reshape((616, 1))
    y_pred = y_pred > .5

    # Match up predictions with file locations
    results = np.hstack([y_true, y_pred])

    # Show images that were predicted wrong
    # showIncorrectImgs(results, test_generator.filenames, fileName)

    # Create Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    labels = np.asarray(['airport', 'non_airport'])
    plt.cmPlot(cm, labels, fileName)

    # Generate ROC Curve
    y_pred = probabilities[:, 0]
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plt.rocPlot(fpr, tpr, fileName)

# Execution Code
model_path = 'saved_models'
datasetsPath = '/home/afit/Desktop/Datasets/'
models = os.listdir(model_path)
input_shape = (256, 256, 1)
num_test_imgs = 1891

for model in models:
    print(model)

    if model == '6clr':
        input_shape = (256, 256, 6)
    elif model == '5clr':
        input_shape = (256, 256, 5)
    elif model == '4clr':
        input_shape = (256, 256, 4)
    elif model == '3clr':
        input_shape = (256, 256, 3)
    elif model == '2clr':
        input_shape = (256, 256, 2)
    else:
        input_shape =(256, 256, 1)

    loaded_model = loadModel(os.path.join(model_path, model, model))
    datasetPath = os.path.join(datasetsPath, model, 'test_dict')
    modelAccuracy(loaded_model, input_shape, datasetPath, num_imgs=num_test_imgs)