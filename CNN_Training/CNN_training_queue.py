import os
import train_squeezenet
import pickle
import time
from matplotlib import pyplot as plt

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        f.close()

# Params
batch_size = 64
epochs = 200
input_shape = (256, 256, 1)
num_classes = 2
num_training = 6621
num_validation = 945
num_test = 1891

datasetsPath = '/home/afit/Desktop/Datasets/'
historyPath = 'History'
datasets = os.listdir(datasetsPath)

# Init history list
history = []
if not os.path.exists(historyPath):
    os.makedirs(historyPath)

for dataset in datasets:

    if dataset == '6clr':
        input_shape = (256, 256, 6)
    elif dataset == '5clr':
        input_shape = (256, 256, 5)
    elif dataset == '4clr':
        input_shape = (256, 256, 4)
    elif dataset == '3clr':
        input_shape = (256, 256, 3)
    elif dataset == '2clr':
        input_shape = (256, 256, 2)
    else:
        input_shape =(256, 256, 1)

    if dataset == 'B5' or dataset == 'B8':
        # Start Time
        start_time = time.time()
        print(dataset)
        # Execution Code
        fileName = dataset.split('_')[0]
        datasetPath = os.path.join(datasetsPath, dataset)
        run_history = train_squeezenet.squeezeNet_Train(datasetPath, fileName, batch_size=batch_size, epochs=epochs,
                                                         input_shape=input_shape, num_classes=num_classes,
                                                         num_training=num_training, num_validation=num_validation,
                                                         num_test=num_test)
        history.append(run_history)
        save_obj(run_history.history, os.path.join(historyPath, dataset + '_history'))

        # End Time
        print('--- %s seconds ---' % (time.time() - start_time))
