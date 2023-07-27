import numpy as np
import itertools
from matplotlib import pyplot as plt
from sklearn.metrics import auc

# Plot Accuracy and Loss over Time
# RETRIEVED: http://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
def plot(history, fileName):
    # Accuracy
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model_Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Num_Epochs')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(fileName + '_Model_Accuracy.png')
    # Loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model_Loss')
    plt.ylabel('Loss')
    plt.xlabel('Num_Epochs')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(fileName + '_Model_Loss.png')
    plt.close()

def cmPlot(cm, class_labels, fileName):
    class_count = len(class_labels)
    fig = plt.figure(fileName)
    fig.set_size_inches(10, 8)
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(fileName)
    plt.colorbar()
    tick_marks = np.arange(class_count)
    plt.xticks(tick_marks, class_labels, rotation=0)
    plt.yticks(tick_marks, class_labels)
    #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('plots/CM_' + fileName + '.png', dpi=100)
    np.set_printoptions(precision=2)
    plt.close()

def rocPlot(fpr, tpr, fileName):
    # Compute
    roc_auc = auc(tpr, fpr)

    plt.figure()
    lw = 2
    plt.plot(tpr, fpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('plots/ROC_' + fileName + '.png')
