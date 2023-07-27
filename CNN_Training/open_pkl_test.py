import pickle
import os
from matplotlib import pyplot as plt

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

path = 'History'
history = []

plt.figure()

models = os.listdir(path)
model_names = []

models.remove('2clr_history.pkl')
models.remove('3clr_history.pkl')
models.remove('4clr_history.pkl')
models.remove('B3_history.pkl')
models.remove('B4_history.pkl')
models.remove('B5_history.pkl')
models.remove('B6_history.pkl')

for model in models:
    history.append(load_obj(os.path.join(path, model)))
    model_names.append(model.split('.')[0])

for i in range(len(models)):
    plt.plot(history[i]['val_acc'])

plt.title('Validation_Accuracy_Combined')
plt.ylabel('Accuracy')
plt.xlabel('Num_Epochs')
plt.legend(model_names, loc='lower right')
plt.savefig('plots/Reduced_Training_Val_Accuracy_Combined.png')
