from models.mlp import mlp
from models.rnn import rnn

from util import Util

import argparse, itertools
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np

modelMap = {
    'MLP': [mlp()],
    'RNN': [
        rnn(cellType="basic", redChars=True, wrapDropout=True),
        rnn(cellType="LSTM", redChars=True, wrapDropout=True),
        rnn(cellType="GRU", redChars=True, wrapDropout=True)
    ]
}


"""
This function prints and plots the confusion matrix.
Normalization can be applied by setting `normalize=True`.
"""
def plotConfusionMatrix(cm,classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Testing Module")
    parser.add_argument('-model', type=str, help="Model Name")

    args = parser.parse_args()
    plt.figure()
    util = Util()
    classes = util.mainCharacters
    classes.append('others')

    for obj in modelMap[args.model]:
        X, Y = obj.Xtest, obj.Ytest

        obj.model()
        Yhat = obj.forecast(X)

        Y = Y.reshape(-1, Y.shape[-1])
        Y = np.argmax(Y, axis=1)
        cnfMat = metrics.confusion_matrix(Y, Yhat)
        print cnfMat
        # plotConfusionMatrix(cnfMat, classes, normalize=True, title=obj.name)

    plt.show()
