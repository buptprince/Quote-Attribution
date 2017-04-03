import tensorflow as tf
import numpy as np
import os, pickle
from config import Config
from util import Util

class mlp:
    def __init__(self):
        self.config = Config()
        self.util = Util()

    def loadData(self):
        dataPath = self.config.qVecMatPath
        X = None

        for fname in os.listdir(dataPath):
            with open(os.path.join(dataPath, fname), 'rb') as f:
                mat = pickle.load(f)
                if X is None:
                    X = mat
                    continue
                X = np.append(X, mat, axis=0)
        Y = X[:, 0]
        X = X[:, 1:]
        print "X:", X.shape
        print "Y:", Y.shape
        return X, Y

    def forwardProp(self, X, W1, W2):
        h = tf.nn.sigmoid(tf.matmul(X, W1))
        yhat = tf.matmul(h, W2)
        return yhat

    def initWeight(self, shape):
        weight = tf.random_normal(shape, stddev=self.config.mlp['stddev'])
        return tf.Variable(weight)

    def model(self):
        dataPath = self.config.qVecMatPath
        X, Y = self.loadData()

        trainLen = int(X.shape[0]*self.config.mlp['train'])
        Xtrain, Xtest = X[:trainLen], X[trainLen:]
        Ytrain, Ytest = Y[:trainLen], Y[trainLen:]

        print "Xtrain:", Xtrain.shape, "Xtest:", Xtest.shape
        print "Ytrain:", Ytrain.shape, "Ytest:", Ytest.shape

        X = tf.placeholder("float", shape=[None, self.config.wordDim])
        y = tf.placeholder("float", shape=[None, self.util.nSpeakers])

        W1 = self.initWeight([self.config.wordDim, self.config.mlp['nHidden']])
        W2 = self.initWeight([self.config.mlp['nHidden'], self.util.nSpeakers])

        yhat = self.forwardProp(X, W1, W2)
        predict = tf.argmax(yhat, axis=1)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
        update = tf.train.GradientDescentOptimizer(self.config.mlp['alpha']).minimize(cost)


if __name__ == '__main__':
    obj = mlp()
    obj.model()
