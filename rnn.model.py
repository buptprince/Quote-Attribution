'''
 Recurrent Neural Network Model


'''

import tensorflow as tf
import numpy as np
import os, pickle
from config import Config
from util import Util

class rnn:
    def __init__(self):
        self.config = Config()
        self.util = Util()

        X, Y = self.util.loadData()
        trainLen = int(X.shape[0]*self.config.mlp['train'])
        self.Xtrain, self.Xtest = X[:trainLen], X[trainLen:]
        self.Ytrain, self.Ytest = Y[:trainLen], Y[trainLen:]

    def rnnCell(self):
        cell = None
        if self.config.rnn['isLSTM']:
            cell = tf.contrib.rnn.BasicLSTMCell(self.config.rnn['stateSize'])
        else:
            cell = tf.contrib.rnn.BasicRNNCell(self.config.rnn['stateSize'])
        return cell

    def model(self):
        self.X = tf.placeholder("float", shape=(None, self.config.rnn['numStep'], self.config.wordDim))
        self.y = tf.placeholder("float", shape=(None, self.config.rnn['numStep'], self.util.nSpeakers))

        cell = self.rnnCell()
        rnnOutputs, rnnStates = tf.nn.dynamic_rnn(cell, self.X, dtype="float")

        W = tf.Variable(tf.random_normal(shape=(self.config.rnn['stateSize'], self.util.nSpeakers)))
        b = tf.Variable(tf.constant(0.1, shape=[self.util.nSpeakers]))

        rnnOutputs = tf.unstack(rnnOutputs, num=self.config.rnn['numStep'], axis=1)

        logits = [tf.matmul(x, W)+b for x in rnnOutputs]
        # yhat = tf.argmax(tf.reshape(logits, (logits.shape[0]*logits.shape[1], logits.shape[2])))

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits))
        self.update = tf.train.AdamOptimizer(self.config.rnn['alpha']).minimize(self.cost)




if __name__ == '__main__':
    obj = rnn()
    obj.model()
