'''
 Recurrent Neural Network Model


'''

import tensorflow as tf
import numpy as np
import os, pickle
from config import Config
from util import Util

class rnn:
    def __init__(self, cellType="basic"):
        self.cellType = cellType

        self.config = Config()
        self.config.rnn = self.config.rnn[self.cellType]
        self.util = Util()

        X, Y = self.util.loadData()
        trainLen = int(X.shape[0]*self.config.rnn['train'])
        self.Xtrain, self.Xtest = X[:trainLen], X[trainLen:]
        self.Ytrain, self.Ytest = Y[:trainLen], Y[trainLen:]

        self.Xtrain, self.Xtest = self.genBatch(self.Xtrain), self.genBatch(self.Xtest)
        self.Ytrain, self.Ytest = self.genBatch(self.Ytrain), self.genBatch(self.Ytest)

    def rnnCell(self):
        cell = None
        if self.cellType is "LSTM":
            cell = tf.contrib.rnn.BasicLSTMCell(self.config.rnn['stateSize'])
            print "Loaded basic LSTM Cell"
        else:
            cell = tf.contrib.rnn.BasicRNNCell(self.config.rnn['stateSize'])
            print "Loaded basic RNN Cell"
        return cell

    def genBatch(self, x):
        numStep = self.config.rnn['numStep']

        batchSize = x.shape[0] // numStep
        x = x[:batchSize*numStep-x.shape[0],:]
        x = x.reshape(batchSize, numStep, x.shape[-1])

        return x

    def model(self):
        self.X = tf.placeholder("float", shape=(None, self.config.rnn['numStep'], self.config.wordDim))
        self.y = tf.placeholder("float", shape=(None, self.config.rnn['numStep'], self.util.nSpeakers))

        cell = self.rnnCell()
        rnnOutputs, rnnStates = tf.nn.dynamic_rnn(cell, self.X, dtype="float")

        W = tf.Variable(tf.random_normal(shape=(self.config.rnn['stateSize'], self.util.nSpeakers), stddev=self.config.rnn['stddev']))
        b = tf.Variable(tf.constant(0.1, shape=[self.util.nSpeakers]))

        rnnOutputs = tf.unstack(rnnOutputs, num=self.config.rnn['numStep'], axis=1)

        logits = [tf.matmul(x, W)+b for x in rnnOutputs]
        logits = tf.stack(logits, axis=1)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits))
        self.update = tf.train.AdamOptimizer(self.config.rnn['alpha']).minimize(self.cost)

        self.accuracy = tf.equal(tf.argmax(logits,1), tf.argmax(self.y,1))

    def train(self):
        print "Xtrain:", self.Xtrain.shape, "Xtest:", self.Xtest.shape
        print "Ytrain:", self.Ytrain.shape, "Ytest:", self.Ytest.shape

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in xrange(self.config.rnn['epochs']):
            sess.run(self.update, feed_dict={
                self.X: self.Xtrain,
                self.y: self.Ytrain
            })

            trainAccuracy = np.mean(sess.run(self.accuracy, feed_dict={
                self.X: self.Xtrain,
                self.y: self.Ytrain
            }))

            testAccuracy = np.mean(sess.run(self.accuracy, feed_dict={
                self.X: self.Xtest,
                self.y: self.Ytest
            }))

            if (epoch+1) % self.config.rnn['disp'] == 0:
                print "Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%" % (epoch + 1, 100. * trainAccuracy, 100. * testAccuracy)

        sess.close()

if __name__ == '__main__':
    obj = rnn(cellType="basic")
    obj.model()
    obj.train()