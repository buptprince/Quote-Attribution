'''
 Contextual Recurrent Neural Model
 [[ 170   76   38    0    3    0  560]
 [  65   66   16    1    0    0  408]
 [ 109   42   69    0    4    0  439]
 [  28   13   11    0    1    0  230]
 [  24   18    7    0    0    0  158]
 [  15   22   14    0    1    0  109]
 [ 150  121   69    4    4    1 1449]]
 [ 0.24147727  0.14442013  0.15558061  0.          0.          0.          0.5626092 ]
'''

import tensorflow as tf
import numpy as np
import os, pickle, sys
sys.path.insert(0, '..')

from config import Config
from util import Util
import test

class contRNN:
    def __init__(self):
        self.config = Config()
        self.util = Util()

        X, Y = self.util.loadData(redChars=True)
        trainLen = int(X.shape[0]*self.config.contRNN['train'])
        self.Xtrain, self.Xtest = X[:trainLen], X[trainLen:]
        self.Ytrain, self.Ytest = Y[:trainLen], Y[trainLen:]

        self.Xtrain, self.Xtest = self.genBatch(self.Xtrain), self.genBatch(self.Xtest)
        self.Ytrain, self.Ytest = self.genBatch(self.Ytrain), self.genBatch(self.Ytest)

    def genBatch(self, x):
        numStep = self.config.contRNN['numStep']

        batchSize = x.shape[0] // numStep
        x = x[:batchSize*numStep-x.shape[0],:]
        x = x.reshape(batchSize, numStep, x.shape[-1])

        return x

    def model(self):
        self.X = tf.placeholder('float', shape=(None, self.config.contRNN['numStep'], self.config.wordDim))
        self.y = tf.placeholder('float', shape=(None, self.config.contRNN['numStep'], self.util.nSpeakers))


        contOutput = self.addContextLayer()
        cell = tf.contrib.rnn.GRUCell(self.config.contRNN['stateSize'])
        cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(cell)
        outputs, states = tf.nn.dynamic_rnn(cell, contOutput,  dtype='float')

        W = tf.Variable(tf.random_normal(shape=(self.config.contRNN['stateSize'], self.util.nSpeakers), stddev=self.config.contRNN['stddev']))
        b = tf.Variable(tf.constant(0.1, shape=[self.util.nSpeakers]))

        outputs = tf.unstack(outputs, num=self.config.contRNN['numStep'], axis=1)

        logits = [tf.matmul(x, W)+b for x in outputs]
        logits = tf.stack(logits, axis=1)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits))
        self.update = tf.train.AdamOptimizer(self.config.contRNN['alpha']).minimize(self.cost)

        self.predict = tf.argmax(tf.reshape(logits, [-1, self.util.nSpeakers]),1)
        y_ = tf.argmax(tf.reshape(self.y, [-1, self.util.nSpeakers]),1)
        self.accuracy = tf.equal(self.predict, y_)

    # def addContextLayer(self):
    #     W = tf.Variable(tf.random_normal(shape=(self.config.wordDim, self.config.wordDim), stddev=self.config.contRNN['stddev']))
    #     b = tf.Variable(tf.constant(0.1, shape=[self.config.wordDim]))
    #
    #     X_ = tf.unstack(self.X, num=self.config.contRNN['numStep'], axis=1)
    #
    #     logits = [tf.tanh(tf.matmul(x, W)+b) for x in X_]
    #     return tf.stack(logits, axis=1)

    def addContextLayer(self):
        cell = tf.contrib.rnn.LSTMCell(self.config.contRNN['contStateSize'])
        cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(cell)
        outputs, states = tf.nn.dynamic_rnn(cell, self.X, dtype="float")
        return outputs

    def train(self):
        print "Xtrain:", self.Xtrain.shape, "Xtest:", self.Xtest.shape
        print "Ytrain:", self.Ytrain.shape, "Ytest:", self.Ytest.shape

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()

        for epoch in xrange(self.config.contRNN['epochs']):
            sess.run(self.update, feed_dict={
                self.X: self.Xtrain,
                self.y: self.Ytrain,
            })

            trainAccuracy = np.mean(sess.run(self.accuracy, feed_dict={
                self.X: self.Xtrain,
                self.y: self.Ytrain
            }))

            testAccuracy = np.mean(sess.run(self.accuracy, feed_dict={
                self.X: self.Xtest,
                self.y: self.Ytest
            }))

            if (epoch+1) % self.config.contRNN['disp'] == 0:
                print "Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%" % (epoch + 1, 100. * trainAccuracy, 100. * testAccuracy)

        save_path = saver.save(sess, self.config.contRNN['modelPath'])
        print "Model saved in file: %s" % save_path
        sess.close()

    def forecast(self, X):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.config.contRNN['modelPath'])
            Yhat = sess.run(self.predict, feed_dict={
                self.X: X
            })
            return Yhat

if __name__ == '__main__':
    os.chdir('..')

    obj = contRNN()
    obj.model()
    obj.train()

    test.genReport(obj.Ytest, obj.forecast(obj.Xtest))
