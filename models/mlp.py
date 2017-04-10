'''
 Multi Layered Perceptron Model
 3 layered perceptron model with
 1st Layer[Input Layer]: 50 nodes [config.wordDim]
 2nd Layer[Hidden Layer]: 1200 Nodes [config.mlp['nHidden']]
 3rd Layer[Output Layer]: 65 Nodes [util.nSpeakers]

 H = sigmoid(X.W1)
 Yhat = softmax(H.W2)

 Test Accuracy: 36.96%
 Train Accuracy: 41.98%
'''



import tensorflow as tf
import numpy as np
import os, pickle, sys
sys.path.insert(0,'..')
from config import Config
from util import Util

class mlp:
    def __init__(self):
        self.config = Config()
        self.util = Util()

        X, Y = self.util.loadData()
        trainLen = int(X.shape[0]*self.config.mlp['train'])
        self.Xtrain, self.Xtest = X[:trainLen], X[trainLen:]
        self.Ytrain, self.Ytest = Y[:trainLen], Y[trainLen:]

    def forwardProp(self, X, W1, W2):
        h = tf.nn.sigmoid(tf.matmul(X, W1))
        yhat = tf.matmul(h, W2)
        return yhat

    def initWeight(self, shape):
        weight = tf.random_normal(shape, stddev=self.config.mlp['stddev'])
        return tf.Variable(weight)

    def model(self):
        self.X = tf.placeholder("float", shape=(None, self.config.wordDim))
        self.y = tf.placeholder("float", shape=(None, self.util.nSpeakers))

        self.W1 = self.initWeight([self.config.wordDim, self.config.mlp['nHidden']])
        self.W2 = self.initWeight([self.config.mlp['nHidden'], self.util.nSpeakers])

        self.yhat = self.forwardProp(self.X, self.W1, self.W2)
        self.predict = tf.argmax(self.yhat, axis=1)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.yhat))
        self.update = tf.train.GradientDescentOptimizer(self.config.mlp['alpha']).minimize(self.cost)

    def train(self):
        print "Xtrain:", self.Xtrain.shape, "Xtest:", self.Xtest.shape
        print "Ytrain:", self.Ytrain.shape, "Ytest:", self.Ytest.shape

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()
        for epoch in xrange(self.config.mlp['epochs']):
            sess.run(self.update, feed_dict={
                self.X: self.Xtrain,
                self.y: self.Ytrain
            })

            trainAccuracy = np.mean(np.argmax(self.Ytrain, axis=1) ==
                                 sess.run(self.predict, feed_dict={
                                 self.X: self.Xtrain,
                                }))
            testAccuracy  = np.mean(np.argmax(self.Ytest, axis=1) ==
                                 sess.run(self.predict, feed_dict={
                                 self.X: self.Xtest,
                                }))
            if epoch%self.config.mlp['disp'] == 0:
                print "Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%" % (epoch + 1, 100. * trainAccuracy, 100. * testAccuracy)

        # save_path = saver.save(sess, self.config.mlp['modelPath'])
        # print "Model saved in file: %s" % save_path
        sess.close()

    def predict(self, X):
        with tf.Session() as sess:
            sess.restore(sess, self.config.mlp['modelPath'])
            Yhat = sess.run(self.predict, feed_dict={
                self.X: X
            })
            print Yhat

if __name__ == '__main__':
    os.chdir('..')
    
    obj = mlp()
    obj.model()
    obj.train()
