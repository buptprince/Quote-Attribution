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

        X= np.array(X)
        Y = np.array(X[:, 0], dtype=int)
        Y = np.eye(self.util.nSpeakers)[Y]
        X = X[:, 1:]
        return X, Y

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
        dataPath = self.config.qVecMatPath
        X, Y = self.loadData()

        trainLen = int(X.shape[0]*self.config.mlp['train'])
        Xtrain, Xtest = X[:trainLen], X[trainLen:]
        Ytrain, Ytest = Y[:trainLen], Y[trainLen:]

        print "Xtrain:", Xtrain.shape, "Xtest:", Xtest.shape
        print "Ytrain:", Ytrain.shape, "Ytest:", Ytest.shape

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()
        for epoch in xrange(self.config.mlp['epochs']):
            sess.run(self.update, feed_dict={
                self.X: Xtrain,
                self.y: Ytrain
            })

            trainAccuracy = np.mean(np.argmax(Ytrain, axis=1) ==
                                 sess.run(self.predict, feed_dict={
                                 self.X: Xtrain,

                                }))
            testAccuracy  = np.mean(np.argmax(Ytest, axis=1) ==
                                 sess.run(self.predict, feed_dict={
                                 self.X: Xtest,

                                }))
            if epoch%self.config.mlp['disp'] == 0:
                print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * trainAccuracy, 100. * testAccuracy))

        save_path = saver.save(sess, self.config.mlp['modelPath'])
        print "Model saved in file: %s" % save_path
        sess.close()

    def predict(self, X):
        with tf.Session() as sess:
            sess.restore(sess, self.config.mlp['modelPath'])
            Yhat = sess.run(self.predict, feed_dict={
                self.X: X
            })

            print Yhat



if __name__ == '__main__':
    obj = mlp()
    obj.model()
    obj.train()
