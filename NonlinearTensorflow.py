import tensorflow as tf
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

#X = 8 * np.random.rand(100, 1) - 4
#Y = 2.7 - 1.75 * X + 0.42 * X * X + np.random.randn(100, 1)
x_data = 8 * np.random.rand(100, 1) - 4
y_data = 2.7 - 1.75 * x_data + 0.42 * x_data * x_data + np.random.randn(100, 1)

X = tf.placeholder(dtype=tf.float32, shape=[None, 1])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

W1 = tf.Variable(initial_value=[0], dtype='float32')
W2 = tf.Variable(initial_value=[0], dtype='float32')
b= tf.Variable(initial_value=0, dtype='float32')

hypo = W1*X + W2*X*X + b
cost= tf.reduce_mean(tf.square(hypo - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

cost_hist = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(201):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        cost_hist.append(sess.run(cost, feed_dict={X:x_data, Y:y_data}))
        if i % 20 == 0:
            ans = sess.run(cost, feed_dict={X:x_data, Y:y_data})
            print(i, ans)
    
    y_pred = sess.run(hypo, feed_dict={X:x_data})
    x = sorted(zip(x_data, y_pred))
    np_x = np.array(x)
    plt.scatter(x_data, y_data)
    plt.plot(np_x[:, 0], np_x[:,1])

plt.plot(range(201), cost_hist)

# logistic regression
from sklearn import datasets
iris = datasets.load_iris()

iris_X = tf.placeholder(dtype=tf.float32, shape=[None,4])
iris_Y = tf.placeholder(dtype=tf.float32, shape=[None,1])
iris_W = tf.Variable(initial_value=np.zeros((4,1)), dtype='float32')
b = tf.Variable(initial_value = 1, dtype='float32')
hyp = tf.sigmoid(tf.matmul(iris_X, iris_W) + b)
iris_cost = tf.reduce_mean(-tf.matmul(tf.transpose(iris_Y),tf.math.log(hyp))- tf.matmul(tf.transpose(1-iris_Y), tf.math.log(1 - hyp)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0036).minimize(iris_cost)


#data preprocessing
x_data = iris['data']
y_data = iris['target']
y_for0 = y_data == 0
y_for1 = y_data == 1
y_for2 = y_data == 2
x_data = np.array(x_data)
y_for0 = np.array(y_for0)
y_for0 =  np.resize(np.array(y_for0), (len(y_data), 1)).astype('float32')
y_for1 = np.resize(np.array(y_for1), (len(y_data), 1)).astype('float32')
y_for2 =  np.resize(np.array(y_for2), (len(y_data), 1)).astype('float32')

y_data = [y_for0, y_for1, y_for2]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(201):
        sess.run(optimizer, feed_dict={iris_X:x_data, iris_Y:y_data[2]})
        if i % 20 == 0:
            print(i, 'th train', sess.run(iris_cost, feed_dict={iris_X:x_data, iris_Y:y_data[2]}))
            pred = sess.run(hyp, feed_dict={iris_X:x_data})
            pred = pred > 0.5
            right = y_data[2] == 1
    
            per = np.logical_not(np.logical_xor(pred, right))
            per = per.astype('float32')
            print('training accuracy:',np.mean(per) * 100,'%')
    

# multiclass classification with one-vs-all method
mul_iris_X = tf.placeholder(dtype=tf.float32, shape=[None,4])
mul_iris_Y = tf.placeholder(dtype=tf.float32, shape=[None,1])
mul_iris_W = tf.Variable(initial_value=np.zeros((4,1)), dtype='float32')
mul_b = tf.Variable(initial_value = 1, dtype='float32')
mul_hyp = tf.sigmoid(tf.matmul(mul_iris_X, iris_W) + b)
mul_iris_cost = tf.reduce_mean(-tf.matmul(tf.transpose(mul_iris_Y),tf.math.log(mul_hyp))- tf.matmul(tf.transpose(1-mul_iris_Y), tf.math.log(1 - mul_hyp)))
mul_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00120).minimize(mul_iris_cost)

all_W = tf.placeholder(dtype=tf.float32, shape = [4,3])

pred_hyp = tf.argmax(tf.sigmoid(tf.matmul(mul_iris_X, all_W)),1)
weights = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for j in range(3):
        for i in range(2001):
            sess.run(mul_optimizer, feed_dict={mul_iris_X:x_data, mul_iris_Y:y_data[j]})
            if i % 200 == 0:
                print(i, 'th train', sess.run(mul_iris_cost, feed_dict={mul_iris_X:x_data, mul_iris_Y:y_data[j]}))
                pred = sess.run(mul_hyp, feed_dict={mul_iris_X:x_data})
                pred = pred > 0.5
                right = y_data[j] == 1
    
                per = np.logical_not(np.logical_xor(pred, right))
                per = per.astype('float32')
                print('training accuracy:',np.mean(per) * 100,'%')
        
        print()
        weights.append(sess.run(iris_W))
    weights = np.array(weights)
    weights = np.resize(weights, (3,4))
    weights = np.transpose(weights)
    pred = sess.run(pred_hyp, feed_dict={mul_iris_X:x_data, all_W:weights})
    pred = pred == iris['target']
    print('accuracy:', np.mean(pred)*100, '%')

