import tensorflow as tf
import sys
import os
import numpy as np

sess = tf.Session()

sess.run(tf.global_variables_initializer())
result = sess.run(operation)
print(result)

# Linear Regression
X = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='input')
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='output')
b = tf.Variable(initial_value = 1, dtype='float32')
W = tf.Variable(initial_value = [0], dtype='float32')
hypothesis = W * X + b
m = tf.shape(X)[0]
m = tf.dtypes.cast(m, dtype=tf.float32)

Cost = (1 / m) * tf.math.reduce_sum(tf.math.square(hypothesis - Y))


import matplotlib.pyplot as plt
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
x_data = np.random.rand(100, 1)
y_data = 8 - 2.5 * x_data + np.random.randn(100, 1)
sess.run(tf.global_variables_initializer())
cost_hist = []
for step in range(201):
    sess.run(optimizer.minimize(Cost), feed_dict={X: x_data, Y:y_data})
    cost_hist.append(sess.run(Cost, feed_dict={X:x_data, Y:y_data}))
    if step % 20 == 0:
        a= sess.run(hypothesis, feed_dict = {X:x_data, Y:y_data})
        print(step, sess.run(Cost, feed_dict = {X:x_data, Y: y_data}))

plt.plot(range(201), cost_hist)

plt.subplot(121)        
plt.plot(range(201), cost_hist)
y_prediction = sess.run(hypothesis, feed_dict = {X: x_data})
plt.subplot(122)
plt.scatter(x_data, y_data)
plt.plot(x_data, y_prediction)

# normal equation
W_norm = np.linalg.pinv(np.matmul(np.transpose(x_data), x_data)) #pseudo inverse rather than inverse
W_norm = np.matmul(W_norm, np.transpose(x_data))
W_norm = np.matmul(W_norm, y_data-sess.run(b))
sess.run(tf.global_variables_initializer())
hyp_norm = W_norm * X + b
Cost_norm = tf.math.reduce_mean(tf.square(hyp_norm - Y))
y_norm_pred = sess.run(hyp_norm ,feed_dict= {X:x_data}) 
print(y_norm_pred.shape)
sess.run(Cost_norm, feed_dict={X:x_data, Y: y_data})
plt.scatter(x_data, y_data)
plt.plot(x_data, y_norm_pred)
