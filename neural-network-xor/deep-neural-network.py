import tensorflow as tf
import numpy as np


x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(dtype=tf.float32)
Y = tf.placeholder(dtype=tf.float32)

W1 = tf.Variable(tf.random_normal([2,10]), name='weight')
b1 = tf.Variable(tf.random_normal([10]), name='bias')
layer1 = tf.sigmoid(tf.matmul(X,W1) + b1)

W2 = tf.Variable(tf.random_normal([10,10]), name='weight')
b2 = tf.Variable(tf.random_normal([10]), name='bias')
layer2 = tf.sigmoid(tf.matmul(layer1,W2) + b2)

W3 = tf.Variable(tf.random_normal([10,10]), name='weight')
b3 = tf.Variable(tf.random_normal([10]), name='bias')
layer3 = tf.sigmoid(tf.matmul(layer2,W3) + b3)


W4 = tf.Variable(tf.random_normal([10,1]), name='weight')
b4 = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = tf.sigmoid(tf.matmul(layer3,W4) + b4)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if(step%100 == 0):
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))

    h,c,a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("HYPOTHESIS: {}, CORRECT:{}, ACCURACY:{}".format(h,c,a))
