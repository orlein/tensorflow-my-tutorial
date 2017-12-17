import numpy as np
import tensorflow as tf
import math
xy = np.loadtxt('../data/data-03-diabetes.csv', delimiter=',', dtype = np.float32)

datalen = math.floor(len(xy) * 0.8)

x_train_data = xy[:datalen, 0:-1]
y_train_data = xy[:datalen, [-1]]
x_test_data = xy[datalen:, 0:-1]
y_test_data = xy[datalen:, [-1]]

#python 슬라이싱 기능 해설: xy를, 처음부터 끝까지 배열인데, 각각 인덱스 0부터 마지막 수 직전까지 잘라냄 + xy를, 처음부터 끝까지 배열인데, 각각 인덱스 마지막 수만 배열로써 잘라냄
# print(x_data.shape, x_data, len(x_data), x_data.shape[1])
# print(y_data.shape, y_data)

features = x_train_data.shape[1]
classes = y_train_data.shape[1]

X = tf.placeholder(tf.float32, shape=[None, features])
Y = tf.placeholder(tf.float32, shape=[None, classes])

W = tf.Variable(tf.random_normal([features, classes]), name='weight')
b = tf.Variable(tf.random_normal([classes]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed = {X: x_train_data, Y: y_train_data}
    cost_val = 40
    step = 0
    a = 0
    # for step in range(2001):
    while (a < 0.78):
        step+=1
        cost_val, _ = sess.run([cost, train], feed_dict=feed)
        if step % 20 == 0:
            # print("STEP: {} /// Cost: {}".format(step, cost_val))
            h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict=feed)
            # print("\nhypothesis: {}, Correct: {}, Accuracy: {}".format(h, c, a))
            print("Accuracy: {}, cost: {}".format(a, cost_val))

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_test_data, Y: y_test_data})
    print("\n\n\n\n test Accuracy: {}".format(a))
