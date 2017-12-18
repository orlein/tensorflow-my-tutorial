import numpy as np
import tensorflow as tf
import math
xy = np.loadtxt('../data/data-02-stock_daily.csv', delimiter=',', dtype = np.float32)

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)
    #normalize하는 함수임

xy = MinMaxScaler(xy) 
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
cost = tf.reduce_mean(tf.square(hypothesis - Y))

train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:x_train_data, Y:y_train_data})
        print("[{}], Cost: {}, prediction: {}".format(step, cost_val, np.mean(hy_val, 0)))
    print("{}".format(sess.run([cost, hypothesis], feed_dict={X: x_test_data, Y:y_test_data})))
    