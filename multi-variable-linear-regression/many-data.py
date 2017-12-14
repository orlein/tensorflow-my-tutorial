import numpy as np
import tensorflow as tf
import math
xy = np.loadtxt('../data/data-01-test-score.csv', delimiter=',', dtype = np.float32)

datalen = math.floor(len(xy) * 0.8)




x_train_data = xy[:datalen, 0:-1]
y_train_data = xy[:datalen, [-1]]
x_test_data = xy[datalen:, 0:-1]
y_test_data = xy[datalen:, [-1]]

#python 슬라이싱 기능 해설: xy를, 처음부터 끝까지 배열인데, 각각 인덱스 0부터 마지막 수 직전까지 잘라냄 + xy를, 처음부터 끝까지 배열인데, 각각 인덱스 마지막 수만 배열로써 잘라냄
# print(x_data.shape, x_data, len(x_data), x_data.shape[1])
# print(y_data.shape, y_data)

X = tf.placeholder(tf.float32, shape=[None, x_train_data.shape[1]])
Y = tf.placeholder(tf.float32, shape=[None, y_train_data.shape[1]])

W = tf.Variable(tf.random_normal([x_train_data.shape[1], y_train_data.shape[1]]), name='weight')
b = tf.Variable(tf.random_normal([y_train_data.shape[1]]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = 40
step = 0
# for step in range(2001):
while (cost_val > 1e-5):
    step+=1
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict = {X: x_train_data, Y: y_train_data})
    if step % 20 == 0:
        print("STEP: {} /// Cost: {}, Prediction: {}".format(step, cost_val, hy_val))

for instance in x_test_data:
    print("YOUR SCORE WILL BE: {}".format(
            sess.run(hypothesis, feed_dict={X: [instance]})
        )
    )
    
