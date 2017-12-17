import tensorflow as tf
import numpy as np

xy = np.loadtxt('../data/data-04-zoo.csv', delimiter=',', dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[: ,[-1]]

features = x_data.shape[1]
classes = 7 # 0~ 6

X = tf.placeholder(tf.float32, [None, features])
Y = tf.placeholder(tf.int32, [None, 1])

Y_one_hot = tf.one_hot(Y, classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, classes])

W = tf.Variable(tf.random_normal([features, classes]), name = "weight")
b = tf.Variable(tf.random_normal([classes]), name="bias" )

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1)) #Y => one hot

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y_one_hot)
cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    dict = {X: x_data, Y: y_data}
    for step in range(2000):
        sess.run(optimizer, feed_dict = dict)
        if step % 100 == 0:
            loss, acc= sess.run([cost, accuracy], feed_dict = dict)
            print("STEP: {}, LOSS: {}, ACC: {}".format(step, loss, acc))

    #let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_data})
    #flatten... [[1], [0], [2], [3]..[]] => [ 1, 0, 2, 3, ...]
    for p, y in zip(pred, y_data.flatten()): #zip: 각각의 element를 tuple로 묶는듯.
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))