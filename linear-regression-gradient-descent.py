import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = X * W

cost = tf.reduce_sum(tf.square(hypothesis - Y))

#minimize the gradient descent using derivative
# W := W - learning_rate * derivative of W
# 그러나 tensorflow에서는 그냥 W = f(W) 하는 식으로 집어넣을 수 없고,
# assign을 이용해야한다. 
learning_rate = 0.1
gradient = tf.reduce_mean(( W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)
#update를 그래프에서 실행시키면 된다
#수동 cost minimize임

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
train = optimizer.minimize(cost)
#이건 자동 minimize임


sess = tf.Session()

sess.run(tf.global_variables_initializer())


for step in range(21):
    sess.run(update, feed_dict = {X: x_data, Y: y_data})
    print("Step: {} // cost: {} // W: {})".format(step, sess.run(cost,  feed_dict = {X: x_data, Y: y_data}), sess.run(W)))
    

