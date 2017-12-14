import tensorflow as tf

#x_data = [[73, 80, 75], [93, 88, 93], [89, 91, 90], [96, 98, 100], [73, 66, 70]]
# x1_data = [73., 80., 75.]
# x2_data = [93., 88., 93.]
# x3_data = [89., 91., 90.]
# x4_data = [96., 98., 100.]
# x5_data = [73., 66., 70.]
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x1 * w1 + x2 * w2 + x3 * w3


cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = 100
step = 0
# for step in range(2001):
while(cost_val > 1e-3):
    step+=1
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict = {x1: x1_data, x2: x2_data, x3: x3_data, y: y_data})
    
    if step % 20 == 0:
        print(step, cost_val, hy_val)

#아름답지 못함...ㅋㅋㅋ