import tensorflow as tf

x_data = [[73., 80., 75.], [93., 88., 93.], [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]

X = tf.placeholder(tf.float32, shape=[None, 3]) #3은 variable 개수고... None은 그냥 instance가 n개란 뜻. 여기서 5개긴 하지만 원하는만큼 주고싶다는거.
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b # X * W가 아니라 matmul에 주의


cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = 100
# step = 0
for step in range(2001):
# while(cost_val > 1e-3):
    # step+=1
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict = {X: x_data, Y: y_data})
    
    if step % 20 == 0:
        print(step, cost_val, hy_val)