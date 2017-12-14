import tensorflow as tf

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.Variable(5.) #직접 가져온 W값과 그 변화

hypothesis = X * W

gradient = tf.reduce_mean((hypothesis - Y) * X) * 2 #직접미분한 수동계산값

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)

gvs = optimizer.compute_gradients(cost) #자동계산한 gradient와 W값 tuple

apply_gradients = optimizer.apply_gradients(gvs) # 자동계산값 적용

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print("step: {}// (gradient, W): [{}, {}] // gvs: {}".format(step, sess.run(gradient), sess.run(W), sess.run(gvs)))  # 스텝 // 수동값 // 자동값
    sess.run(apply_gradients)