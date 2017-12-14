import tensorflow as tf
import matplotlib.pyplot as plt

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32) #임의대로 바꿔서 보고싶다
#이전엔 그냥 랜덤값이었는데, 이렇게 하는 이유는 W값에 따른 코스트 변화를 보고싶어서
#당연히 W=1에서 극소값을 가지는 2차함수 꼴임

hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))


sess = tf.Session()
sess.run(tf.global_variables_initializer())

W_val = []
cost_val = [] 
err = 1
for i in range(-30, 50):
    feed_W = i*0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict = {W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

plt.plot(W_val, cost_val)
plt.show()