import tensorflow as tf
import random
import numpy as np

pool = []
ans = []


for i in range(5000):
    r = np.random.randint(100, size= (10, 1))
    pool.append(r)
    ans.append(np.sqrt(r))

sess = tf.InteractiveSession()

W1 = tf.Variable(tf.random_normal([1, 100], stddev=0.01))
# W1 = tf.Variable(tf.zeros([1, 10]))
b1 = tf.Variable(tf.zeros([100]))
x1 = tf.placeholder(tf.float32, [None,1])

y1 = tf.nn.relu(tf.matmul(x1, W1) + b1)

W2 = tf.Variable(tf.random_normal([100,50], stddev=0.01))
b2 = tf.Variable(tf.zeros([50]))
x2 = y1
y2 = tf.nn.relu(tf.matmul(x2, W2) + b2)

W3 = tf.Variable(tf.random_normal([50, 1], stddev=0.01))
b3 = tf.Variable(tf.zeros([1]))

x3 = y2
y3 = tf.matmul(x3, W3) + b3

y_ = tf.placeholder(tf.float32, [None, 1])
# cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y2,1e-10,1.0)))

# error = tf.subtract(y_, y3)
mse = tf.reduce_mean(tf.squared_difference(y3 ,y_))

train_step = tf.train.GradientDescentOptimizer(.001).minimize(mse)

init = tf.global_variables_initializer()
sess.run(init)

# def test():
#     counter = 0
#     test_pool = []
#     test_ans = []

#     for k in range(50):
#         r = random.randint(1, 100)
#         test_pool.append(r)
#         test_ans.append(r ** (1/2))
#     for m in range(50):
#         counter += (100 * abs(sess.run(y3, feed_dict = {x1: [[test_pool[m]]]})[0][0] - test_ans[m]) / test_ans[m])
#     #         counter += 1
#     print (counter / 50)

for i in range(5000):
    sess.run(train_step, feed_dict = {x1: pool[i], y_: ans[i]})
    if i % 100 ==0:
        pass
        # test()

while (True):
    print ("Enter a Number:")
    inp = input()
    print ("Sqrt ( "+ str(inp) + " ) = " + str(np.sqrt(int(inp))))
    print ("Predicted:" + str(sess.run(y3, feed_dict = {x1: [[inp]]})[0][0]))
