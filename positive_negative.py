import tensorflow as tf
import random

probs = []
ans = []


for i in range(600):
    r = random.randint(-1000, 1000)
    probs.append(r)
    if (r < 0):
        ans.append(0)
    else:
     ans.append(1)

sess = tf.InteractiveSession()

W1 = tf.get_variable("W1", shape=[1, 10],
           initializer=tf.contrib.layers.xavier_initializer())
# W1 = tf.Variable(tf.zeros([1, 10]))
b1 = tf.Variable(tf.zeros([10]))
x1 = tf.placeholder(tf.float32, [1,1])

y1 = tf.matmul(x1, W1) + b1

W2 = W1 = tf.get_variable("W2", shape=[10, 1],
           initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.zeros([1]))
x2 = y1
y2 = tf.nn.sigmoid(tf.matmul(x2, W2) + b2)

y_ = tf.placeholder(tf.float32, [1, 1])
# cross_entropy = tf.reduce_mean(-tf.reduce_mean(y_ * tf.log(y1), reduction_indices = [1]))
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y1,1e-10,1.0)))

error = tf.subtract(y_, y2)
mse = tf.reduce_mean(tf.square(error))

train_step = tf.train.GradientDescentOptimizer(.01).minimize(mse)

init = tf.global_variables_initializer()
sess.run(init)

def test():
    counter = 0
    test_pool = []
    test_ans = []
    for k in range(100):
        r = random.randint(-1000, 1000)
        test_pool.append(r)
        if r < 0:
            test_ans.append(0)
        else:
            test_ans.append(1)
    for m in range(100):
        if sess.run(y2, feed_dict = {x1: [[test_pool[m]]]})[0][0] > .5 and test_ans[m] == 0:
            counter += 1
    print (counter / 100)

for i in range(500):
    sess.run(train_step, feed_dict = {x1: [[probs[i]]], y_: [[ans[i]]]})
    if i % 50 ==0:
        test()

while (True):
    print ("Enter a Number:")
    inp = input()
    if sess.run(y2, feed_dict = {x1: [[inp]]})[0][0] > 0:
        print ("This is a positive number\n")
    else:
        print ("This is a negative number\n")

