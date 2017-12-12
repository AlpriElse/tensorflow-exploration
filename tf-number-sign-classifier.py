import tensorflow as tf;
bias = 1.0

training_input = [
    [1, bias],
    [2, bias],
    [3, bias],
    [4, bias],
    [111233, bias],
    [13, bias],
    [6, bias],
    [19827, bias],
    [129038, bias],
    [600, bias],
    [12, bias],
    [10, bias],
    [-1, bias],
    [-2, bias],
    [-3, bias],
    [-4, bias],
    [-111233, bias],
    [-13, bias],
    [-6, bias],
    [-19827, bias],
    [-129038, bias],
    [-600, bias],
    [-12, bias],
    [-10, bias]
]
training_output = [
    [1.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
    [-1.0],
    [-1.0],
    [-1.0],
    [-1.0],
    [-1.0],
    [-1.0],
    [-1.0],
    [-1.0],
    [-1.0],
    [-1.0],
    [-1.0],
    [-1.0],
]

#   Define Random Weights
w = tf.Variable(tf.random_normal([2,1]), dtype=tf.float32)

#   Step Activation function
def step(x):
    is_greater = tf.greater(x, 0)
    as_float = tf.to_float(is_greater)
    doubled = tf.multiply(as_float, 2)
    return tf.subtract(doubled, 1)

output = step(tf.matmul(training_input, w))
error = tf.subtract(training_output, output)
mse = tf.reduce_mean(tf.square(error))

delta = tf.matmul(training_input, error, transpose_a=True)
train = tf.assign(w, tf.add(w, delta))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

err, target = 1, 0
epoch, max_epochs = 0, 10

while err > target and epoch < max_epochs:
    epoch += 1
    err, _ = sess.run([mse, train])

print('epoch:', epoch, 'mse:', err)
print(sess.run(w))

def classifyNumber(num):
    output = sess.run(step(tf.matmul([[num, bias]], w)))
    if output[0][0] > 0:
        return "Positive"
    else:
        return "Negative"

print(classifyNumber(-100))
print(classifyNumber(-1))
print(classifyNumber(-2))
print(classifyNumber(1.0))
