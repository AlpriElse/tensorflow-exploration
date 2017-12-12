import tensorflow as tf;
#   Define input values
T, F = 1.0, -1.0
bias = 1.0

#   Define training data
training_input = [
    [T, T, bias],
    [T, F, bias],
    [F, T, bias],
    [F, F, bias]
]
training_output = [
    [T],
    [T],
    [T],
    [F]
]

w = tf.Variable(tf.random_normal([3, 1]), dtype=tf.float32)

#   Step Activation Function
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

# Input as boooleans
def logicalOR(param1, param2):
    temp = {
        True: T,
        False: F
    }
    output = sess.run(step(tf.matmul([[temp[param1], temp[param2], bias]], w)))
    return output[0][0] > 0

print(logicalOR(True, True))
print(logicalOR(True, False))
print(logicalOR(False, True))
print(logicalOR(False, False))
