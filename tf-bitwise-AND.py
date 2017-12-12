# The following code follows a tutorial found at:
# https://towardsdatascience.com/tensorflow-for-absolute-beginners-28c1544fb0d6

import tensorflow as tf;

#   Declare values and training data
T, F = 1.0, -1.0
bias = 1.0
training_input = [
    [T, T, bias],
    [T, F, bias],
    [F, T, bias],
    [F, F, bias]
]
training_output = [
    [T],
    [F],
    [F],
    [F]
]

#   Declare random neural network weights - to be optimized
w = tf.Variable(tf.random_normal([3, 1]), dtype=tf.float32)

#   Step Activation Function - "useful for binary classification"
#   This determines whether or not a "Neuron" (Node) will fire
def step(x):
    is_greater = tf.greater(x, 0)
    as_float = tf.to_float(is_greater)
    doubled = tf.multiply(as_float, 2)
    return tf.subtract(doubled, 1)

#   Set output equal to an evaluation of the
#   neural network with training input and weights
output = step(tf.matmul(training_input, w))

#   Set error equal to an error calculation
#   between evaluated values and output
#   mse - "mean squared error" - tracks training progress
error = tf.subtract(training_output, output)
mse = tf.reduce_mean(tf.square(error))

#   Calculate the amount of change needed to be
#   assigned to the weights
delta = tf.matmul(training_input, error, transpose_a=True)

#   Use calculated deltas to adjust weights
train = tf.assign(w, tf.add(w, delta))

#   Run graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#   Declaring training variables
err, target = 1, 0
epoch, max_epochs = 0, 10

#   Train Network
#   epochs - the number of times to step toward an optimized solution
#   target - the target error we are attempting to achieve
while err > target and epoch < max_epochs:
    epoch += 1
    err, _ = sess.run([mse, train])

print('epoch:', epoch, 'mse:', err)
print(sess.run(w))

# Input as booleans
def logicalAND(param1, param2):
    temp = {
        True: T,
        False: F
    }
    first = temp[param1]
    second = temp[param2]
    output = sess.run(step(tf.matmul([[first,second,bias]],w)))
    return output[0][0] > 0

# Test function that uses calculated weights
print(logicalAND(True,True)) # Expected: True
print(logicalAND(True,False)) # Expected: False
print(logicalAND(False,True)) # Expected: False
print(logicalAND(False,False)) # Expected: False
