import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Downloading MNIST data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Standard soft max
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Placeholders for storing the labels
y_ = tf.placeholder(tf.float32, [None, 10])
# Getting the cross entropy loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# Initializing optimizer and throwing the loss function into the optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

session = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Training standard soft max
for _ in range(100):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# Initialization methods for weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Convolution 2D operation
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# Max pool 2x2 operation
def max_pool_2x2_(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# CNN architecture
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2_(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2_(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# Matmul for fast matrix multiplication
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Reduce methods to calculate cross entropy loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# test accuracy = 0.9921000003814697
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for iteration in range(20000):
        batch = mnist.train.next_batch(50)

        if iteration % 100 == 0:
            train_accuracy = accuracy.eval(
                feed_dict={
                    x: batch[0],
                    y_: batch[1],
                    keep_prob: 1.0
                }
            )
            print("Step {}, training accuracy = {}".format(iteration, train_accuracy))

        # Whenever you are running the graph, make sure to fill in placeholders with feed_dict
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy = {}".format(accuracy.eval(
        feed_dict={
            x: mnist.test.images,
            y_: mnist.test.labels,
            keep_prob: 1.0
        }
    )))
