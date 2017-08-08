import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32)
# Also tf.float32 implicitly
node2 = tf.constant(4.0)

print(node1, node2)

session = tf.Session()
print(session.run([node1, node2]))

node3 = tf.add(node1, node2)
print("node3 = {}".format(node3))
print("session.run(node3) = {}".format(session.run(node3)))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
# + provides a shortcut for tf.add(a, b)
adder_node = a + b

print(session.run(adder_node, {a: 3, b: 4.5}))
print(session.run(adder_node, {a: [1, 3], b: [2, 4]}))

add_and_Triple = adder_node * 3
print(session.run(add_and_Triple, {a: 3, b: 4.5}))

W = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()
session.run(init)
print(session.run(linear_model, {x: [1, 2, 3, 4]}))

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(session.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

fixW = tf.assign(W, [-1.0])
fixB = tf.assign(b, [1.0])
session.run([fixW, fixB])
print(session.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# Reset values to incorrect defaults
session.run(init)

for _ in range(1000):
    session.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(session.run([W, b]))

# Evaluate training accuracy
curr_W, curr_b, curr_loss = session.run([W, b, loss], {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
print("W: {} b: {} loss: {}".format(curr_W, curr_b, curr_loss))

import numpy as np

# Declare list of features. We only have one real-valued feature. There are
# many types of columns that are more complicated and useful
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# logistic regression, linear classification, logistic classification, and
# many neural network classifiers and regressors. The following code
# provides an estimator that does linear regression.
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use two data sets: one for training and one for evaluation
# We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

input_function = tf.contrib.learn.io.numpy_input_fn({"x": x_train}, y_train,
                                                    batch_size=4,
                                                    num_epochs=1000)
evaluation_input_function = tf.contrib.learn.io.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000
)

# We can invoke 1000 training steps by invoking the method and passing the
# training data set.
estimator.fit(input_fn=input_function, steps=1000)

# Evaluate the model
train_loss = estimator.evaluate(input_fn=input_function)
eval_loss = estimator.evaluate(input_fn=evaluation_input_function)
print("Train loss = {}".format(train_loss))
print("Evaluation loss = {}".format(eval_loss))
