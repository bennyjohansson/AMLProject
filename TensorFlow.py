import numpy as np
import tensorflow as tf


#Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

#Defining model
x = tf.placeholder(tf.float32)
linear_model = W*x + b
y = tf.placeholder(dtype=tf.float32)

#Loss
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

#Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#training loop
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

for i in range(1,1000):

	sess.run(train, {x: x_train, y: y_train})
	
#Running answer
print sess.run([W,b])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
