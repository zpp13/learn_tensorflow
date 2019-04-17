import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

n_input = 784
n_layer1 = 10
examples_to_show = 10

x_input = tf.placeholder('float', [None, n_input])
y_input = tf.placeholder('float', [None, n_layer1])

layer1_weights = tf.Variable(tf.random_normal([n_input, n_layer1]))
layer1_biases = tf.Variable(tf.random_normal([n_layer1]))

def addLayer(x_input, layer1_weights, layer1_biases, activation_function = None):
    output = tf.add(tf.matmul(x_input, layer1_weights), layer1_biases)
    if activation_function == None:
        return tf.nn.sigmoid(output)
    else:
        return tf.nn.softmax(output)
#输入测试集的x_input 和 y_input
def compute_acc(x_input_test,y_true_test):
     y_pre_test = sess.run(y_pre, feed_dict={x_input:x_input_test , y_input:y_true_test})
     correct_prediction=tf.equal(tf.argmax(y_pre_test,1),tf.argmax(y_true_test,1))
     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
     result = sess.run(accuracy, feed_dict={x_input:x_input_test , y_input:y_true_test})
     return result
y_pre = addLayer(x_input, layer1_weights, layer1_biases, activation_function=tf.nn.softmax)
y_true = y_input

cross_entropy = -tf.reduce_sum(y_true * tf.log(y_pre))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x_input:batch_xs, y_input:batch_ys})
    if (i%50==0):
        print (sess.run(cross_entropy, feed_dict={x_input:batch_xs, y_input:batch_ys}))
        print("prediction acc : ", compute_acc(mnist.test.images[:100], mnist.test.labels[:100]))

res = sess.run(y_pre, feed_dict={x_input: mnist.test.images[:examples_to_show]})

print (sess.run(tf.argmax(res, 1)))

f, a = plt.subplots(2, 10, figsize = (10, 2))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
plt.show()
sess.close()