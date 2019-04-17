import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# from __future__ import print_function

#  添加神经层的函数，四个参数：输入值、输入的大小、输出的大小、激励函数
# 我们默认激励函数是空
"""
所以，这里的表示方式是： input * weights 
假如，输入层的结点个数是2，隐层是3
input=[n*2]  ,weihts=[2*3] ,bias=[1,3]
input*weigths=[n,3] + bias=[1,3] ，这样的矩阵维度相加的时候，python会执行它的广播机制
so,这一层的输出的维度是 [n,3]
"""
def add_layer(inputs, in_size, out_size, activation_function = None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    return outputs

"""
构造一个3层的网络
输入层一个结点，隐层3个结点，输出层一个结点
输入层的维度是[n,1]
隐层的维度是  [1,10]
输出层的维度是[10,1]
so,
权值矩阵的维度是：
weight1=[1,10]
bais1=[10,1]
weight2=[10,1]
bais2=[1,1]
"""

x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

prediction = add_layer(l1, 10, 1, activation_function= None)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
plt.scatter(x_data, y_data)
for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i % 50 == 0:
        prediction_value = sess.run(prediction, feed_dict={xs: x_data, ys: y_data})
        # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        plt.plot(x_data, prediction_value)
plt.show()