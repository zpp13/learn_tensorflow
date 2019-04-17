import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.3 + 0.1# 初始化随机的数据集

weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))#随机初始化权重偏置

y = x_data*weights + biases
loss = tf.reduce_mean(tf.square(y-y_data))# 计算损失

optimizer = tf.train.GradientDescentOptimizer(0.5)# 学习了
train = optimizer.minimize(loss)# 梯度下降优化
"""
到目前为止, 我们只是建立了神经网络的结构, 还没有使用这个结构. 
在使用这个结构之前, 我们必须先初始化所有之前定义的Variable, 所以这一步是很重要的
"""
init = tf.global_variables_initializer()#

sess = tf.Session()# 创建对话
sess.run(init)

print (x_data)
print (y_data)

for step in range(200):
    sess.run(train)
    if step%20 == 0:
        print (step, sess.run(weights), sess.run(biases))

"""
0 [-0.00586408] [ 0.41099712]
20 [ 0.22088484] [ 0.14566249]
40 [ 0.28280813] [ 0.10992255]
60 [ 0.29626417] [ 0.10215619]
80 [ 0.29918817] [ 0.10046856]
100 [ 0.29982358] [ 0.10010182]
120 [ 0.29996166] [ 0.10002214]
140 [ 0.2999917] [ 0.1000048]
160 [ 0.29999822] [ 0.10000104]
180 [ 0.29999959] [ 0.10000023]
这个结果基本是对的
"""