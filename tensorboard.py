import tensorflow as tf

def add_layer(inputs, in_size, out_size, activation_function = None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal([in_size, out_size]), name = 'W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name = 'b')
        with tf.name_scope('wx_plus_b'):
            wx_plus_b = tf.add(tf.matmul(inputs, weights) + biases)
        if activation_function == None:
            outputs = wx_plus_b
        else:
            outputs = activation_function(wx_plus_b)
        return  outputs
# 这个就是在tensorboard上可视化的时候的区别：
# 使用with tf.name_scope('inputs')可以将xs和ys包含进来
# 形成一个大的图层，图层的名字就是with tf.name_scope()方法里的参数

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

prediction = add_layer(l1, 10, 1, activation_function= None)
prediction2 = add_layer(prediction, 1, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction2), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)