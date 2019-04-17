import tensorflow as tf
import numpy
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=3)

def add_layer(inputs, in_size, out_size, layer_name, activation_name = None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    wx_plus_b = tf.add(tf.matmul(inputs, weights), biases)
    # 使用dropout的地方，丢掉一部分的 wx_plus_b
    wx_plus_b = tf.nn.dropout(wx_plus_b, keep_prob)
    if activation_name == None:
        outputs = wx_plus_b
    else:
        outputs = activation_name(wx_plus_b)
    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])
ys = tf.placeholder(tf.float32, [None, 10])

l1 = add_layer(xs, 64, 50, 'l1', activation_name=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, 'l2', activation_name=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('logs/train', sess.graph)
test_writer = tf.summary.FileWriter('logs/test', sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

for i in range(500):
    sess.run(train_step, feed_dict={xs:X_train, ys:y_train, keep_prob:0.5})
    if i%50 == 0:
        train_result = sess.run(merged, feed_dict={xs:X_train, ys:y_train, keep_prob:1})
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)