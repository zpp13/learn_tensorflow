import tensorflow as tf

var = tf.Variable(0,name = 'myvar')# 变量

con_var = tf.constant(1)# 常量

new_var = tf.add(var, con_var)# 加法

init = tf.global_variables_initializer()# 初始化

sess = tf.Session()

sess.run(init)

print (sess.run(var))
print (sess.run(con_var))
print (sess.run(new_var))

sess.close()