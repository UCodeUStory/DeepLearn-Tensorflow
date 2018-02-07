import tensorflow as tf

# 输出Hello, TensorFlow!
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print sess.run(hello)

# 计算 10 + 32 并输出
a = tf.constant(10)
b = tf.constant(32)
print sess.run(a+b)

# 把一个1*2的矩阵和一个2*1的矩阵相乘并输出
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
product = tf.matmul(matrix1, matrix2)
result = sess.run(product)
print result

sess.close()