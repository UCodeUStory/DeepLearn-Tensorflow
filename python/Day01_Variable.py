import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

#声明w1,w2两个变量。这里还通过seed参数设定随机种子
#这样可以保证每一次运行得到的结果是一样的
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

x= tf.constant([[0.7,0.9]])

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

sess = tf.Session()
#必须调用初始化函数,tensorflow.python.framework.errors_impl.FailedPreconditionError: Attempting to use uninitialized value Variable
#sess.run(w1.initializer)
#sess.run(w2.initializer)

# 也可以统一初始化
#init_op = tf.initialize_all_variables()
init_op = tf.global_variables_initializer()
sess.run(init_op)
##变量的声明函数是一种运算，运算结果是一个张量，所以变量是张量的一种

#tf.global_variables tf.all_variables
for variable in tf.global_variables():  
    print(variable)

print(sess.run(w1))
print(sess.run(w2))
print(sess.run(y))
sess.close()


