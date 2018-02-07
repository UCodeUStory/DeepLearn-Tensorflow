import tensorflow as tf

#NumPy 是一个科学计算的工具包，这里通过NumPy工具包生成模拟数据集
from numpy.random import RandomState

#定义训练数据batch的大小
batch_size = 8;
# 定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

x = tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_ = tf.placeholder(tf.float32,shape=(None,1),name='y-input')

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#定义损失函数的反向传播的算法

cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#通过随机数生成一个模拟数据集。
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)

# 定义规则来给出样本的标签，这里所有x1+x2<1的样例都被认为是正样本（比如合格零件）
# 而其他为负样本。 在这里0来表示负样本，1来表示正样本

Y = [[int(x1+x2 <1)] for (x1,x2) in X]

''' 
 在训练样本之前w1数据
[[-0.8113182   1.4845988   0.06532937]
 [-2.4427042   0.0992484   0.5912243 ]]
 在训练样本之前w2数据
 [[-0.8113182 ]
 [ 1.4845988 ]
 [ 0.06532937]]
'''
# 创建一个会话来运行TensorFlow程序
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    #初始化变量。
    sess.run(init_op)
    print (sess.run(w1))
    print (sess.run(w2))

    #设定训练的轮数
    STEPS = 5000
    for i in range(STEPS):
        #每次选取batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size,dataset_size)

        #通过选取的样本训练神经网络并更新参数。
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})

        if i %1000 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})

            print("After %d trainning step(s),cross entropy on all data is %g" %(i,total_cross_entropy))
        
    print(sess.run(w1))
    print(sess.run(w2))



   