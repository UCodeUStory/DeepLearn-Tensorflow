import tensorflow as tf

#NumPy 是一个科学计算的工具包，这里通过NumPy工具包生成模拟数据集
from numpy.random import RandomState

#定义训练数据batch的大小
batch_size = 8;
# 定义神经网络的参数,2x3矩阵，标准差1
''' 
 在训练样本之前w1数据
[[-0.8113182   1.4845988   0.06532937]
 [-2.4427042   0.0992484   0.5912243 ]]
 在训练样本之前w2数据
 [[-0.8113182 ]
 [ 1.4845988 ]
 [ 0.06532937]]
'''
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

# tf.placeholder(dtype, shape=None, name=None) 
# dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
# shape：数据形状。默认是None，就是一维值，也可以是多维，比如[2,3], [None, 3]表示列是3，行不定
# name：名称
x = tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_ = tf.placeholder(tf.float32,shape=(None,1),name='y-input')

# 矩阵相乘
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#定义损失函数的反向传播的算法

cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#通过随机数生成一个模拟数据集。0~1之间的数
rdm = RandomState(1)
#数据产生128行
dataset_size = 128   
#产生128行，每行2列的数据，X表示的是矩阵(128x2)，实际用数组表示
#由 m × n 个数aij排成的m行n列的数表称为m行n列的矩阵，简称m × n矩阵。
X = rdm.rand(dataset_size,2)

#print(X)  
'''
4.17022005e-01 就是 0.417022005  9.23385948e-02  0.0923385948e-02
[[4.17022005e-01 7.20324493e-01]
 [1.14374817e-04 3.02332573e-01]
                .
                .
                .
 [4.02024891e-04 9.76759149e-01]
 [3.76580315e-01 9.73783538e-01]
 [6.04716101e-01 8.28845808e-01]]
'''

# 定义规则来给出样本的标签，这里所有x1+x2<1的样例都被认为是正样本（比如合格零件）
# 而其他为负样本。 在这里0来表示负样本，1来表示正样本
# 生成和X行数相同，列数为1的矩阵，这个矩阵里面是1和0为X矩阵每个做标记用
Y = [[int(x1+x2 <1)] for (x1,x2) in X]

print(Y)


# 创建一个会话来运行TensorFlow程序
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    #初始化变量。
    sess.run(init_op)
    print (sess.run(w1))
    print (sess.run(w2))

    #设定训练的轮数 ,9000次就能够确定参数 
    STEPS = 20000
    for i in range(STEPS):
        #每次选取batch_size个样本进行训练，这里一轮8个
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size,dataset_size)

        #通过选取的样本训练神经网络  并更新参数。train_step
        #feed_dict 里面的值可以是一个数据赋值，也可以是一组数据赋值,具体按照上面定义来
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})

        if i %1000 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})

            print("After %d trainning step(s),cross entropy on all data is %g" %(i,total_cross_entropy))
        
    print(sess.run(w1))
    print(sess.run(w2))



   