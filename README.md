# DeepLearn-Tensorflow
开始迈向人工智能，学习主流的深度学习框架Tensorflow

## 概念
人工智能、机器学习、深度学习都什么区别？

一、按照提出时间顺序分

1. 1956年,几个计算机科学家相聚在达特茅斯会议（Dartmouth Conferences）,提出了“人工智能（Artificial Intelligence 简称AI）”的概念。简单解释为机器赋予人的智能；

2. Langley（1996) 定义的机器学习是“机器学习是一门人工智能的科学，该领域的主要研究对象是人工智能；
----机器学习是实现人工智能的一种方法

3. 深度学习是机器学习研究中的一个新的领域，其动机在于建立、模拟人脑进行分析学习的神经网络，它模仿人脑的机制来解释数据，例如图像，声音和文本；深度学习——一种实现机器学习的技术，深度学习使得机器学习能够实现众多的应用，并拓展了人工智能的领域范围。

## Tensorflow （Google深度学习框架）
1. 安装

      [官方文档](https://www.tensorflow.org/install/)

2. python3

      [学习地址](http://www.runoob.com/python3/python3-tutorial.html)
3. TensorFlow中两个重要的工具

     - ProTocol Buffer:谷歌开发的处理结构化数据的工具。它独立于语言，独立于平台。google 提供了多种语言的实现：java、c#、c++、go 和 python，每一种实现都包含了相应语言的编译器以及库文件。由于它是一种二进制的格式，比使用 xml 进行数据交换快许多。可以把它用于分布式应用之间的数据通信或者异构环境下的数据交换。作为一种效率和兼容性都很优秀的二进制数据传输格式，可以用于诸如网络传输、配置文件、数据存储等诸多领域。

     - Bazel ：谷歌开元的自动化构建工具，相比MakeFile、Ant、Maven，Bazel在速度、可伸缩性、灵活性、以及对不同程序语言和平台的支持都要更出色，Tensorflow本身就是由Bazel来编译的
       Bazel 对python支持的遍历方式三种,py_binary,py_library,py_test,分别是编译成可执行文件，函数库，测试程序

4. TensorFlow入门开始
   - 1. TensorFlow 计算模型--计算图
        TensorFlow Tensor和Flow组成，Tensor是张量的意思，Flow是流的意思，TensorFlow是一个通过计算图的形式来表述计算的编译系统。每一个计算都是计算图上的一个节点，节点之间的边描述了计算之间的依赖关系。
         
   - 2. TensorFlow程序一般可以分为两个阶段
        在第一个阶段需要定义的计算图中所有的计算,第二个阶段就是执行计算
  
   - 3. TensorFlow数据模型--张量
        张量是TensorFlow管理数据的形式,在TensorFlow 中所有数据都是通过张量的形式来表示，张量可以简单的理解成多维数组
        
            import tensorflow as tf
            a = tf.constant([1.0,2.0],name="a") #constant是一个计算，计算结果为一个张量，保存在变量a中
            b = tf.constant([2.0,3.0],name="b")
            result = tf.add(a,b,name="add")
            print result

            输出结果 Tensor（"add:0",shape=(2,),dtype=float32）
           
           Tensorflow 计算结果不是一个值，而是一个张量的结构；上面打印结果表示这个张量的名字为”add：0“(这是张量的唯一标识符)，shape表示维度信息，上面表示的则就是1维数组，长度是2，第三个属性也就是类型，每个张量会有唯一类型，不同类型不能直接操作

           例如:

            import tensorflow as tf
            a = tf.constant([1,2],name="a")
            b = tf.constant([2.0,3.0],name="b")
            result = a + b
            ## 这里程序就会报错
            指定类型避免报错
            a = tf.constant([1,2],name="a",dtype=tf.float32) 这样子就不报错了

   - 4. 张量的使用
           1 保存中间结果,增加可读性
           
           2 当计算图构造完成之后，张量可以用来获取计算结果，也就是得到真实数据，虽然张量本身没有存储具体的数字，但是可以通过会话就可以得到这些具体的数字了
           

          比如上面的代码就可以使用 tf.Session().run(result)语句来计算结果
   - 5. TensorFlow运行模型-- 会话
         前面介绍了TensorFlow是如何组织数据和运算的。接下来我们需要通过tensorFlow中的会话来执行定义好的运算

         ** 会话拥有并管理TensorFlow运行时的所有资源，当所有计算完之后必须关闭，否者，就会出现内存泄露的问题

         ** 会话的种类一般分为两种
           - 需要明确的调用会话的生成函数和关闭函数
              形式如：

            sess = tf.Session()
            sess.run(...)
            sess.close()

           ** 上面如果在执行过程中发现了异常，就会到时sess.close()不会被执行,所以python中可以用with语句来管理,这样也就不需要手动调用close，python会自动调用

           
            with tf.Session() as sess:
                  sess.run(...)

   - 6. TensorFlow实现神经网络
           提取特征向量
           卡迪尔坐标系
           -  1.所谓神经网络结构就是指不同神经元之间连接的结构
           -  2.一个神经元可以有多个输入，但只有一个输出，一个神经元的输出可以作为下一个神经元的输入
           - 3.前向传播算法
           - 4.全连接神经网络前向传播算法：相邻两层之间任意两个节点都有链接
           - 5.除了输入层之外的节点都代表了一个神经元的结构，一个神经元也叫一个节点，每一个节点的取值都是输入层取值的加权和(X1*W1+X2*W2..)
           - 6.前向传播算法可以理解为向量的乘法
           
           - [前向传播算法计算图解](https://github.com/UCodeUStory/DeepLearn-Tensorflow/blob/master/vector.jpeg)
           -  [矩阵和方程转换详解](http://www.ruanyifeng.com/blog/2015/09/matrix-multiplication.html)

         变量的声明：tf.Variable(tf.random_normal([2,3],stddev=2))//生成变量，必须初始值
         初始值得方法有很多，这里使用随机数生成函数，产生一个2x3矩阵，均值为0（mean可以指定均值，默认0），标准差2

   - 7. 通过TensorFlow训练神经网络模型

        1.监督式学习：监督式学习的重要思想就是，在已知答案的标注数据集上，模型给出的数据结果要尽量接近真实答案.通过调整神经网络中的参数对训练数据的拟合,可以使得模型对未知数据样本达到预测能力

        2.神经网络优化算法中，反向传播算法是最常用的方式

        总结：训练神经网络需要3个步骤
        - 1定义神经网络结构和前向传播的结果
        - 2定义损失函数以及反向传播优化算法
        - 3生成会话，并在训练数据上反复运行反向传播优化算法

5. 优化神经网络（更好的对未知样本预测）
       - 线性模型的局限性，生活中大多数问题是不能线性的解决
       - 线性函数与非线性函数：指的就是一次函数y= ax+b，其他x非一次幂的函数都是非线性的函数，曲线
       - 深度学习:一类通过多层非线性变换对高复杂性数据建模算法的集合，因为深层神经网络是实现“多层非线性变化的常用方法”，所以在实际当中，可以认为深度学习就是深层神经网络的代名词
       - 深度学习的特性：多层和非线性
       - 激活函数去实现非线性化：
          前面神经元的输出都是所有输入的加权和，导致整个神经网络是个线性模型，如果每隔神经元的输出通过一个非线性函数，name整个神经网络的模型也就不再是线性的了，这个非线性的函数就是激活函数。
       - 偏置项：
          偏置项是神经网络中非常常用的一种结构。偏执项可以表达为输入项永远为1的节点
       - [常用激活函数]()
       - [加上激活函数和偏置项的算法公式]()
   - 2. 合理设定神经网络优化目标（也就是定损函数）、常用定损函数


 ## 代码
 - [Day01参数变量的使用](https://github.com/UCodeUStory/DeepLearn-Tensorflow/blob/master/python/Day01_Variable.py)
 - [Day02完整的神经网络训练](https://github.com/UCodeUStory/DeepLearn-Tensorflow/blob/master/python/Day02_SimpleTrain.py)


