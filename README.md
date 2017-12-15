# DeepLearn-Tensorflow
开始迈向人工智能，学习主流的深度学习框架Tensorflow

## 概念
人工智能、机器学习、深度学习都什么区别？

一、按照提出时间顺序分

1. 1956年,几个计算机科学家相聚在达特茅斯会议（Dartmouth Conferences）,提出了“人工智能（Artificial Intelligence 简称AI）”的概念。简单解释为机器赋予人的智能；

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

     - Bazel ：谷歌开元的自动化构建工具，相比MakeFile、Ant、Maven，Bazel在速度、可伸缩性、灵活性、以及对不同程序语言和平台的支持都要更出色，Tensorflow本身就是由Bazel来编译的
       Bazel 对python支持的遍历方式三种,py_binary,py_library,py_test,分别是编译成可执行文件，函数库，测试程序

4. TensorFlow入门开始
   - 1. TensorFlow 计算模型--计算图
        TensorFlow Tensor和Flow组成，Tensor是张量的意思，Flow是流的意思，TensorFlow是一个通过计算图的形式来表述计算的编译系统。每一个计算都是计算图上的一个节点，节点之间的边描述了计算之间的依赖关系。
         
   - 2. TensorFlow程序一般可以分为两个阶段
        在第一个阶段需要定义的计算图中所有的计算,第二个阶段就是执行计算