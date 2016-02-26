title: TensorFlow学习笔记2
date: 2016-01-31 13:02:56
tags: [TensorFlow, Machine Learning, Deep Learning, Artificial Intelligence]
categories: Machine Learning
---

[上篇博文](http://www.jeyzhang.com/tensorflow-learning-notes.html)主要是TensorFlow的一个简单入门，并介绍了如何实现Softmax Regression模型，来对MNIST数据集中的数字手写体进行识别。

然而，由于Softmax Regression模型相对简单，识别准确率并不算很高。下面将针对MNIST数据集，构建更加复杂精巧的模型，以进一步提高识别准确率。

----------

## 深度学习模型 ##

TensorFlow很适合用来进行大规模的数值计算，其中，也包括实现和训练深度神经网络模型。下面将介绍TensorFlow中模型的基本组成部分，同时将构建一个CNN模型来对MNIST数据集中的数字手写体进行识别。

### 基本设置 ###

在我们构建模型之前，我们首先加载MNIST数据集，然后开启一个TensorFlow会话(session)。

### 加载MNIST数据集 ###

TensorFlow中已经有相关脚本，来自动下载和加载MNIST数据集。（脚本会自动创建MNIST_data文件夹来存储数据集）。下面是脚本程序：

	import input_data
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

这里`mnist`是一个轻量级的类文件，存储了NumPy格式的训练集、验证集和测试集，它同样提供了数据中mini-batch迭代的功能。

### 开启TensorFlow会话 ###

TensorFlow后台计算依赖于C++的高效性，


----------

本文结束，感谢欣赏。

**欢迎转载，请注明本文的链接地址：**

http://www.jeyzhang.com/tensorflow-learning-notes.html

**参考资料**

