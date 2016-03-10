title: 卷积神经网络(CNN)学习笔记2：模型训练
date: 2016-03-03 13:41:05
tags: [Machine Learning, Deep Learning, CNN]
categories: Machine Learning
---

[**上篇博文**](http://www.jeyzhang.com/cnn-learning-notes-1.html)主要对CNN的基本网络结构及连接方式做了简单的介绍，还介绍了一个界内经典的**`LeNet-5`**模型。下面重点介绍CNN模型的训练过程/参数学习，在阅读本文之前，最好需要有以下方面的预备知识：

- 神经网络基础（网络结构，前向/后向传播方式，激活函数等）；
- 基础的最优化求解方法（梯度法，牛顿法等）；
- 机器学习基础
----------

神经网络模型常用于处理有监督学习的问题，例如分类问题，CNN也不例外。模型需要一些有标注的数据进行训练，训练过程中主要涉及到网络的**前向传播**和**反向传播**计算，前向传播体现了特征信息的传递，而反向传播则是体现误差信息对模型参数的矫正。

### CNN前向传播 ###

与普通的神经网络的前向传播过程一样。用 \\( l \\) 表示当前层，\\( x^{l} \\) 表示当前层的输出，\\( W^{l} \\) 和 \\( b^{l} \\) 分别表示当前层的权值和偏置，则前向传播可以用下面的公式表示：

$$ x^{l} = f\left( u^{l}\right), \ with \; u^{l} = W^{l}x^{l-1} + b^{l} $$

其中 \\(f\left( \right)\\) 函数为激活函数，可以选择`sigmod`或者`tanh`等函数。

对于卷积层，其前向传播如下图：

![](http://i.imgur.com/4mCbV3d.png)

### CNN反向传播 ###

#### 代价函数 ####

代价函数（或损失函数）有较多形式，常用的有平方误差函数，交叉熵等。这里我们用平方误差函数作为代价函数，公式如下：

$$ E^{n} = \dfrac {1} {2}\sum \_{k=1}^{c}\left( t\_{k}^{n} - y\_{k}^{n}\right) ^{2} = \dfrac {1} {2}||t^{n} - y^{n}||\_{2}^{2}$$

以上公式描述了样本 \\( n \\) 的训练误差，其中 \\( c \\) 为输出层节点的个数（通常就是最终的分类类别数目），\\( t \\) 是训练样本的正确结果，\\( y \\) 是网络训练的输出结果。

#### BP反向传播 ####

基本的反向传播与BP神经网络类似，首先，简单回顾一下BP神经网络中的反向传播计算过程：

权值参数调整的方向如下公式：

$$ \Delta W^{l} = -\eta \dfrac {\partial E} {\partial W^{l}}, \ \ \dfrac {\partial E} {\partial W^{l}} = x^{l-1}(\delta ^{l})^{T} $$

其中，\\( \eta \\) 为学习率。

$$ \dfrac {\partial E} {\partial b} = \dfrac {\partial E} {\partial u} \dfrac {\partial u} {\partial b} = \dfrac {\partial E} {\partial u} = \delta $$

其中，\\( \delta \\) 称之为**敏感度**，也就是**误差度**。 \\( \delta \\)的计算方式如下：

$$ \delta ^{L} = f'(u^{L})\circ (y^{n} - t^{n}) $$

$$ \delta ^{l} = (W^{l+1})^{T}\circ f'(u^{l}) $$

其中，\\( L \\) 表示网络的最后一层，\\( l \\) 表示网络的其他层，\\( \circ \\) 表示点乘。 以上的两个公式反映了误差由网络的最后一层逐步向前传递的计算过程。

#### 特殊的反向传播 ####

由于CNN中有不同类型的层级，并且层级之间的连接关系有可能是不确定的（如LeNet-5网络中S2层到C3层）。所以，有几个情形下的反向传播比较特别：

- **情况一**：当前为Pooling层，前一层是卷积层；
- **情况二**：当前为卷积层，前一层是Pooling层；
- **情况三**：当前层与前一层的连接关系不确定（**？尚不理解？**）；

#### 情况一：当前为Pooling层，前一层是卷积层####

![](http://i.imgur.com/TcfFA3Y.png)

![](http://i.imgur.com/zsFtAhI.png)

![](http://i.imgur.com/iUghGWY.png)

其中，`Kronecker`乘积的计算如下：

![](http://i.imgur.com/zH179Jk.png)

#### 情况二：当前为卷积层，前一层是Pooling层 ####

![](http://i.imgur.com/2R5A7VT.png)

![](http://i.imgur.com/tm4daZv.png)

以上的矩阵1和矩阵2进行卷积操作时，需要将矩阵2先**水平翻转**，然后再**垂直翻转**；最后在矩阵1上进行`卷积操作`（和前向传播时类似）。

![](http://i.imgur.com/Q5pSR1x.png)

![](http://i.imgur.com/fm30a99.png)

#### 情况三：当前层与前一层的连接关系不确定 ####

个人理解，当前层与前一层的连接关系不确定时，反向传播与传统的BP算法类似，只不过更新的是局部连接的那些值。所以需要提前记录当前层的神经元与前一层的哪些元素是连接的。

----------

本文结束，感谢欣赏。

**欢迎转载，请注明本文的链接地址：**

http://www.jeyzhang.com/cnn-learning-notes-2.html

**参考资料**

[卷积神经网络全面解析](http://www.moonshile.com/post/juan-ji-shen-jing-wang-luo-quan-mian-jie-xi)

[CNN卷积神经网络反向传播机制的理解](http://blog.csdn.net/vintage_1/article/details/17253997)


