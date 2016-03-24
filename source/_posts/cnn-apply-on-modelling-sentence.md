title: 卷积神经网络(CNN)在句子建模上的应用
date: 2016-03-11 10:36:35
tags: [Machine Learning, Deep Learning, CNN, Sentence Model, NLP]
categories: Machine Learning
---

之前的博文已经介绍了CNN的基本原理，本文将大概总结一下最近CNN在NLP中的句子建模（或者句子表示）方面的应用情况，主要阅读了以下的文献（[打包下载地址](http://pan.baidu.com/s/1jHfWnzg)）：

> Kim Y. Convolutional neural networks for sentence classification[J]. arXiv preprint arXiv:1408.5882, 2014.

> Kalchbrenner N, Grefenstette E, Blunsom P. A convolutional neural network for modelling sentences[J]. arXiv preprint arXiv:1404.2188, 2014.

> Hu B, Lu Z, Li H, et al. Convolutional neural network architectures for matching natural language sentences[C]//Advances in Neural Information Processing Systems. 2014: 2042-2050.

下面对文献中CNN的结构和细节进行梳理。

----------

### Kim Y's Paper ###

#### 模型结构及原理 ####

模型的结构如下：

![](http://i.imgur.com/yxoZDt9.png)

说明如下：

- **输入层**

如图所示，输入层是句子中的词语对应的word vector依次（从上到下）排列的矩阵，假设句子有 \\( n \\) 个词，vector的维数为 \\( k \\) ，那么这个矩阵就是 \\( n × k \\) 的。

这个矩阵的类型可以是静态的(static)，也可以是动态的(non static)。静态就是word vector是固定不变的，而动态则是在模型训练过程中，word vector也当做是可优化的参数，通常把反向误差传播导致word vector中值发生变化的这一过程称为**`Fine tune`**。

对于未登录词的vector，可以用0或者随机小的正数来填充。

- **第一层卷积层**

输入层通过卷积操作得到若干个`Feature Map`，卷积窗口的大小为 \\( h × k \\) ，其中 \\( h \\) 表示纵向词语的个数，而 \\( k \\) 表示word vector的维数。通过这样一个大型的卷积窗口，将得到若干个列数为1的`Feature Map`。

- **池化层**

接下来的池化层，文中用了一种称为**`Max-over-time Pooling`**的方法。这种方法就是简单地从之前一维的`Feature Map`中提出最大的值，文中解释最大值代表着最重要的信号。可以看出，这种Pooling方式可以解决可变长度的句子输入问题（因为不管`Feature Map`中有多少个值，只需要提取其中的最大值）。

最终池化层的输出为各个`Feature Map`的最大值们，即一个一维的向量。

- **全连接 + Softmax层**

池化层的一维向量的输出通过全连接的方式，连接一个Softmax层，Softmax层可根据任务的需要设置（通常反映着最终类别上的概率分布）。

最终实现时，我们可以在倒数第二层的全连接部分上使用`Dropout`技术，即对全连接层上的权值参数给予**`L2正则化`**的限制。这样做的好处是防止隐藏层单元自适应（或者对称），从而减轻过拟合的程度。

#### 实验部分 ####

**1. 数据**

实验用到的数据集如下（具体的名称和来源可以参考论文）：

![](http://i.imgur.com/8VDJDDJ.png)

**2. 模型训练和调参**

- 修正线性单元(Rectified linear units)
- 滤波器的h大小：3,4,5；对应的Feature Map的数量为100；
- Dropout率为0.5，L2正则化限制权值大小不超过3；
- mini-batch的大小为50；

这些参数的选择都是基于SST-2 dev数据集，通过网格搜索方法(Grid Search)得到的最优参数。另外，训练过程中采用随机梯度下降方法，基于shuffled mini-batches之上的，使用了Adadelta update rule(Zeiler, 2012)。

**3. 预训练的Word Vector**

这里的word vector使用的是公开的数据，即连续词袋模型(COW)在Google News上的训练结果。未登录次的vector值是随机初始化的。

**4. 实验结果**

实验结果如下图：

![](http://i.imgur.com/sNpll24.png)

其中，前四个模型是上文中所提出的基本模型的各个变种：

- **CNN-rand**: 所有的word vector都是随机初始化的，同时当做训练过程中优化的参数；
- **CNN-static**: 所有的word vector直接使用无监督学习即Google的Word2Vector工具(COW模型)得到的结果，并且是固定不变的；
- **CNN-non-static**: 所有的word vector直接使用无监督学习即Google的Word2Vector工具(COW模型)得到的结果，但是会在训练过程中被`Fine tuned`；
- **CNN-multichannel**: CNN-static和CNN-non-static的混合版本，即两种类型的输入；

**5. 结论**

- **`CNN-static`**较与**`CNN-rand`**好，**说明pre-training的word vector确实有较大的提升作用**（这也难怪，因为pre-training的word vector显然利用了更大规模的文本数据信息）；
- **`CNN-non-static`**较于**`CNN-static`**大部分要好，**说明适当的Fine tune也是有利的，是因为使得vectors更加贴近于具体的任务**；
- **`CNN-multichannel`**较于**`CNN-single`**在小规模的数据集上有更好的表现，实际上**`CNN-multichannel`**体现了一种折中思想，即既不希望Fine tuned的vector距离原始值太远，但同时保留其一定的变化空间。

值得注意的是，static的vector和non-static的相比，有一些有意思的现象如下表格：

![](http://i.imgur.com/fW6pr0p.png)

- 原始的word2vector训练结果中，`bad`对应的最相近词为`good`，原因是这两个词在句法上的使用是极其类似的（可以简单替换，不会出现语句毛病）；而在`non-static`的版本中，`bad`对应的最相近词为`terrible`，这是因为在`Fune tune`的过程中，vector的值发生改变从而更加贴切数据集（是一个情感分类的数据集），所以在情感表达的角度这两个词会更加接近；
- 句子中的**`!`**最接近一些表达形式较为激进的词汇，如`lush`等；而**`,`**则接近于一些连接词，这和我们的主观感受也是相符的。


### Kalchbrenner's Paper ###

Kal的这篇文章引用次数较高，他提出了一种名为DCNN(Dynamic Convolutional Neural Network)的网络模型，在上一篇（Kim's Paper）中的实验结果部分也验证了这种模型的有效性。这个模型的精妙之处在于Pooling的方式，使用了一种称为**`动态Pooling`**的方法。

下图是这个模型对句子语义建模的过程，可以看到底层通过组合邻近的词语信息，逐步向上传递，上层则又组合新的Phrase信息，从而使得句子中即使相离较远的词语也有交互行为（或者某种语义联系）。从直观上来看，这个模型能够通过词语的组合，提取出句子中重要的语义信息（通过Pooling），某种意义上来说，层次结构的**`feature graph`**的作用类似于一棵语法解析树。

![](http://i.imgur.com/3IbLJX4.png)

DCNN能够处理可变长度的输入，网络中包含两种类型的层，分别是**一维的卷积层**和**动态k-max的池化层(Dynamic k-max pooling)**。其中，动态k-max池化是最大化池化更一般的形式。之前LeCun将CNN的池化操作定义为一种非线性的抽样方式，返回一堆数中的最大值，原话如下：

> The max pooling operator is a non-linear subsampling function that returns the maximum of a set of values (LuCun et al., 1998).

而文中的k-max pooling方式的一般化体现在：

- pooling的结果不是返回一个最大值，而是返回k组最大值，这些最大值是原输入的一个子序列；
- pooling中的参数k可以是一个动态函数，具体的值依赖于输入或者网络的其他参数；

#### 模型结构及原理 ####

DCNN的网络结构如下图：

![](http://i.imgur.com/CNMa0VL.png)

网络中的卷积层使用了一种称之为**`宽卷积(Wide Convolution)`**的方式，紧接着是动态的k-max池化层。中间卷积层的输出即`Feature Map`的大小会根据输入句子的长度而变化。下面讲解一下这些操作的具体细节：

**1. 宽卷积**

相比于传统的卷积操作，宽卷积的输出的`Feature Map`的宽度(width)会更宽，原因是卷积窗口并不需要覆盖所有的输入值，也可以是部分输入值（可以认为此时其余的输入值为0，即填充0）。如下图所示：

![](http://i.imgur.com/YgM3Tsg.png)

图中的右图即表示宽卷积的计算过程，当计算第一个节点即\\( s\_1 \\)时，可以假使\\( s\_1 \\)节点前面有四个输入值为0的节点参与卷积（卷积窗口为5）。明显看出，狭义上的卷积输出结果是宽卷积输出结果的一个子集。

**2. k-max池化**

给出数学形式化的表述是，给定一个\\( k \\)值，和一个序列\\( p \in R^p \\)(其中\\( p ≥ k \\))，`k-max pooling`选择了序列\\( p \\)中的前\\( k \\)个最大值，这些最大值保留原来序列的次序（实际上是原序列的一个子序列）。

`k-max pooling`的好处在于，既提取除了句子中的较重要信息（不止一个），同时保留了它们的次序信息（相对位置）。同时，由于应用在最后的卷积层上只需要提取出\\( k \\)个值，所以这种方法允许不同长度的输入（输入的长度应该要大于\\( k \\)）。然而，对于中间的卷积层而言，池化的参数\\( k \\)不是固定的，具体的选择方法见下面的介绍。

**3. 动态k-max池化**

动态k-max池化操作，其中的\\( k \\)是`输入句子长度`和`网络深度`两个参数的函数，具体如下：

$$ K\_{l}=\max \left( k\_{top}, \left \lceil \frac {L-l}{L} s \right \rceil \right) $$

其中\\( l \\)表示当前卷积的层数（即第几个卷积层），\\( L \\)是网络中总共卷积层的层数；\\( k\_{top} \\)为最顶层的卷积层pooling对应的\\( k \\)值，是一个固定的值。举个例子，例如网络中有三个卷积层，\\( k\_{top} = 3\\)，输入的句子长度为18；那么，对于第一层卷积层下面的pooling参数\\( k\_{1} = 12\\)，而第二层卷积层对于的为\\( k\_{2} = 6\\)，而\\( k\_{3} = k\_{top} = 3\\)。

动态k-max池化的意义在于，从不同长度的句子中提取出相应数量的语义特征信息，以保证后续的卷积层的统一性。

**4. 非线性特征函数**

pooling层与下一个卷积层之间，是通过与一些权值参数相乘后，加上某个偏置参数而来的，这与传统的CNN模型是一样的。

**5. 多个Feature Map**

和传统的CNN一样，会提出多个Feature Map以保证提取特征的多样性。

**6. 折叠操作(Folding)**

之前的宽卷积是在输入矩阵\\( d × s \\)中的每一行内进行计算操作，其中\\(d\\)是word vector的维数，\\(s\\)是输入句子的词语数量。而**`Folding`**操作则是考虑相邻的两行之间的某种联系，方式也很简单，就是将两行的vector相加；该操作没有增加参数数量，但是提前（在最后的全连接层之前）考虑了特征矩阵中行与行之间的某种关联。

#### 模型的特点 ####

- 保留了句子中词序信息和词语之间的相对位置；
- 宽卷积的结果是传统卷积的一个扩展，某种意义上，也是n-gram的一个扩展；
- 模型不需要任何的先验知识，例如句法依存树等，并且模型考虑了句子中相隔较远的词语之间的语义信息；

#### 实验部分 ####

**1. 模型训练及参数**

- 输出层是一个类别概率分布（即softmax），与倒数第二层全连接；
- 代价函数为交叉熵，训练目标是最小化代价函数；
- L2正则化；
- 优化方法：mini-batch + gradient-based (使用Adagrad update rule, Duchi et al., 2011)

**2. 实验结果**

在三个数据集上进行了实验，分别是(1)电影评论数据集上的情感识别，(2)TREC问题分类，以及(3)Twitter数据集上的情感识别。结果如下图：

![](http://i.imgur.com/zuf2bSu.png)

![](http://i.imgur.com/6lWY7zC.png)

![](http://i.imgur.com/PX9N2JB.png)

可以看出，DCNN的性能非常好，几乎不逊色于传统的模型；而且，DCNN的好处在于不需要任何的先验信息输入，也不需要构造非常复杂的人工特征。


### Hu's Paper ###

#### 模型结构与原理 ####

**1. 基于CNN的句子建模**

这篇论文主要针对的是**句子匹配(Sentence Matching)**的问题，但是基础问题仍然是句子建模。首先，文中提出了一种基于CNN的句子建模网络，如下图：

![](http://i.imgur.com/kG7AbW3.png)

图中灰色的部分表示对于长度较短的句子，其后面不足的部分填充的全是0值(Zero Padding)。可以看出，模型解决不同长度句子输入的方法是规定一个最大的可输入句子长度，然后长度不够的部分进行0值的填充；图中的卷积计算和传统的CNN卷积计算无异，而池化则是使用Max-Pooling。

- **卷积结构的分析**

下图示意性地说明了卷积结构的作用，作者认为卷积的作用是**从句子中提取出局部的语义组合信息**，而多张`Feature Map`则是从多种角度进行提取，也就是**保证提取的语义组合的多样性**；而池化的作用是对多种语义组合进行选择，过滤掉一些置信度低的组合（可能这样的组合语义上并无意义）。

![](http://i.imgur.com/yrFS2k1.png)

**2. 基于CNN的句子匹配模型**

下面是基于之前的句子模型，建立的两种用于两个句子的匹配模型。

**2.1 结构I**

模型结构如下图：

![](http://i.imgur.com/xaP0KNV.png)

简单来说，首先分别单独地对两个句子进行建模（使用上文中的句子模型），从而得到两个相同且固定长度的向量，向量表示句子经过建模后抽象得来的特征信息；然后，将这两个向量作为一个多层感知机(MLP)的输入，最后计算匹配的分数。

这个模型比较简单，但是有一个较大的缺点：两个句子在建模过程中是完全独立的，没有任何交互行为，一直到最后生成抽象的向量表示后才有交互行为（一起作为下一个模型的输入），这样做使得句子在抽象建模的过程中会丧失很多语义细节，同时过早地失去了句子间语义交互计算的机会。因此，推出了第二种模型结构。

**2.2 结构II**

模型结构如下图：

![](http://i.imgur.com/NWvAPVr.png)

图中可以看出，这种结构提前了两个句子间的交互行为。

- **第一层卷积层**

第一层中，首先取一个固定的卷积窗口\\( k1 \\)，然后遍历 \\( S\_{x} \\) 和 \\( S\_{y} \\) 中所有组合的二维矩阵进行卷积，每一个二维矩阵输出一个值（文中把这个称作为一维卷积，因为实际上是把组合中所有词语的vector排成一行进行的卷积计算），构成Layer-2。下面给出数学形式化表述：

![](http://i.imgur.com/f3DqYsp.png)

- **第一层卷积层后的Max-Pooling层**

从而得到Layer-2，然后进行2×2的Max-pooling：

![](http://i.imgur.com/DaFv3ps.png)

- **后续的卷积层**

后续的卷积层均是传统的二维卷积操作，形式化表述如下：

![](http://i.imgur.com/Pr5Mm9n.png)

- **二维卷积结果后的Pooling层**

与第一层卷积层后的简单Max-Pooling方式不同，后续的卷积层的Pooling是一种**动态Pooling方法**，这种方法来源于参考文献[1]。

- **结构II的性质**

1. 保留了词序信息；
2. 更具一般性，实际上结构I是结构II的一种特殊情况（取消指定的权值参数）；

#### 实验部分 ####

**1. 模型训练及参数**

- 使用基于排序的自定义损失函数(Ranking-based Loss)
- BP反向传播+随机梯度下降；
- mini-batch为100-200,并行化；
- 为了防止过拟合，对于中型和大型数据集，会提前停止模型训练；而对于小型数据集，还会使用Dropout策略；
- Word2Vector：50维；英文语料为Wikipedia(~1B words)，中文语料为微博数据(~300M words)；
- 使用ReLu函数作为激活函数；
- 卷积窗口为3-word window；
- 使用Fine tuning；

**2. 实验结果**

一共做了三个实验，分别是(1)句子自动填充任务，(2)推文与评论的匹配，以及(3)同义句识别；结果如下面的图示：

![](http://i.imgur.com/wLIUAHW.png)

![](http://i.imgur.com/fO0Xhnj.png)

![](http://i.imgur.com/qRfsoB0.png)

其实结构I和结构II的结果相差不大，结构II稍好一些；而相比于其他的模型而言，结构I和结构II的优势还是较大的。

----------

本文结束，感谢欣赏。

**欢迎转载，请注明本文的链接地址：**

http://www.jeyzhang.com/cnn-apply-on-modelling-sentence.html

**参考文献**

[1] R. Socher, E. H. Huang, and A. Y. Ng. Dynamic pooling and unfolding recursive autoencoders for paraphrase detection. In Advances in NIPS, 2011.

**推荐资料**

[A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)

[Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)

[Kim Y's Implement: Convolutional Neural Networks for Sentence Classification](https://github.com/yoonkim/CNN_sentence)



