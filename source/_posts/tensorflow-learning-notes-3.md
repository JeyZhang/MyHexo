title: TensorFlow学习笔记3：词向量
date: 2016-03-16 21:24:38
tags: [TensorFlow, Machine Learning, Deep Learning, Artificial Intelligence]
categories: Machine Learning
---

[**上篇博文**](http://www.jeyzhang.com/tensorflow-learning-notes-2.html)讲了如何构建一个简单的CNN模型，并运行在MNIST数据集上。下面讲述一下如何在TensorFlow中生成词向量(Word Embedding)，使用的模型来自[Mikolov et al](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)。

本文的目录如下：

- 解释使用连续词向量的原因；
- 词向量模型的原理及训练过程；
- 在TensorFlow中实现模型的简单版本，并给出优化的方法；


TensorFlow实现了两个版本的模型：[简单版](https://www.tensorflow.org/code/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)和[正式版](https://www.tensorflow.org/code/tensorflow/models/embedding/word2vec.py)。如果想看源码的，可以直接下载。

----------

### 为什么要使用Word Embedding ###

在信号处理领域，图像和音频信号的输入往往是表示成高维度、密集的向量形式，在图像和音频的应用系统中，如何对输入信息进行编码(Encoding)显得非常重要和关键，这将直接决定了系统的质量。然而，在自然语言处理领域中，传统的做法是将词表示成离散的符号，例如将 [cat] 表示为 [Id537]，而 [dog] 表示为 [Id143]。**这样做的缺点在于，没有提供足够的信息来体现词语之间的某种关联**，例如尽管cat和dog不是同一个词，但是却应该有着某种的联系（如都是属于动物种类）。由于这种一元表示法(One-hot Representation)使得词向量过于稀疏，所以往往需要大量的语料数据才能训练出一个令人满意的模型。而Word Embedding技术则可以解决上述传统方法带来的问题。

![](http://i.imgur.com/dHTf4Gq.png)

**向量空间模型(Vector space models, VSMs)**将词语表示为一个连续的词向量，并且语义接近的词语对应的词向量在空间上也是接近的。VSMs在NLP中拥有很长的历史，但是所有的方法在某种程度上都是基于一种**[分布式假说](https://en.wikipedia.org/wiki/Distributional_semantics#Distributional_Hypothesis)**，该假说的思想是**如果两个词的上下文(context)相同，那么这两个词所表达的语义也是一样的**；换言之，两个词的语义是否相同或相似，取决于两个词的上下文内容，上下文相同表示两个词是可以等价替换的。

基于分布式假说理论的词向量生成方法主要分两大类：**计数法**(count-based methods, e.g. [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis))和**预测法**(predictive methods, e.g. [neural probabilistic language models](http://www.scholarpedia.org/article/Neural_net_language_models))。[Baroni等人](http://clic.cimec.unitn.it/marco/publications/acl2014/baroni-etal-countpredict-acl2014.pdf)详细论述了这两种方法的区别，简而言之，计数法是在大型语料中统计词语及邻近的词的共现频率，然后将之为每个词都映射为一个稠密的向量表示；预测法是直接利用词语的邻近词信息来得到预测词的词向量（词向量通常作为模型的训练参数）。

**`Wrod2vec`**是一个典型的预测模型，用于高效地学习Word Embedding。实现的模型有两种：**连续词袋模型(CBOW)**和**Skip-Gram模型**。算法上这两个模型是相似的，只不过CBOW是从输入的上下文信息来预测目标词(例如利用 [the cat sits on the] 来预测 [mat] )；而skip-gram模型则是相反的，从目标词来预测上下文信息。一般而言，这种方式上的区别使得CBOW模型更适合应用在小规模的数据集上，能够对很多的分布式信息进行平滑处理；而Skip-Gram模型则比较适合用于大规模的数据集上。

下面重点将介绍Skip-Gram模型。

### 噪声-对比(Noise-Contrastive)训练 ###

基于神经网络的概率语言模型通常都是使用**[最大似然估计](https://en.wikipedia.org/wiki/Maximum_likelihood)**的方法进行训练的，通过Softmax函数得到在前面出现的词语 \\( h \\) (`history`)的情况下，目标词 \\( w\_{t} \\) (`target`)出现的最大概率，数学表达式如下：

![](http://i.imgur.com/vpOKwSG.png)

其中，\\( score(w\_t, h) \\) 为词 \\(w\_t\\) 和上下文 \\(h\\) 的 [兼容程度]。上式的对数形式如下：

![](http://i.imgur.com/jG5Rppa.png)

理论上可以根据这个来建立一个合理的模型，但是现实中目标函数的计算代价非常昂贵，这是因为在训练过程中的每一步，我们都需要计算词库 \\(w'\\) 中其他词在当前的上下文环境下出现的概率值，这导致计算量十分巨大。

![](http://i.imgur.com/Ck90mom.png)

然而，对于word2vec中的特征学习，可以不需要一个完整的概率模型。CBOW和Skip-Gram模型在输出端使用的是一个二分类器(即Logistic Regression)，来区分目标词和词库中其他的 \\(k\\) 个词。下面是一个CBOW模型的图示，对于Skip-Gram模型输入输出是倒置的。

![](http://i.imgur.com/KnqFhUD.png)

此时，最大化的目标函数如下：

![](http://i.imgur.com/g4PPKUW.png)

其中，\\( Q\_\theta(D=1 | w, h) \\) 为二元逻辑回归的概率，具体为在数据集 \\(D\\) 中、输入的embedding vector \\( \theta \\)、上下文为 \\( h \\) 的情况下词语 \\(w\\) 出现的概率；公式后半部分为 \\(k\\) 个从 [噪声数据集] 中随机选择 \\(k\\) 个对立的词语出现概率(log形式)的期望值（即为[Monte Carlo average](https://en.wikipedia.org/wiki/Monte_Carlo_integration)）。

可以看出，目标函数的意义是显然的，即尽可能的 [分配(assign)] 高概率给真实的目标词，而低概率给其他 \\( k \\) 个 [噪声词]，这种技术称为**[负采样(Negative Sampling)](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)**。同时，该目标函数具有很好的数学意义：**即在条件限制(训练时间)的情况下尽可能的逼近原有的Softmax函数（选择 \\( k \\) 个 [噪声点] 作为整个 [噪声数据] 的代表）**，这样做无疑能够大大提升模型训练的速度。实际中我们使用的是类似的[噪声对比估计损失函数(noise-contrastive estimation (NCE))](http://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf)，在TensorFlow中对应的实现函数为`tf.nn.nce_loss()`。

下面看看具体是如何训练Skip-Gram模型的。

### Skip-Gram模型 ###

举个例子，假设现在的数据集如下：

	the quick brown fox jumped over the lazy dog

这个数据集中包含了词语及其上下文信息。值得说明的是，**上下文信息(Context)**是一个比较宽泛的概念，有多种不同的理解：例如，词语周边的句法结构，词语的左边部分的若干个词语信息，对应的右半部分等。这里，我们使用最原始和基本的定义，即认为**词语左右相邻的若干个词汇是该词对应的上下文信息**。例如，取左右的词窗口为1，下面是数据集中的**`(上下文信息，对应的词)`**的pairs：

	([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox), ...

Skip-Gram模型是通过输入的目标词来预测其对应的上下文信息，所以目标是通过[quick]来预测[the]和[brown]，通过[brown]来预测[quick]和[fox]... 将上面的pair转换为**`(input, output)`**的形式如下：

	(quick, the), (quick, brown), (brown, quick), (brown, fox), ...

目标函数定义如上，使用[随机梯度下降算法(SGD)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)来进行最优化求解，并且使用mini-batch方法 (通常batch_size在16到512之间)。

下面将详细剖析一下训练过程。假设在训练的第 \\(t\\) 步，目标是得到上面第一个实例输入 [quick] 的输出预测；我们选择`num_noise`个 [噪声点数据]，简单起见，这里`num_noise`为1，假设选择 [sheep] 作为噪声对比词。那么，此时的目标函数值如下：

![](http://i.imgur.com/tmgHXZZ.png)

目标是**更新embedding参数 \\(\theta\\) 以增大目标函数值**，更新的方式是计算损失函数对参数 \\(\theta\\) 的导数，即 \\( \frac{\partial}{\partial \theta} J\_\text{NEG} \\) (TensorFlow中有相应的函数以方便计算)，然后使得参数 \\(\theta\\) 朝梯度方向进行调整。当这个过程在训练数据集上执行多次后，产生的效果是使得输入的embedding vector的值发生改变，使得模型最终能够很好地区别目标词和 [噪声词]。

我们可以将学到的词向量进行降维(如[t-SNE降维技术](\frac{\partial}{\partial \theta} J_\text{NEG}))和可视化，通过可视化发现**连续的词向量能够捕捉到更多的语义和关联信息**；有趣的是，在降维空间中某些特定的方向表征着特定的语义信息，例如下图中的[man->women]，[king->queen]方向表示性别关系(出自[Mikolov et al., 2013](http://www.aclweb.org/anthology/N13-1090))。

![](http://i.imgur.com/vM1dtFq.png)

这也证实了连续词向量的作用，目前有非常多NLP中的任务(例如词性标注、命名实体识别等)都是使用连续词向量作为特征输入（更多可参考[Collobert et al., 2011](http://arxiv.org/abs/1103.0398)，[Turian et al., 2010](http://www.aclweb.org/anthology/P10-1040)）。

下面看看具体在TensorFlow中，是如何实现模型的创建和训练的。

### 构建模型 ###

首先，我们要定义一下**词嵌入矩阵(Embedding Matrix)**，并随机初始化。

	embeddings = tf.Variable(
	tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

噪声-对比估计的损失函数在输出的逻辑回归模型中定义，为此，需要定义词库中每个词的权值和偏置参数(称为输出层权值参数)，如下：

	nce_weights = tf.Variable(
	  tf.truncated_normal([vocabulary_size, embedding_size],
		stddev=1.0 / math.sqrt(embedding_size)))
	nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

现在我们有了这些模型参数，接下来需要定义Skip-Gram模型。简单起见，假设我们已经将语料库中的词[**整数化**]，即每个词被表示为一个整数(具体见[tensorflow/examples/tutorials/word2vec/word2vec_basic.py](https://www.tensorflow.org/code/tensorflow/examples/tutorials/word2vec/word2vec_basic.py))。Skip-Gram模型有两种输入，都是整数形式表示：一种是批量的上下文词汇，一种是目标词。我们先为这些输入创建占位符(placeholder)，之后再进行数据的填充。

	# Placeholders for inputs
	train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
	train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

我们还需要能够查找(look up)batch中的输入词对应的vector，如下：

	embed = tf.nn.embedding_lookup(embeddings, train_inputs)

现在，我们有了每个词对应的embedding，接下来使用噪声-对比策略来预测目标词：

	# Compute the NCE loss, using a sample of the negative labels each time.
	loss = tf.reduce_mean(
	  tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
	                 num_sampled, vocabulary_size))

现在，我们有了损失函数节点(loss node)，还需要利用随机梯度下降来进行优化，定义如下的优化器：

	# We use the SGD optimizer.
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

### 模型的训练 ###

模型的训练方式很简单，只需要迭代地通过`feed_dict`进行训练数据的填充，并启动一个session。

	for inputs, labels in generate_batch(...):
	  feed_dict = {training_inputs: inputs, training_labels: labels}
	  _, cur_loss = session.run([optimizer, loss], feed_dict=feed_dict)

完整的示例代码请参考[tensorflow/examples/tutorials/word2vec/word2vec_basic.py](https://www.tensorflow.org/code/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)。

### Embedding的可视化 ###

模型训练结束后，我们利用`t-SNE技术`实现学习到的embedding可视化，如下图所示：

![](http://i.imgur.com/z2VpgFz.png)

正如我们期望的那样，语义相似的词语会聚集在一起。关于word2vec更加高级的实现版本，可参考[tensorflow/models/embedding/word2vec.py](https://www.tensorflow.org/code/tensorflow/models/embedding/word2vec.py)。

### Embedding的评价：类比推理(Analogical Reasoning)###

Embedding在许多的NLP任务中都很有效果，那么如何评价Embedding的效果呢？一种简单的方式是，直接用来预测句法和语义的关联性，例如预测**`king is to queen as father is to ?`**，这称作**`Analogical Reasoning`**(By [Mikolov and colleagues](http://msr-waypoint.com/en-us/um/people/gzweig/Pubs/NAACL2013Regularities.pdf), 评价数据集可在[这里](https://www.google.com/url?q=https://word2vec.googlecode.com/svn/trunk/questions-words.txt&usg=AFQjCNHs2OomcnDRRaht8ih-rL2oHnOSwQ)下载)。

具体如何进行评价的，可以参考[正式word2vec版本](https://www.tensorflow.org/code/tensorflow/models/embedding/word2vec.py)中的`build_eval_graph()`和`eval()`函数。

评价任务的准确性依赖于模型的超参数们，为了达到最佳的效果，往往需要将评价任务建立在一个巨大的数据集上，还可能需要使用一些trick，例如数据抽样、适当的fine tuning等。

### 进一步的优化 ###

以上的Vanilla版本展示了TensorFlow的简单易用。例如，只需要调用`tf.nn.nce_loss()`就可以替换`tf.nn.sampled_softmax_loss()`。如果你有关于损失函数的新想法，也可以自己在TensorFlow中手写一个，然后使用优化器计算导数并作优化。TensorFlow的简单易用，可以帮助你快速验证自己的想法。

一旦你有了一个令人满意的模型结构，你可以针对它进行优化使其更加高效。例如，原始版本中有个不足之处是，数据读取和填充是用Python实现的，因此会相对低效。你可以自己实现一个reader，参考[数据格式要求](https://www.tensorflow.org/versions/r0.7/how_tos/new_data_formats/index.html)。对于Skip-Gram模型，我们在[这个版本](https://www.tensorflow.org/code/tensorflow/models/embedding/word2vec.py)中自定义了reader，可供参考。

如果你的模型在I/O上足够好了，但仍然想要提升效率，你可以自己编写TensorFlow Ops（[参考这里](https://www.tensorflow.org/versions/r0.7/how_tos/adding_an_op/index.html)）.[优化版本](https://www.tensorflow.org/code/tensorflow/models/embedding/word2vec_optimized.py)中提供了示例。

### 总结 ###

这篇博文介绍了word2vec模型，一个用来高效学习出word embedding的模型。我们解释了为什么word embedding是有效的，讨论了如何更加高效地训练模型以及如何在TensorFlow中去实现。

----------

本文结束，感谢欣赏。

**欢迎转载，请注明本文的链接地址：**

http://www.jeyzhang.com/tensorflow-learning-notes-3.html

**参考资料**

[TensorFlow: Vector Representation of Words](https://www.tensorflow.org/versions/r0.7/tutorials/word2vec/index.html)