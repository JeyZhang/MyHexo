title: TensorFlow中遇到的问题及解决方法
date: 2016-03-25 22:36:46
tags: [TensorFlow, Machine Learning, Deep Learning, Artificial Intelligence]
categories: Machine Learning
---

本文记录一下自己在使用TensorFlow的过程中，遇到的较为困扰的问题及最终的解决方法。

----------

### Q1. 如何查看TensorFlow中Tensor, Variable, Constant的值？ ###

TensorFlow中的许多方法返回的都是一个Tensor对象。在Debug的过程中，我们发现只能看到Tensor对象的一些属性信息，无法查看Tensor具体的输出值；而对于Variable和Constant，我们很容易对其进行创建操作，但是如何得到它们的值呢？

假设`ts`是我们想要查看的对象(Variable / Constant / 0输入的Tensor)，运行

	ts_res = sess.run(ts)
	print(ts_res)

其中，`sess`为之前创建或默认的`session`. 运行后将得到一个`narray`格式的`ts_res`对象，通过`print`函数我们可以很方便的查看其中的内容。

但是，如果`ts`是一个有输入要求的Tensor，需要在查看其输出值前，填充(feed)输入数据。如下（假设ts只有一种输入）：

	input = ××××××  # the input data need to feed
	ts_res = sess.run(ts, feed_dict=input)
	print(ts_res)

其他要求多种输入的Tensor类似处理即可。

### Q2. 模型训练完成后，如何获取模型的参数？ ###

模型训练完成后，通常会将模型参数存储于/checkpoint/×××.model文件(当然文件路径和文件名都可以更改，许多基于TensorFlow的开源包习惯将模型参数存储为model或者model.ckpt文件)。那么，在模型训练完成后，如何得到这些模型参数呢？

需要以下两个步骤：

**Step 1: 通过tf.train.Saver()恢复模型参数** 

运行

	saver = tf.train.Saver()

通过`saver`的`restore()`方法可以从本地的模型文件中恢复模型参数。大致做法如下：

	# your model's params
	# you don't have to initialize them

	x = tf.placeholder(tf.float32)
	y = tf.placeholder(tf.float32)
	W = tf.Variable(...)
	b = tf.Variable(...)

	y_ = tf.add(b, tf.matmul(x, w))

	# create the saver

	saver = tf.train.Saver()

	# creat the session you used in the training processing
	# launch the model

	with tf.Session() as sess:
	  # Restore variables from disk.
	  saver.restore(sess, "/your/path/model.ckpt")
	  print("Model restored.")
	  # Do some work with the model, such as do a prediction
	  pred = sess.run(y_, feed_dict={batch_x})
	  ...

有关TensorFlow中变量的创建、存储及恢复操作，详细见[API文档](http://tensorflow.org/how_tos/variables/index.md).

**Step 2: 通过tf.trainable\_variables()得到训练参数**

tf.trainable\_variables()方法将返回模型中所有可训练的参数，详细见[API文档](https://www.tensorflow.org/versions/r0.7/api_docs/python/state_ops.html#trainable_variables)。类似于以下的变量参数不会被返回：

	tf_var = tf.Variable(0, name="××××××", trainable=False)

还可以通过`Variable`的`name`属性过滤出需要查看的参数，如下：

	var = [v for v in t_vars if v.name == "W"]


（不断更新中...）


----------

本文结束，感谢欣赏。

**欢迎转载，请注明本文的链接地址：**

http://www.jeyzhang.com/problems-with-solution-in-tensorflow.html

**参考资料**

[Tensorflow: How to restore a previously saved model (python)](https://stackoverflow.com/questions/33759623/tensorflow-how-to-restore-a-previously-saved-model-python)

[Get original value of Tensor in Tensorflow](https://stackoverflow.com/questions/34172622/get-original-value-of-tensor-in-tensorflow)

[Get the value of some weights in a model trained by TensorFlow](https://stackoverflow.com/questions/36193553/get-the-value-of-some-weights-in-a-model-trained-by-tensorflow)


