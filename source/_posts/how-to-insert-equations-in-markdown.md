title: 如何在markdown中插入公式
date: 2016-01-14 20:57:19
tags: [Markdown, Equation, MathJax, MarkdownPad 2]
categories: Markdown
---

## **MathJax插件** ##

著名的[Stackoverflow](http://stackoverflow.com/)网站上的漂亮公式，就是使用了MathJax插件的效果。添加MathJax插件也非常简单，只需要在markdown文件中，添加`MathJax CDN`，如下：

	<script type="text/javascript"
	   src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
	</script>

就可以在md文件中插入Tex格式的公式了。

`行间公式`的形式为

	$$ 此处插入公式 $$

而`行内公式`的形式为

	\\( 此处插入公式 \\)

## **在MarkdownPad 2中编辑公式** ##

之前的博文有推荐[Markdown Pad 2](http://markdownpad.com/)作为Window下的Markdown编辑器。如果你是使用该软件作为markdown的编辑器，你只需要在软件的`Tools-> Options-> Advanced-> HTML Head Editor`中添加上述的`MathJax CDN`即可。

这样你就不必每次都在md文件中重复添加了。

## **好用的Tex公式生成器** ##

推荐一个在线手写公式转Tex格式的利器：[Web Equation](https://webdemo.myscript.com/#/demo/equation)。通过手写公式，即可得到公式所对应的Tex格式，非常好用。

## **示例** ##

举个栗子。在`Markdown Pad 2`中新建文件，添加如下内容：

	最后，我们在一个图片类别的evidence中加入偏置(bias)，加入偏置的目的是加入一些与输入独立无关的信息。所以图片类别的evidence为
	
	$$ evidence\_{i}=\sum \_{j}W\_{ij}x\_{j}+b\_{i} $$
	
	其中，\\( W\_i \\) 和 \\( b\_i \\) 分别为类别 \\( i \\) 的权值和偏置。

（**注意**：markdown文件中的`_`前需要加上`\`转移符。）

最终效果如下（在Markdown Pad 2编辑器进行预览，快捷键为`F6`）：

![](http://i.imgur.com/dCW2j68.png)

## **Hexo中显示数学公式** ##

值得注意的是，原生的Hexo并不支持数学公式的显示。所以，如果你仅仅完成了以上步骤，在`hexo g -d`之后，你会发现公式的效果并没有被渲染出来。

### 安装hexo-math插件 ###

在网站根目录下，打开`git bash`，输入

	npm install hexo-math --save

然后，在根目录下的`_config.yml`文件中添加

	plugins: 
	  hexo-math

之后重新生成和部署网站即可。


----------


**参考资料**：

[MathJax with Markdownpad 2](http://pencilandengine.com/2013/08/27/mathjax-with-markdownpad-2/)

[Hexo上使用MathJax来实现数学公式的表达](http://hijiangtao.github.io/2014/09/08/MathJaxinHexo/)
