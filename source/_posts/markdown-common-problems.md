title: markdown写作中的常见问题
date: 2016-01-22 21:44:33
tags: [Markdown]
categories: Markdown
---

本文记录markdown写作过程中所遇到的问题及相应的解决方法，以供参考。

----------


## Q1: 代码中出现{% raw %}{% 和 %}{% endraw %}所包围的语句 ##

如果代码中出现了类似于{% raw %}{%××××××%}{% endraw %}格式的语句，需要在这些`语句块的首尾`加上

```
{% raw %}
```
和
```
{% endraw %}
```

以保证显示原始的语句。

示例如下：

![](http://i.imgur.com/aYitATq.png)

显示效果如下：

	{% raw %}
	{% if theme.leancloud_visitors.enable %}
	{% include '_scripts/lean-analytics.swig' %}
	{% endif %}
	{% endraw %}

## Q2: 如何显示原始的html代码 ##

在html代码块的上下均加上`三个连续的反引号`。

示例如下：

![](http://i.imgur.com/B1XqXXr.png)

显示效果如下：

```
	{% if theme.leancloud_visitors.enable %}
	{% include '_scripts/lean-analytics.swig' %}
	{% endif %}
```

## Q3: 怎样给字体阴影的效果 ##

例如`这样的阴影效果`，只需要在内容的前后加上一个反引号，如下图：

![](http://i.imgur.com/fr7Fapa.png)

## Q4: 如何显示公式中的花括号{} ##

markdown中正常文本中使用`\`对`{}`进行转义即可；而公式中的`{}`即使这样转义也是不会显示的，正确做法是使用`\lbrace \rbrace`来表示左右花括号。

----------

本文结束，感谢欣赏。

**欢迎转载，请注明本文的链接地址：**

http://www.jeyzhang.com/markdown-common-problems.html


**参考资料**

[Markdown 语法说明 (简体中文版)](http://wowubuntu.com/markdown/)

[issue#587: Markdown代码块中的Markdown语法](https://github.com/hexojs/hexo/issues/587)

[V2EX: markdown反引号内怎么转义反引号](https://www.v2ex.com/t/57233)

[Markdown中写数学公式](http://jzqt.github.io/2015/06/30/Markdown%E4%B8%AD%E5%86%99%E6%95%B0%E5%AD%A6%E5%85%AC%E5%BC%8F/)