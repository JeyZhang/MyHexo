title: Hexo的NexT主题个性化：添加文章阅读量
date: 2016-01-22 09:28:40
tags: [Hexo, Next, LeanCloud]
categories: Hexo
---

关于Hexo的文章阅读量设置问题，大多数人都是使用[不蒜子](http://service.ibruce.info/)的代码实现。但是这个方法仅局限于在文章页面显示阅读数，首页是不显示的。

下面介绍如何在首页及文章页面都显示`文章的阅读量`，显示效果如下：

![](http://i.imgur.com/AMdIdpW.png)

----------

## **配置[LeanCloud](https://leancloud.cn/)** ##

### 注册 ###

打开LeanCloud官网，进入[注册页面](https://leancloud.cn/login.html#/signup)注册。完成邮箱激活后，点击头像，进入`控制台`页面，如下：

![](http://i.imgur.com/WyRLYr3.png)

### 创建新应用 ###

创建一个新应用(类型为`JavaScript SDK`)，点击应用进入；

创建名称为`Counter`的Class

![](http://i.imgur.com/5VUiBAy.png)

![](http://i.imgur.com/C8LWKT2.png)

### 修改配置文件 ###

编辑网站根目录下的`_config.yml`文件，添加如下：

	# add post views
	leancloud_visitors:
	  enable: true
	  app_id: **你的app_id**
	  app_key: **你的app_key**

其中，app_id和app_key在你所创建的应用的`设置->应用Key`中。

### Web安全性 ###

为了保证应用的统计计数功能仅应用于自己的博客系统，你可以在`应用->设置->安全中心`的`Web安全域名`中加入自己的博客域名，以保证数据的调用安全。

## **修改NexT主题文件** ##

### 添加lean-analytics.swig文件 ###

在主题目录下的`\layout\_scripts`路径下，新建一个名称为`lean-analytics.swig`的文件，并添加如下内容：

```
	<!-- custom analytics part create by xiamo -->
	<script src="https://cdn1.lncld.net/static/js/av-core-mini-0.6.1.js"></script>
	<script>AV.initialize("{{theme.leancloud_visitors.app_id}}", "{{theme.leancloud_visitors.app_key}}");</script>
	<script>
	function showTime(Counter) {
		var query = new AV.Query(Counter);
		$(".leancloud_visitors").each(function() {
			var url = $(this).attr("id").trim();
			query.equalTo("url", url);
			query.find({
				success: function(results) {
					if (results.length == 0) {
						var content = '0 ' + $(document.getElementById(url)).text();
						$(document.getElementById(url)).text(content);
						return;
					}
					for (var i = 0; i < results.length; i++) {
						var object = results[i];
						var content = object.get('time') + ' ' + $(document.getElementById(url)).text();
						$(document.getElementById(url)).text(content);
					}
				},
				error: function(object, error) {
					console.log("Error: " + error.code + " " + error.message);
				}
			});
	
		});
	}
	
	function addCount(Counter) {
		var Counter = AV.Object.extend("Counter");
		url = $(".leancloud_visitors").attr('id').trim();
		title = $(".leancloud_visitors").attr('data-flag-title').trim();
		var query = new AV.Query(Counter);
		query.equalTo("url", url);
		query.find({
			success: function(results) {
				if (results.length > 0) {
					var counter = results[0];
					counter.fetchWhenSave(true);
					counter.increment("time");
					counter.save(null, {
						success: function(counter) {
							var content =  counter.get('time') + ' ' + $(document.getElementById(url)).text();
							$(document.getElementById(url)).text(content);
						},
						error: function(counter, error) {
							console.log('Failed to save Visitor num, with error message: ' + error.message);
						}
					});
				} else {
					var newcounter = new Counter();
					newcounter.set("title", title);
					newcounter.set("url", url);
					newcounter.set("time", 1);
					newcounter.save(null, {
						success: function(newcounter) {
						    console.log("newcounter.get('time')="+newcounter.get('time'));
							var content = newcounter.get('time') + ' ' + $(document.getElementById(url)).text();
							$(document.getElementById(url)).text(content);
						},
						error: function(newcounter, error) {
							console.log('Failed to create');
						}
					});
				}
			},
			error: function(error) {
				console.log('Error:' + error.code + " " + error.message);
			}
		});
	}
	$(function() {
		var Counter = AV.Object.extend("Counter");
		if ($('.leancloud_visitors').length == 1) {
			addCount(Counter);
		} else if ($('.post-title-link').length > 1) {
			showTime(Counter);
		}
	}); 
	</script>
```

其中，控制显示的格式的主要为`content`变量，按自己的需求相应修改即可。

### 修改post.swig文件 ###

在主题的`layout\_macro`路径下，编辑`post.swig`文件，找到相应的插入位置（大概在98行左右）：

![](http://i.imgur.com/l21gZ2f.png)

插入如下代码

```
		  {% if theme.leancloud_visitors.enable %}
			 &nbsp; | &nbsp;
			 <span id="{{ url_for(post.path) }}"class="leancloud_visitors" data-flag-title="{{ post.title }}">
             &nbsp;{{__('post.visitors')}}
            </span>
		  {% endif %}
```
### 修改layout.swig文件 ###

在主题目录下的`layout`目录下，编辑`_layout.swig`文件，在`</body>`的上方（大概在70行左右）插入如下代码：

```
	{% if theme.leancloud_visitors.enable %}
	{% include '_scripts/lean-analytics.swig' %}
	{% endif %}
```

### 修改语言配置文件 ###

如果你的网站使用的是英语，则只需要编辑主题目录下的`languages\en.yml`文件，增加`post`字段如下：

	post:
	  sticky: Sticky
	  posted: Posted on
	  visitors: Views // 增加的字段
	  ...

如果网站使用的是中文，则编辑`languages\zh-Hans.yml`文件，相应的增加

	post:
	  posted: 发表于
	  visitors: 阅读次数
	  ...

其他语言与之类似，将`visitors`设置成你希望翻译的字段。

**最后，重新生成并部署你的网站即可。**

## **增加网站的浏览次数与访客数量统计功能** ##

网站的浏览次数，即`pv`；网站的访客数为`uv`。`pv`的计算方式是，单个用户连续点击n篇文章，记录n次访问量；`uv`的计算方式是，单个用户连续点击n篇文章，只记录1次访客数。你可以根据需要添加相应的统计功能。

## 安装`busuanzi.js`脚本 ##

如果你使用的是NexT主题（其他主题类似），打开`/theme/next/layout/_partial/footer.swig`文件，拷贝下面的代码至文件的开头。

```
<script async src="https://dn-lbstatics.qbox.me/busuanzi/2.3/busuanzi.pure.mini.js">
</script>
```

## 显示统计标签 ##

同样编辑`/theme/next/layout/_partial/footer.swig`文件。

如果你想要显示`pv`统计量，复制以下代码至你想要放置的位置，

```
<span id="busuanzi_container_site_pv">
    本站总访问量<span id="busuanzi_value_site_pv"></span>次
</span>
```

如果你想要显示`uv`统计量，复制以下代码至你想要放置的位置，

```
<span id="busuanzi_container_site_uv">
  本站访客数<span id="busuanzi_value_site_uv"></span>人次
</span>
```

你可以自己修改文字样式，效果图如下：

![](http://i.imgur.com/rWtu2TU.png)

----------

本文结束，感谢欣赏。

**欢迎转载，请注明本文的链接地址：**

http://www.jeyzhang.com/hexo-next-add-post-views.html


**参考资料**

[为NexT主题添加文章阅读量统计功能](http://notes.xiamo.tk/2015-10-21-%E4%B8%BANexT%E4%B8%BB%E9%A2%98%E6%B7%BB%E5%8A%A0%E6%96%87%E7%AB%A0%E9%98%85%E8%AF%BB%E9%87%8F%E7%BB%9F%E8%AE%A1%E5%8A%9F%E8%83%BD.html)