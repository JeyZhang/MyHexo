title: NexT主题个性化设置
date: 2015-12-14 19:50:04
tags: [Hexo, Next]
categories: Hexo
---

**提前说明**：

假设网站的根目录为"D:/Hexo/"，也称为**站点目录**

**站点配置文件** 是指网站目录下的_config.yml文件，即"D:/Hexo/_config.yml"

**主题配置文件** 是指网站目录下对应的主题文件夹下的_config.yml文件，即"D:/Hexo/themes/next/_config.yml".

下面的功能设置完成后，记得

	hexo g -d

以完成网站的生成和部署。

----------

## 添加分类、标签云、关于等页面 ##

以添加分类页面为例，

在**站点目录**下，打开git bash，输入

	hexo new page "categories"

之后在站点目录下的source文件夹下，会新增一个"categories"的文件夹，里面有一个"index.md"文件，打开如下

	title: categories
	date: 2015-12-04 15:37:22
	type: "categories"
	comments: false
	---

其中，comments可以设置为false，含义是打开分类页面，评论插件不显示；如要显示则改为"true"。

tags, about页面的创建类似，输入

	hexo new page "tags"
	hexo new page "about"

## 添加站内搜索功能 ##

NexT支持[Swiftype插件](https://swiftype.com/)以实现站内搜索功能。

**Step 1**: 注册[Swiftype](https://swiftype.com/users/sign_up)

**Step 2**: 创建一个新的搜索引擎 (点击"Create an engine"，按要求创建即可)

**Step 3**: 点击新建的搜索引擎，按如下点击"INSTALL SEARCH"

![](http://i.imgur.com/ZUvxhKH.png)

然后复制下面蓝色底的字串

![](http://i.imgur.com/6RfwX4n.png)

**Step 4**: 编辑站点配置文件，添加如下内容

	# Swiftype Search Key
	swiftype_key: xxxxxxxxx(粘贴以上复制的内容)

## 设置右侧栏头像 ##

编辑站点配置文件，添加如下内容

	avatar: your avatar url

其中，your avatar url可以是
(1) 完整的互联网URL，你可以先将设置的头像图片放到图床上；
(2) 本地地址：如/upload/image/avatar.png (你需要将avatar.png文件放在/站点目录/source/upload/image/里面)。

## 设置favicon图标 ##

**Step 1**:
首先要有一个常见格式名(如.jpg, .png等)的图片作为备选favicon，选择一个favicon制作网站完成制作，例如[比特虫](http://www.bitbug.net/)是一个免费的在线制作ico图标网站。

**Step 2**:
将"favicon.ico"文件放在网站根目录下的source文件夹内即可。刷新网站，就可以看到效果了。

## 添加社交链接 ##

编辑站点配置文件，添加

	social:
	  github: https://github.com/your-user-name
	  twitter: https://twitter.com/your-user-name
	  weibo: http://weibo.com/your-user-name
	  douban: http://douban.com/people/your-user-name
	  zhihu: http://www.zhihu.com/people/your-user-name
	  # 等等

可根据自身需要自行删减。

## 添加友情链接 ##

以添加github官网( https://www.github.com )为友情链接为例

编辑站点配置文件，添加如下内容

	# title
	links_title: Links
	# links
	links:
	  Github: https://www.github.com

其中，links_title为友情链接的名称。

## 添加评论区 ##

支持Disqus和多说两种评论样式。建议中文网站选择多说，英文网站选择Disqus。下面以Disqus为例说明。

**Step 1**: 注册[Disqus](https://disqus.com/)

**Step 2**: 登陆后进入到"Settings"，点击"Add Disqus To Site"，然后点击页面的右上角的"Install on Your Site"

**Step 3**: 复制你的shortname

![](http://i.imgur.com/1LZfdm5.png)

**Step 4**: 编辑站点配置文件，添加

	disqus_shortname: your disqus shortname

这样你的所有文章及页面下面，会自动加载Disqus的评论插件。如果在分类、标签云等页面，不想显示评论区，可以打开这个page文件夹下的md文件，添加

	comments: false

## 首页文章以摘要形式显示 ##

最简单的方式是：打开**主题配置文件**，找到如下位置，修改

	auto_excerpt:
	  enable: true
	  length: 150

其中length代表显示摘要的截取字符长度。

## 设置首页文章显示篇数 ##

**Step 1**: 安装相关插件

输入如下命令

	npm install --save hexo-generator-index
	npm install --save hexo-generator-archive
	npm install --save hexo-generator-tag

**Step 2**: 

安装完插件后，在站点配置文件中，添加如下内容

	index_generator:
	  per_page: 5
	
	archive_generator:
	  per_page: 20
	  yearly: true
	  monthly: true
	
	tag_generator:
	  per_page: 10

其中per_page字段是你希望设定的显示篇数。index, archive及tag开头分表代表主页，归档页面和标签页面。

## 设置404公益页面 ##

在**站点目录**的source文件夹下，新建404.html文件，将下面的代码复制进去保存即可。

	<!DOCTYPE HTML>
	<html>
	<head>
		<title>404 - arao'blog</title>
		<meta name="description" content="404错误，页面不存在！">
		<meta http-equiv="content-type" content="text/html;charset=utf-8;"/>
		<meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1" />
		<meta name="robots" content="all" />
		<meta name="robots" content="index,follow"/>
	</head>
	<body>
		<script type="text/javascript" src="http://qzonestyle.gtimg.cn/qzone_v6/lostchild/search_children.js" charset="utf-8"></script>
	</body>
	</html>

显示效果如下

![](http://i.imgur.com/n5wN34M.png)
----------

更多关于NexT主题的个性化设置，可参考[NexT官方帮助文档](http://theme-next.iissnan.com/)。