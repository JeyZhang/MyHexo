title: Hexo网站优化之SEO
date: 2015-12-17 10:04:18
tags: [Hexo, SEO, Web]
categories: Hexo
---

SEO (Search Engine Optimization)，即搜索引擎优化。对网站做SEO优化，有利于提高搜索引擎的收录速度及网页排名。下面讲解一些简单的SEO优化方法，主要针对Hexo网站。

----------

## SEO优化之title ##

编辑站点目录下的`themes/layout/index.swig`文件，

将下面的代码

![](http://i.imgur.com/jjHCELX.png)

改成

![](http://i.imgur.com/hNv09sO.png)

这时将网站的描述及关键词加入了网站的`title`中，更有利于详细地描述网站。

## 添加robots.txt ##

robots.txt是一种存放于网站根目录下的ASCII编码的文本文件，它的作用是告诉搜索引擎此网站中哪些内容是可以被爬取的，哪些是禁止爬取的。robots.txt应该放在站点目录下的source文件中，网站生成后在网站的根目录(`站点目录/public/`)下。

我的robots.txt文件内容如下

	User-agent: *
	Allow: /
	Allow: /archives/
	Allow: /categories/
	Allow: /about/
	
	Disallow: /vendors/
	Disallow: /js/
	Disallow: /css/
	Disallow: /fonts/
	Disallow: /vendors/
	Disallow: /fancybox/
	
## 添加sitemap ##

Sitemap即网站地图，它的作用在于便于搜索引擎更加智能地抓取网站。最简单和常见的sitemap形式，是XML文件，在其中列出网站中的网址以及关于每个网址的其他元数据（上次更新时间、更新的频率及相对其他网址重要程度等）。

**Step 1**: 安装sitemap生成插件

	npm install hexo-generator-sitemap --save
	npm install hexo-generator-baidu-sitemap --save

**Step 2**: 编辑站点目录下的_config.yml，添加

	# hexo sitemap网站地图
	sitemap:
	path: sitemap.xml
	baidusitemap:
	path: baidusitemap.xml

**Step 3**: 在robots.txt文件中添加

	Sitemap: http://www.jeyzhang.com/sitemap.xml
	Sitemap: http://www.jeyzhang.com/baidusitemap.xml


----------

本文结束，感谢欣赏。

**欢迎转载，请注明本文的链接地址：**

http://www.jeyzhang.com/hexo-website-seo.html