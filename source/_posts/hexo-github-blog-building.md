title: 'Hexo+Github: 搭建属于自己的静态博客'
date: 2015-12-07 21:44:01
tags: [Hexo, Github Page, Blog, Personal Website]
categories: Hexo
---
Hexo是一个快速、简洁且高效的博客框架，而Github是一个免费的代码托管工具，利用Github Page可以免费创建一个静态网站。下面将介绍如何使用Hexo和Github，在win10环境下搭建一个静态的博客。

全文分为三个部分：

1. 安装和配置Hexo及Github
2. 选择Hexo主题及发表文章
3. 注册及绑定自己的域名地址

----------

## **安装和配置Hexo及Github** ##

### 安装Hexo ###

安装Hexo前，需要安装以下：

- [Node.js](http://nodejs.org/)
- [Git](https://git-scm.com/download/win)

如果已经安装完成以上程序，打开Git-bash或者cmd，输入

	npm install -g hexo-cli

即可完成Hexo的安装。

### 使用Hexo进行本地建站 ###

选择一个本地的文件夹，如`D:\hexo`。

输入

	hexo init D:\hexo
	cd D:\hexo
	npm install

如果hexo安装成功，则在`D:\hexo`文件夹下的文件目录为

	.
	├── _config.yml // 网站的配置信息，你可以在此配置大部分的参数。
	├── package.json 
	├── scaffolds // 模板文件夹。当你新建文章时，Hexo会根据scaffold来建立文件。
	├── source // 存放用户资源的地方
	|   ├── _drafts
	|   └── _posts
	└── themes // 存放网站的主题。Hexo会根据主题来生成静态页面。

详细文件或文件夹的具体含义见 [Hexo官方文档之建站](https://hexo.io/zh-cn/docs/setup.html)

为了测试本地建站是否成功，输入

	hexo s

如果显示如下

![](http://i.imgur.com/7iVVdep.png)

则说明本地建站成功，访问[本地地址](http://localhost:4000/)可以看到Hexo默认主题的效果。

至此，Hexo的安装和本地建站完成，如需更加深入全面地了解Hexo，可访问[Hexo官方文档](https://hexo.io/zh-cn/docs/)。

### 创建Github账号 ###

如果已经注册Github，可跳过此步骤。否则，访问[Github官网](https://github.com/)进行注册，下面假设你注册Github账号名为MyGithub。

### 创建与账号同名的Repository ###

注册并登陆Github官网成功后，点击页面右上角的`+`，选择`New repository`。

在`Repository name`中填写`你的Github账号名.github.io`，这里是`MyGithub.github.io`。`Description`中填写对此repository的描述信息(可选，但建议填写，如`Personal website`)。

点击`Create repository`，完成创建。

### 配置SSH ###

**(1) 生成SSH**

检查是否已经有SSH Key，打开Git Bash，输入

	cd ~/.ssh

如果没有这个目录，则生成一个新的SSH，输入

	ssh-keygen -t rsa -C "your e-mail"

其中，`your e-mail`是你注册Github时用到的邮箱。

然后接下来几步都直接按回车键，最后生成如下

![](http://i.imgur.com/RSCTurW.jpg)

**(2) 复制公钥内容到Github账户信息中**

打开`~/.ssh/id_rsa.pub`文件，复制里面的内容；

打开Github官网，登陆后进入到个人设置(`点击头像->setting`)，点击右侧的`SSH Keys`，点击`Add SSH key`；填写title之后，将之前复制的内容粘贴到Key框中，最后点击`Add key`即可。

**(3) 测试SSH是否配置成功**

输入

	ssh -T git@github.com

如果显示以下，则说明ssh配置成功。

	Hi username! You've successfully authenticated, but GitHub does not
	provide shell access.


### 将网站发布到Github的同名repository中 ###

打开`D:\Hexo`文件夹中的`_config.yml`文件，找到如下位置，填写

	# Deployment
	## Docs: http://hexo.io/docs/deployment.html
	deploy: 
	  type: git
	  repo: git@github.com:MyGithub/MyGithub.github.io

**注**： (1) 其中`MyGithub`替换成你的Github账户; (2) 注意在yml文件中，`:`后面都是要带空格的。

此时，通过访问`http://MyGithub.github.io`可以看到默认的Hexo首页面（与之前本地测试时一样）。

## **选择Hexo主题及发表文章** ##

### 简洁的Next主题 ###

本网站使用的是[Next主题](https://github.com/iissnan/hexo-theme-next)。该主题简洁易用，在移动端也表现不错。

**(1) 下载Next主题**

	cd D:\Hexo
	git clone https://github.com/iissnan/hexo-theme-next themes/next

**(2) 修改网站的主题为Next**

打开`D:\Hexo`下的`_config.yml`文件，找到`theme`字段，将其修改为`next`

	# Extensions
	## Plugins: http://hexo.io/plugins/
	## Themes: http://hexo.io/themes/
	theme: next

**(3) 本地验证是否可用**

输入

	hexo s --debug

访问[本地网站](http://localhost:4000)，确认网站主题是否切换为Next.

**(4) 更新Github**

输入

	hexo g -d

完成Github上网页文件的更新。

### 发表新文章 ###

发表文章操作非常简单，在网站存放的根目录打开`git bash`，输入

	hexo n "name of the new post"

回车后，在source文件夹下的_post文件夹下，可以看到新建了一个`name of the new post.md`的文件，打开

	title: name of the new post
	date: 2015-12-09 22:55:25
	tags:
	---

可以给文章贴上相应的tags，如有多个则按照如下格式

	[tag1, tag2, tag3, ...]

在`- - -`下方添加正文内容即可，注意需要使用markdown语法进行书写。

[在这里](http://wowubuntu.com/markdown/)有关于Markdown语法的简单说明。推荐使用[MarkdownPad2](http://markdownpad.com/)进行md文件的编辑工作。

文章撰写完成后保存，输入

	hexo g -d

即可生成新网站，并且同步Github上的网站内容。

## **注册及绑定自己的域名地址** ##

截止到目前为止，你应该可以通过访问`http://MyGithub.github.io`来看到以上创建的网站了。

但是，如何拥有一个属于自己的域名地址，并将其指向在Github上所创建的网站呢？

### 注册域名 ###

推荐选择国内的[万网](http://wanwang.aliyun.com/)或者国外的[Goddady](https://www.godaddy.com/)进行域名的注册。

### DNS域名解析设置 ###

如果你选择的是万网注册的域名，可以使用其自带的域名解析服务。

进入[万网](http://wanwang.aliyun.com/)，登陆后进入到个人中心(点击用户名即可)，点击左侧的"云解析"，点击之前所购买的域名，在"解析设置"中，添加如下解析规则:

![](http://i.imgur.com/AqhPQst.png)

其中，当记录类型为A时，记录值为服务器的ip地址，这里的服务器地址为存放`Github page`的地址，你可以通过命令行输入

	ping github.io

得到。

DNS域名解析设置需要一定时间，之后你可以通过ping自己的域名地址来查看是否解析成功。

### 在Github对应的repository中添加CNAME文件 ###

即在 MyGithub/MyGithub.github.io 中加入名为"CNAME"的文件，文件内容为你的域名地址，如

	www.××××××.com

保存即可。

CNAME文件设置的目的是，通过访问 MyGithub.github.io 可以跳转到你所注册的域名上。

为了方便本地文件deploy的时候，CNAME文件不发生丢失，可以在本地网站根目录下的source文件夹下，添加以上的CNAME文件。以后每次deploy的时候，CNAME文件不会发生丢失。

----------

通过以上的设置，相信你已经可以通过注册域名来访问一个默认的hexo主题页面了。之后的工作就在于，(1)如何对主题进行个性化设置及；(2)发表博文以充实网站内容。[这里](http://theme-next.iissnan.com/)有关于next主题的个性化设置说明。

本文结束，感谢欣赏。

**欢迎转载，请注明本文的链接地址：**

http://www.jeyzhang.com/hexo-github-blog-building.html

