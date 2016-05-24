title: 如何将博客托管至Coding及相应的DNS设置
date: 2015-12-20 10:19:18
tags: [Hexo, Gitcafe, Coding, DNS, Github]
categories: Hexo
---
在[之前的博文](http://www.jeyzhang.com/Hexo-Github-%E6%90%AD%E5%BB%BA%E8%87%AA%E5%B7%B1%E7%9A%84%E9%9D%99%E6%80%81%E5%8D%9A%E5%AE%A2/)中，已经介绍了如何创建一个Hexo网站并且将其托管至Github上，从而实现一个静态的博客网站。由于国内访问Github速度较慢甚至无法访问，因此有了国内版的Github，也就是Gitcafe。将网站托管至Gitcafe上的好处就是，你的网站即使在国内的网络环境下也能被访问，同时也便于百度等中文搜索引擎的收录。

将博客托管至Gitcafe与托管至Github类似，但是仍存在一些细小的差别。下面将介绍如何将之前所创建的网站托管至Gitcafe上，主要包含两方面的内容：

1. 将网站托管至Gitcafe上；
2. 相应的DNS域名解析设置以实现国内外分流访问网站（即国外网络环境下是通过Github Page访问你的网站，而国内则是通过Gitcafe Page访问）。

**Gitcafe目前已经面临关闭，博文最后介绍了将静态网站迁移至Coding的方法。**

----------

# 将博客托管至Gitcafe上 #

## 注册Gitcafe账号 ##

打开[Gitcafe官网](https://gitcafe.com)，注册账号。

## 配置SSH ##

假设你已经完成将网站托管至Github上，此时你的本地已经生成了SSH公钥文件，打开这个文件并复制其中的内容。下面是我的ssh公钥文件路径

	C:\Users\ZhangJie\.ssh\id_rsa.pub

打开Gitcafe主页，登陆后进入到“账户设置”中，点击左侧的`SSH公钥管理`，然后`添加新的公钥`，将之前的内容粘贴到对应的栏目里即可。

为了测试ssh是否配置成功，打开本地的`git bash`，输入

	ssh -T git@gitcafe.com

如果显示
	
	Hi USERNAME! You've successfully authenticated...

则说明配置成功，此时你可以免密码将本地的项目文件同步至Gitcafe中。

如果依然存在问题，可以查看[官网详细的帮助页面](https://help.gitcafe.com/manuals/help/ssh-key)。

## 创建同名项目以及修改配置文件 ##

在Gitcafe上新建一个与你的账户名同名的项目。

修改站点目录下的配置文件，找到`deploy`项。我的deploy项内容如下

	# Deployment
	## Docs: http://hexo.io/docs/deployment.html
	deploy:
	  type: git
	  repo: 
	    github: git@github.com:JeyZhang/JeyZhang.github.io.git
	    gitcafe: git@gitcafe.com:JeyZhang/JeyZhang.git,gitcafe-pages

注意对于gitcafe，网站应该托管至其`gitcafe-pages`分支上。

## 将本地网站同步至Gitcafe项目中 ##

在网站根目录下，打开gitbash，输入

	hexo g -d

即可将本地网站同步至Github和Gitcafe上。

（注意：在本地网站根目录下的source文件夹下，需要新建一个文件名为`CNAME`的文件（无后缀名），里面填写你所绑定的域名地址，如www.jeyzhang.com.）

## 测试是否成功将网站托管至Gitcafe上 ##

如果项目根目录下存在`CNAME`文件，暂时现将其删除。（因为目前你还没有将你的网站域名解析至Gitcafe服务器上）

在浏览器输入地址

	http://jeyzhang.gitcafe.io

看网站是否访问成功（国内网络即可）。

# DNS解析设置 #

笔者使用[DNSPod](https://www.dnspod.cn)的域名解析,我的域名解析设置如下

![](http://i.imgur.com/gw8SKtu.png)

国内线路选择Gitcafe，国外线路选择Github，从而实现国内外分流访问网站。主机记录为`@`可以实现访问`××××××.com`时，自动填充"www"开头，因为之前我们绑定的网站是`www.××××××.com`的二级域名。

等待一段时间生效即可。

# 将博客托管至Coding平台 #

托管的方法与Gitcafe的类似，托管完成后，你可以手动删除Gitcafe上的项目（2016年5月31号之后系统也会自动清除的）。

## 注册Coding账户并建立项目 ##

去[Coding的官网](https://coding.net)注册，在个人主页的`项目`中创建一个项目，最好创建与你账户名相同的项目。例如，我的账户名为jeyzhang, 创建的项目名为JeyZhang（大小写不区分）。

## 上传SSH文件 ##

在Coding的个人主页的`账户`中，进入`SSH公钥`。添加你的公钥，如果你之前生成过，公钥在`C:\Users\你的用户名\.ssh\id_rsa.pub`。复制里面的内容在`SSH-RSA公钥内容`中即可。

打开`git bash`，输入
	
	ssh -T git@git.coding.net

进行测试，如果显示如下则SSH配置成功：

	Hello ...! You've conected to Coding.net by SSH successfully!

## 修改网站的配置文件 ##

修改网站根目录下的配置文件`_config.yml`，找到`deploy`的设置处，改为如下：

	deploy:
	  type: git
	  repo: 
	    github: git@github.com:JeyZhang/jeyzhang.github.io.git
	    coding: git@git.coding.net:JeyZhang/JeyZhang.git,master

注意要改成你的项目地址。

## 将网站文件部署至Coding ##

在网站根目录下打开`git bash`，输入

	hexo g -d

进行网站文件的生成和部署。成功之后，进入你的Coding对应的项目中应该能看到网站文件。

## 配置Coding的Page服务 ##

进入你在Coding上的项目，点击左侧的`代码`可以看到`Coding Pages`服务。输入分支为`master`，点击开启服务。在自定义域名处填上你的网站域名，如下图所示：

![](http://i.imgur.com/2Ko41Xk.png)

## 配置DNS ##

笔者使用[DNSPod](https://www.dnspod.cn/)进行网站的DNS设置。将国内线路设置为`CNAME`的`page.coding.me`即可。如下图所示。

![](http://i.imgur.com/rMOVcTS.png)

等待一会儿，你就能在国内网络快速访问你的网站了。

----------

本文结束，感谢欣赏。

**欢迎转载，请注明本文的链接地址：**

http://www.jeyzhang.com/blog-on-gitcafe-with-dns-settings.html