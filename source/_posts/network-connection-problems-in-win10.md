title: Win10下网络连接问题及解决方案
date: 2016-01-28 17:54:15
tags: [Network, Windows, Win10]
categories: Network
---

回家后，发现笔记本能连上家里的wifi，但是就是不能上网。网络诊断问题提示是【**此计算机上缺少一个或多个网络协议**】。尝试了多种方法一直解决不了，经过一番折腾后终于成功，特此记录一下，以供后人参考。

**操作系统环境**：Windows 10

**网络诊断错误**：此计算机缺少一个或多个网络协议

----------

## 可能的解决方法：开启特定的网络服务项 ##

网上大多都是这种方法，如下：

按`windows+R`键，在运行窗口中输入`services.msc`，检查以下服务是否正常开启：

	Telephony;
	Remote Access Connection Manager;
	Remote Access Auto Connection Manager;

找到上述服务中手动开启的项，右键`属性`；确认修改所选服务的启动类型为`自动`，如果服务状态为停止，点击`启动`来启动服务。

但是，这个方法并没有解决问题。本人尝试后失败。

## 可行的解决方法： 卸载目前的驱动程序##

在`开始`处，点击`右键`，选择`网络连接`。

找到连接上的无线网络，点击`右键`，选择`状态`,点开应该如下图：

![](http://i.imgur.com/jFeBr1w.png)

点击图中的`属性`，如下图

![](http://i.imgur.com/Excc8Wr.png)

点击`配置`，选择`驱动程序 > 卸载`，如下图

![](http://i.imgur.com/leNeCOP.png)

勾选卸载相应的驱动程序，完成卸载。重启系统，然后重连wifi即可。

----------

本文结束，感谢欣赏。

**欢迎转载，请注明本文的链接地址：**

http://www.jeyzhang.com/network-connection-problems-in-win10.html




