title: 2015年校招总结：技术面试干货
date: 2016-03-03 20:11:07
tags: [Interview, Job, IT, Algorithm]
categories: Interview
---

关于实习及校招的全面概括性总结在[**这篇博文**](http://www.jeyzhang.com/2015-campus-recruit-summary.html)，里面也提出了一些技术面试过程中的注意事项。本文主要是单纯针对**程序员技术面试的面试内容**，将（1）推荐一些优秀的资源（包括书籍、网站等），以及（2）总结一下自己及周遭同学在实习与校招技术面试过程中遇到的各种原题，以供后人参考。

----------

## 推荐资源 ##

### 书籍 ###

> **算法类**

- **《Crack the code interview》(Gayle Laakmann著) [[PDF下载地址](http://vdisk.weibo.com/s/DpqS8KKk4Vcu)]**

这本书是经典的程序员技术面试必备书了，作者是曾经的Google面试官，从面试官的角度教你应该如何一步步地准备面试。书中分析了硅谷的一些巨头公司的面试风格和特点，对于想要面国外公司的再合适不过了；还帮助你制定了面试准备的流程和计划，给出写简历的建议，如何应对行为面试(Behavioral Interview)等；当然，最主要的篇幅集中在技术面试的准备中，总结了常见的数据结构及相应的算法题，数理概率，及一些其他面试中常见的技术题型。

- **《进军硅谷：程序员面试揭秘》(陈东锋著) [[豆瓣地址](https://book.douban.com/subject/25844586/)]**

尽管这本书在豆瓣上的评分很低（leetcode作者认为该书抄袭了leetcode上的题目...），但对于面试者来说，这本书还是值得推荐的。这本书前面部分也是主要介绍了一下面试流程和注意事项，硅谷公司的特点；其余的大篇幅都是集中在算法题的解题思路分析和代码实现，确实大部分的算法题与leetcode上的一样，所以刷leetcode的时候配合这本书，应该会顺畅挺多的。这本书的代码都是Java，简单易懂。


- **《剑指Offer》(何海涛著) [[PDF下载地址](http://vdisk.weibo.com/s/EjagsS5Ugjw)]**

这本书的结构其实与前两本比较类似，但是有一个亮点是，对于所有的算法题都会给出测试样例，包括特殊边界和正常功能测试样例等。写算法题能够提前考虑测试样例是非常好的编程习惯，称之为**`防御式编程`**；大多数人都是习惯写完代码后，再进行样例测试，然后修修补补之类的。

- **《微软面试100题系列》(July著) [[PDF下载地址](http://vdisk.weibo.com/s/akZyBqthxGDMn?from=page_100505_profile&wvr=6)]**

严格上来说，这个并不是一本正式的书籍。但是这个资料里收集了许多经典真实的企业面试题。题型比较杂，大部分是算法题，还有智力题等。虽然答案不是很全，但是值得好好看看里面的题，从本人的笔试面试经历来看，遇到了里面挺多的原题~

- **《编程之美：微软技术面试心得》[[PDF下载地址](http://download.csdn.net/detail/sunmeng_alex/4606246)]**

如果时间充裕的话，这本书也可一看。这本书是由MSRA的一些FTE和实习生们编写的，老实说，这本书中很多题还是挺有难度的，有许多数学相关的题，不折不扣地考验你的智商……偶尔翻翻，转转脑子也挺好的。

此外，还有一些神书，例如《算法导论》《编程珠玑》也可一看。但是，时间总是有限的，**认真刷刷1-2本书，然后多动手配合刷题（刷题平台下面有推荐）**，应付面试的算法能力自然会慢慢变强。


> **数据结构类**

- **《Java数据结构和算法》(Robert Lafore著) [[PDF下载地址](http://vdisk.weibo.com/s/dhYy6pCj8N-9z?from=page_100505_profile&wvr=6)]**

相比起清华的严奶奶那本，这本书通俗易懂得多:)要是觉得之前的数据结构掌握的不够好，这本书绝对能拉你入门~

- **《数据结构：C语言版》(严蔚敏著) [[PDF下载地址](http://vdisk.weibo.com/s/aFmBnrN-WDuX6)]**

虽然刚学的时候觉得晦涩难懂，但是还是国内经典的书籍，对数据结构研究的比较深刻，内容较上本会丰富很多。

> **编程语言类**

- **Java**

《Java编程思想》(Bruce Eckel著) [[PDF下载地址](http://vdisk.weibo.com/s/aPgqW10HL3Q90)]

《Effective Java》(Joshua Bloch著)(中文版) [[PDF下载地址](http://vdisk.weibo.com/s/grsVw)] | (英文版) [[PDF下载地址](http://vdisk.weibo.com/s/dq65Vm6HA4vmD)]

《疯狂Java讲义》(李刚著) [[PDF下载地址](http://vdisk.weibo.com/s/A-1hO0QV0ZhQ)]

- **C**

《C Primer Plus》(Stephen Prata著) [[PDF下载地址](http://vdisk.weibo.com/s/zfhMNTK9gWJOV)]

《征服C指针》(前桥和弥著) [[PDF下载地址](http://vdisk.weibo.com/s/e41M8kWaqoim)]

- **Python**

《Python基础教程》(Magnus Lie Hetland著) [[PDF下载地址](http://vdisk.weibo.com/s/dhZbFvYADsgqr)]

《Python简明教程》(Swaroop著) [[PDF下载地址](http://vdisk.weibo.com/s/BE2Z8B94-5w97)]

《利用Python进行数据分析》(Wes McKinney著) [[PDF下载地址](http://vdisk.weibo.com/s/AFN3jW3skIDf)]

《Learn Python The Hard Way》[[PDF下载地址](http://vdisk.weibo.com/s/BCRaGM7XY1jut)]


> **数据库类**

《SQL必知必会》(Ben Forta著) [[PDF下载地址](http://vdisk.weibo.com/s/y-3ktzWX4vlzr)]

《深入浅出SQL》(Lynn Beighley著) [[PDF下载地址](http://vdisk.weibo.com/s/aHSh1alRGXpb)]

《高性能MySQL》(Baron Schwarlz等著) [[PDF下载地址](http://vdisk.weibo.com/s/GNZwNnGfiqSm)]

### 刷题网站 ###

- **[Leetcode](https://leetcode.com/)**

众所周知的刷题网站了，许多公司的面试题都是从里面出的。建议刷3遍左右。

- **[Lintcode](http://www.lintcode.com/)**

一个类似于leetcode的刷题网站，但是比起leetcode，里面的题目更加齐全。还有一些特色的功能，如限时提交，编程风格检测等。

- **[九度OJ](http://ac.jobdu.com/)**

里面收录了《剑指Offer》中的题，可以配合看书练习。还有一些考研机试、比赛类型的题，适合刷完leetcode等网站后，磨练算法能力。

- **[hihoCoder](http://www.hihocoder.com/)**

这个平台经常举办一些编程比赛，一些公司的笔试会选择在这个平台进行，例如微软(中国)、网易游戏等。另外，这个平台里面的题有一定难度，适合算法能力中上的人。

### 网站与论坛 ###

- **[九章算法](http://www.jiuzhang.com/)**

曾经上过它的算法课，还可以。里面有leetcode中大多数题的解答（只有代码，大多数是Java），还有一些面筋之类的分享。有时间和米的还可以去听听他家的课，都是PPT+白板+语音的形式。

- **[GeeksforGeeks](http://www.geeksforgeeks.org/)** 

- **[Career Cup](https://www.careercup.com/)**

以上这两个网站上面有很多国外最近的、真实的面试题分享和讨论，也可以经常去水水~另外，这个**[知乎问题](https://www.zhihu.com/question/20368410)**，票数第一的回答还总结了挺多的。

- **[我的cnblog](http://www.cnblogs.com/harrygogo/)**

之前在面试准备过程中，在cnblog上建了个博客，记录了以下刷的算法题及面试题。欢迎访问。

## 真实面筋 ##

### 算法和数据结构 ###

- 非递归实现二叉树深度的求解（Google SED实习）
- 如何实现双端队列（Google SED实习）
- 一维的连连看实现 (阿里春招实习)
- 动归和贪心的区别 (阿里春招实习)
- 大小为999的一维数组，按序存放着1-1000的数字，但有一个数字缺失，找到它 (阿里春招实习)
- 最长公共子序列 (阿里春招实习)
- 

#### 大数据 ####

- 集合A：40亿个未排序，不重复的unsigned int；集合B：1万个unsined int；判断集合B中的数是否属于集合A。（输出1W个bool值）(阿里春招实习)
- 1亿个查询记录，找出frequency top1000；follow up：讲解堆的调整过程。(阿里春招实习)

### 数据挖掘和机器学习 ###

- 如何识别买家评价的虚假评价（阿里春招实习）
- SVM特征怎么提的，参数怎么调的，半监督学习是在干嘛，整体学习是在干嘛?(阿里春招实习)
- 讲解CNN，CNN和DNN相比有什么优点为什么用它?(阿里春招实习)
- 随机森林和决策树的基本原理 (阿里春招实习)
- SVM原理及公式推导 (阿里春招实习)
- Boosting算法 (阿里春招实习)
- 对数线性模型 (阿里春招实习)
- 概率主题模型，LDA思想 (阿里春招实习)


### 编程语言 ###

#### C/C++ ####

- C++的多态和虚函数(阿里春招实习)

#### Java ####

- Spring的IOC是什么？Spring是怎么实现依赖控制的？(阿里春招实习)
- Java的synchronized和lock有什么区别？volatile有什么作用？(阿里春招实习)
- Java的hashmap怎么实现。(阿里春招实习)
- 
- 

#### Python ####


### 移动客户端开发 ###

#### Android ####

- Android的fragment和activity有什么区别？activity能否在不同的进程中启动？(阿里春招实习)
- 

#### iOS ####


### 计算机网络 ###

- HTTP协议中的SSL怎么实现？(阿里春招实习)
- 

### 数据库 ###


### 操作系统 ###

- 操作系统分页和分块有什么区别？(阿里春招实习)
- 什么是线程安全？(阿里春招实习)

#### Linux ####



----------

本文结束，感谢欣赏。

**欢迎转载，请注明本文的链接地址：**

http://www.jeyzhang.com/2015-campus-recurit-technology-interview-summary.html

**最后特别感谢2015年面点交流群各位伙伴的面筋:)**