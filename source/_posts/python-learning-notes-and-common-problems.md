title: python学习笔记与常见问题
date: 2016-04-27 17:22:49
tags: [Python]
categories: Python
---

## **列表** ##

### 使用lambda函数实现列表操作

实现两个list的元素对应相乘，返回同等大小的list结果。

	list1 = [...]
    list2 = [...]
    multiply_list = map(lambda (a, b): a * b, zip(list1, list2))

### 列表的类型转换 ###

例如，经常遇到需要将元素类型为`int`的列表，转为元素类型为`str`的列表，可以方便的使用`join`等函数进行格式化处理。使用`map`函数将很简单：

	int_list = [1, 2, 3]
    str_list = map(str, int_list) # str_list = ['1', '2', '3']
    # 类似的还有：
    map(int, list) # list中的元素均转为int类型
    map(float, list) # list中的元素均转为float类型

## **字符串及编码** ##

### python2中unicode编码下中文显示问题

使用`json`对对象进行包装，再打印或者写入文件，如下

![](http://imgur.com/F7w50wW.png)

写入文件时，需要转化为`UTF-8`编码，如下

	jsond = json.dumps(**, ensure_ascii=False)
    f.write(jsond.encode('utf-8'))
    ...

在python3中打印unicode编码字符串很简单，使用**`pprint`**；python2中也可以使用**`pprint`**，具体做法见[**这里**](https://www.quora.com/How-do-you-print-a-python-unicode-data-structure)。






