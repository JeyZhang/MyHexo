title: whoosh+jieba：python下实现中文全文检索
date: 2016-04-28 11:14:53
tags: [Python, Whoosh, Jieba, Search Engine]
categories: Python
---

## 需要安装 ##

* python 2.xx
* whoosh
* jieba

whoosh和jieba的安装使用`pip install`即可。

## 快速入门 ##

下面的代码实现了简单的中文检索

    # coding=utf-8
    import os
    from whoosh.index import create_in
    from whoosh.fields import *
    from jieba.analyse import ChineseAnalyzer
    import json

    # 使用结巴中文分词
    analyzer = ChineseAnalyzer()

    # 创建schema, stored为True表示能够被检索
    schema = Schema(title=TEXT(stored=True, analyzer=analyzer), path=ID(stored=False),
                    content=TEXT(stored=True, analyzer=analyzer))

    # 存储schema信息至'indexdir'目录下
    indexdir = 'indexdir/'
    if not os.path.exists(indexdir):
        os.mkdir(indexdir)
    ix = create_in(indexdir, schema)

    # 按照schema定义信息，增加需要建立索引的文档
    # 注意：字符串格式需要为unicode格式
    writer = ix.writer()
    writer.add_document(title=u"第一篇文档", path=u"/a",
                        content=u"这是我们增加的第一篇文档")
    writer.add_document(title=u"第二篇文档", path=u"/b",
                        content=u"第二篇文档也很interesting！")
    writer.commit()

    # 创建一个检索器
    searcher = ix.searcher()

    # 检索标题中出现'文档'的文档
    results = searcher.find("title", u"文档")

    # 检索出来的第一个结果，数据格式为dict{'title':.., 'content':...}
    firstdoc = results[0].fields()

    # python2中，需要使用json来打印包含unicode的dict内容
    jsondoc = json.dumps(firstdoc, ensure_ascii=False)

    print jsondoc  # 打印出检索出的文档全部内容
    print results[0].highlights("title")  # 高亮标题中的检索词
    print results[0].score  # bm25分数