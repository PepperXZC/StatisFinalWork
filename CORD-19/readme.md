### Topic

#### 数据清洗

首先运行以下代码，将原本的csv文件导入`DataFrame`格式中，便于接下来`pandas`库的数据清洗：
```python
with open('metadata.csv', 'r', encoding='UTF-8') as f_in:
    reader = csv.DictReader(f_in)
    for row in reader:

        # access some metadata
        # cord_uid = row['cord_uid']
        cord_uid = row['cord_uid']
        title = row['title']
        abstract = row['abstract']
        authors = row['authors'].split('; ')
        publish_time = row['publish_time']

        paper_list.append({
            'cord_uid' : cord_uid,
            'title': title,
            'abstract': abstract,
            'authors': authors,
            'publish_time': publish_time
        })
```

参数解释：

`cord_uid`：每个论文在本数据集中的ID
`title`：论文标题（默认不为空，否则不存在这样的样本）
`abstract`：论文摘要（在本数据集中可能为空字符串，需要清除这类样本）
`authors`：作者（一般不止一个作者，需要进行字符串分割操作）
`publish_time`：发行时间（年份有很多，后续将按照时间清洗掉大部分数据，简化数据集）

<!-- 运行以下代码可以看到目前的样本量是极大的：
```python -->

后面暂时不写了，先记录一下实验。

2-19 11点13分：将`clean.py`中的代码逻辑优化并归为新的`new_clean.py`文件中。该文件从`metadata.csv`中导出一个经过分词处理好棋摘要的文件`new_clean.csv`。重新读取时，请注意其每个`abstract`中为字符串，须重新导入分词后的`list`格式中使用该数据。程序运行时间较长。

2-19 12点08分 注意到有些字符在中文解码下出现了些问题，可在`excel`与`pandas`中分别调用查看`data.iloc[7,3]`看到。再后续调用分词操作的时候，需要考虑将这些内容删去。对此可直接调用的操作如下：
```python
data = pd.read_csv('new_clean.csv')
print(data.iloc[7,3])
```
p.s. 英语以外的语言检测暂时还没有解决方法。

p.p.s 须安装`nltk`包以及`pyenchant`进行程序的使用。

`pyenchant`对应的安装语句：`pip install pyenchant`

`nltk`包在安装时`_sqlite3`的缺失问题。这需要自己去官网下载对应的文件，补入环境文件中的`DLLs`文件夹下。具体操作请参照 CSDN 链接：[CSDN](https://blog.csdn.net/qq_42685893/article/details/116519140)

安装后在调用一些库时仍然会报错，解决代码会在报错中提示，请在`cmd`或`anaconda prompt`中在对应的环境下运行`python`后进入交互环境，并运行如下代码：
```python
>>> import nltk
>>> nltk.download('stopwords')
```
至少有两次类似的问题，请仿照上面的操作将第二次报错解决。这`nltk`包安装产生的通病。

2-19 12点34分 更新了 `new_clean`中的`abstract_check`方法，其中添加了`clean_text`方法，因为`nltk`自带的符号去除函数并不能很好地去除纯数字与特殊符号。我暂时没有运行这个函数，因为这将消耗大量的时间。其中不乏一些重复的删除空格一类的操作，可以考虑删去以节省程序运行时间。

2-19 18点12分：打开`new_clean.csv`发现还存在着一些空摘要、摘要内容为`author`的样本，在`new_clean.csv`的最后补全了如下代码:
```python
data = data[data['abstract'].str.len() >= 30]
```
由`excel`直接显示的`csv`文件会导致解码出现异常。典型的状况是：在`pandas`中读取`abstract`发现没有任何问题的摘要文本，但是在`excel`中读取发现`author`列与`abstract`列数据紊乱（两者颠倒了），我默认在`pandas`中没问题就ok