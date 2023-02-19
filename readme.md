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
以及，英语以外的语言检测暂时还没有解决方法。


