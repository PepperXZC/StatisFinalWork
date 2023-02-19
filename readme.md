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

运行以下代码可以看到目前的样本量是极大的：
```python
