import nltk
import re
import string
import csv
import pandas as pd
# nltk.download('stopwords')
# pip install pyenchant
from enchant.checker import SpellChecker
from nltk.corpus import stopwords

def abstract_check(text) -> list[str]:
    # 去除中文
    text = re.sub('[\u4e00-\u9fa5]','',text)
    # 分词
    text = nltk.word_tokenize(text)
    # 去掉停用词
    stop = set(stopwords.words('english'))
    text = [i for i in text if i not in stop]
    # print(text)
    jstr = " ".join(text)
    # 检查拼写
    chkr = SpellChecker("en_US", jstr)
    for err in chkr:
        # 删去所有的错误拼写单词
        err.replace("")
    # 去掉标点符号
    now = chkr.get_text()
    remove = str.maketrans('','',string.punctuation)
    without_punctuation = now.translate(remove)
    # 再进行一次分词，去掉没有必要的空格
    tokens = nltk.word_tokenize(without_punctuation)
    return tokens

# 打开源文件
paper_list = []
with open('metadata.csv', 'r', encoding='UTF-8') as f_in:
    reader = csv.DictReader(f_in)
    for row in reader:
        # 将所有内容以字典形式添加到列表中
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

# 关闭文件，将列表中的内容转换为 DataFrame
data = pd.DataFrame(paper_list)
# data['publish_time'] = pd.to_datetime(data['publish_time'])
print(data.shape)
print(data.columns)

data['publish_time'] = pd.to_datetime(data['publish_time'])
# 检测 2020 年间的所有论文
start_date = pd.to_datetime('2020-1-31', format='%Y-%m-%d')
end_date = pd.to_datetime('2020-12-31', format='%Y-%m-%d')

data = data[data.publish_time.isin(pd.date_range(start_date, end_date, freq='D'))]
data['publish_year'] = data['publish_time'].dt.year

# 清洗掉abstract为空的样本
# print(data.shape) # (81643, 6)
data = data.dropna(subset=['abstract'])
data = data[data['abstract']!='']
# print('删掉空摘要样本后:',data.shape) # (55750, 6)
# 删除abstract重复的内容
data= data.drop_duplicates(subset=['abstract'], keep='last')
# 删除含重复uid的样本
data= data.drop_duplicates(subset=['cord_uid'], keep='last')
# abstract 中所有字符变为小写
data['abstract'] = data['abstract'].str.lower()

# 删除列，因为已经默认只要是2020年内的
data = data.drop('publish_time',axis=1)

# 设定 covid 单词
covid_word = ['covid', 
              'corana', 
              'pandemic', 
              'pandemics',
              'sars-cov-2', 
              'sarscov', 
              'covid-19', 
              'covid-2019', 
              'coronavirus']

data['covid_abstract'] = data.abstract.apply(lambda x: any(word in x for word in covid_word))
# print(data['covid_abstract'].value_counts())
data = data[data.covid_abstract == True]
# print('删掉没有covid字符的摘要的样本：',len(data)) # 40061
# data.to_csv('new_clean.csv')
# 删除没有必要的索引
data = data.drop('covid_abstract',axis=1)
data = data.reset_index(drop=True)
# print(data)
# 对每个 abstract 进行文本处理
# res = {}
for index in range(data.shape[0]):
    text = data.iloc[index, 2]
    # uid = data.iloc[index, 0]
    # res[uid] = abstract_check(text)
    data.iloc[index, 2] = " ".join(abstract_check(text))

# 保存最终处理好的文件
data.to_csv('new_clean.csv')
    # data.iloc[index, 2] = 0