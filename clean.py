# _*_ coding:utf-8 _*_
import csv
import os
import json
from collections import defaultdict
import pandas as pd
# from enchant.checker import 
import string
import numpy as np
import re
from unidecode import unidecode
from pqdm.processes import pqdm
import nltk
# nltk.download('stopwords')
# pip install pyenchant
from enchant.checker import SpellChecker
from nltk.corpus import stopwords
'''
nltk 包使用时会出现 _sqlite3 的问题：需要自己去官网下载
https://blog.csdn.net/qq_42685893/article/details/116519140
'''

# cord_uid_to_text = defaultdict(list)
paper_list = []

# open the file
'''with open('metadata.csv', 'r', encoding='UTF-8') as f_in:
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
        })'''
    

# print(len(paper_list)) # 29500

'''paper_df = pd.DataFrame(paper_list)
# paper_df['publish_time'] = pd.to_datetime(paper_df['publish_time'])
print(paper_df.shape)
print(paper_df.columns)

paper_df['publish_time'] = pd.to_datetime(paper_df['publish_time'])
start_date = pd.to_datetime('2020-1-31', format='%Y-%m-%d')
end_date = pd.to_datetime('2020-12-31', format='%Y-%m-%d')


paper_df = paper_df[paper_df.publish_time.isin(pd.date_range(start_date, end_date, freq='D'))]
paper_df['publish_year'] = paper_df['publish_time'].dt.year

print(paper_df.publish_year.value_counts())
res = paper_df.to_csv('after_clean.csv')'''
# print(paper_df.columns)

# 清洗abstract
'''data = pd.read_csv(r'after_clean.csv')
print('after_clean数据集大小:', data.shape)
data = data.dropna(subset=['abstract'])
data = data[data['abstract']!='']
print('删掉空摘要样本后:',data.shape)
data['abstract'] = data['abstract'].str.lower()

covid_word = [' covid ', 
              ' corana ', 
              ' pandemic ', 
              ' pandemics ',
              ' sars-cov-2 ', 
              ' sarscov ', 
              ' covid-19 ', 
              ' covid-2019 ', 
              ' coronavirus ']

data['covid_abstract'] = data.abstract.apply(lambda x: any(word in x for word in covid_word))
# print(data['covid_abstract'].value_counts())

# 清洗掉没有这些字符串的
data = data[data.covid_abstract == True]
print('删掉没有covid字符的摘要的样本：',len(data))

def clean_text(text):
    '''
    # [...] 选择...中的东西
    # [^...] 表示不选择...中的东西
    # [A-Z] [a-z] 选A-Z中的内容
    # [\w] 匹配下划线
'''
    ## Remove numbers
    text = re.sub('\d+', '', str(text))
    ## Remove special characters
    text = unidecode(text)

    ## Remove any non-letter characters except for regular sentence-ending punctuation
    text = re.sub(r'[^a-zA-Z\s\.!\?\\n]', '', text)

    ## Replace all \s with a single space except for \n
    # 换掉 tab(\t)
    # 换掉垂直制表符(\x0B)
    # 换掉换页符 \f
    # 换掉回车符
    text = re.sub(r'[ \t\x0B\f\r]+', ' ', text)

    ## Replace a newline with a dot and a space
    # * 表示尽可能多地匹配空格，也就是匹配连续的空格
    # \s 匹配任何空白字符，包括空格、制表符、换页符
    text = re.sub(r'\s*\n\s*', '. ', text)

    ## Remove any leading or trailing spaces
    text = text.strip()
    
    ## Remove duplicates white space
    # 移除多余的空格
    text = re.sub(r'^\s+', "", text)
           
    ## Only keep words that has length longer than 2 characters 
    text = ' '.join([x for x in text.split() if len(x)>2 or x=='of'])      

    ## remove dots
    # 移除点号
    text = re.sub(r'\.*', '', text)

    return text

test_string = '.  of  It is 2%.    p . p  raining     at 9am today..   中文 la $ 90 (The forcast says it will stop later.)'
print(clean_text(test_string))


for i in range(len(data)):
    temp = data.iloc[i,3]
    data.iloc[i,3] = clean_text(temp)

data.to_csv('after_clean_abs.csv')'''

# 删除重复的各种东西

'''data = pd.read_csv('after_clean_abs.csv')
print(data.shape)

data= data.drop_duplicates(subset=['abstract'], keep='last')
# 删除重复的摘要
print(data.shape)
data= data.drop_duplicates(subset=['cord_uid'], keep='last')
print(data.shape)

data['title']=data['title'].str.lower()
data= data.drop_duplicates(subset=['title'], keep='last')
print(data.shape)
data.to_csv('after_clean_dup.csv')'''

# data.to_csv('after_clean_dup.csv')
# 删除一些没必要的列
'''data = pd.read_csv('after_clean_dup.csv')
print(data)
data = data.drop('Unnamed: 0.2',axis=1)
data = data.drop('Unnamed: 0.1',axis=1)
data = data.drop('Unnamed: 0',axis=1)
# data = data.drop('Unnamed: 0',axis=1)
data = data.drop('covid_abstract',axis=1)
data = data.drop('publish_time',axis=1)
print(" ")
data.to_csv('after_clean_col.csv')'''

# 清洗法语（没运行）
'''def preliminary_eng_check(text):
    if 'the' in text or 'The' in text:
        return True
    else:
        return False
    
data = pd.read_csv('after_clean_col.csv')
print(data)
data['paper_id'] = 'id_' + data.index.astype(str)

data['prob_eng'] = data.abstract.apply(preliminary_eng_check)
print(data['prob_eng'].value_counts())

for index, row in data[data.prob_eng].sample(1).iterrows():
    print(row['abstract'])'''

# 我认为这个预处理简直就是一坨屎。下面是重新进行英文abstract的预处理过程：
# data = pd.read_csv(r'after_clean.csv')
# print('after_clean数据集大小:', data.shape)
# data = data.dropna(subset=['abstract'])
# data = data[data['abstract']!='']
# print('删掉空摘要样本后:',data.shape)
# data['abstract'] = data['abstract'].str.lower()

# example
text = 'I amm我是 一个普通的喜欢篮球的男生啊 jast a booy, and (( loved 我baskerball 还a lot. Just a lucky boy喜欢.'

# part = r"""(?x)                   
# 	           (?:[A-Z]\.)+          
# 	           |\d+(?:\.\d+)?%?      
# 	           |\w+(?:[-']\w+)*       
# 	           |\.\.\.  
# 	           |\S\w* 
# 	           |\w+         
# 	           |(?:[.,;"'?():-_`])    
# 	         """
# 去除中文
text = re.sub('[\u4e00-\u9fa5]','',text)
print(text)
# 分词
# text = nltk.regexp_tokenize(text,part)
# print(text)
text = nltk.word_tokenize(text)
print(text)
# 去掉停用词
stop = set(stopwords.words('english'))
text = [i for i in text if i not in stop]
print(text)
jstr = " ".join(text)

# 检查拼写
chkr = SpellChecker("en_US", jstr)
# chkr.set_text(text)
for err in chkr:
    # print(err.word)
    err.replace("")
print(chkr.get_text())

# 去掉标点符号
now = chkr.get_text()
remove = str.maketrans('','',string.punctuation)
without_punctuation = now.translate(remove)
print(without_punctuation)
# 再进行一次分词，去掉没有必要的空格
tokens = nltk.word_tokenize(without_punctuation)
print(tokens)
# print(tokens)
# print
# print(data['abstract'])
# for i in range(len(data)):
#     temp = data.iloc[i, 3]
#     temp = re.sub('[\u4e00-\u9fa5]','',temp)  # 去除中文
    



# data = pd.read_csv('after_clean.csv')
# print(data['abstract'])
# print(data)
# print(data.columns)
# data =  data.reset_index(drop=True)
# print(data)

# print(data.iloc[0,1])
# data.iloc[0,1] = '12345'
# print(data.iloc[0,3])
# data['abstract'] = pqdm(data.abstract, clean_text, n_jobs=5)
# print(data[0])
# data.to_csv('after_clean_abs.csv')


# print(paper_df.index.to_list())
# paper_df.drop(index=[0,1])

# l = paper_df['mc_uid'].to_list()
# move_list = [i for i in range(len(l)) if len(l[i]) <= 1]
# paper_df = paper_df.drop(move_list, axis=0)
# print(paper_df.shape)
# print(l)
        # paper_df = paper_df.drop(index=i,axis=0)
        # print(len(data))

        # print(paper_df.shape)
#     # print(data)

# df = pd.DataFrame(np.arange(12).reshape(3,4), columns=['A', 'B', 'C', 'D'])
# print(df)
# df = df.drop(['B', 'C'], axis=1)
# print(df)