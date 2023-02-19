import numpy as np
import pandas as pd
# import jieba
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from unidecode import unidecode
        
data = pd.read_csv('new_clean.csv')
corpus = data['abstract'].to_list()

cv = CountVectorizer(stop_words='english')
cv_fit = cv.fit_transform(corpus)
print(cv.get_feature_names_out())

# toarray 转化出：(样本，词频) 矩阵。
# 对应每个样本在每个column(每个column对应的单词由get_feature_names_out给出)下的词语出现次数

res = cv_fit.toarray()
# count_0 = np.where(res, 0, 1)
# print(res.shape) # (39675, 77027)
# print(res.sum()) # 4257805

# print(np.sum(count_0))
# test_str = data.iloc[7,3]
# print(type(test_str))
# qq = re.sub('\d+', '', str(test_str))
# qq = re.sub(r'\.*', '', qq)
# print(qq)
# for sample in data['abstract']:
# words = [nltk.word_tokenize(sample) for sample in data['abstract']]
# print(words[7])
# print(words[11])
# print(words[0].split())


# print(cv.get_feature_names_out())

# f=open("k.txt","w")
# l = cv.get_feature_names_out()
# for key in l:   
#     f.writelines(key + '\n')
# f.close()
# print(cv.vocabulary_)
# tfidf_model = TfidfVectorizer(smooth_idf=True,
#                               max_df=0.90,
#                               stop_words=stop_words).fit(data['abstract'])
# sparse_result = tfidf_model.transform(data['abstract']).todense()
