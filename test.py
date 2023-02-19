import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
import re

def get_string(self, each_file):
    temp = self.readtxt(each_file)
    if len(temp) <= 1:
        return ''
    else:
        rp = ''.join([i for i in temp if not i.isdigit()])
        rp = re.sub('[\W_]+', '', rp)
        sentences = rp.split()
        sent_words = [list(jieba.cut(sent0)) for sent0 in sentences]
        return sent_words
        
data = pd.read_csv('new_clean.csv')
# stop_words = 'en_stopwords.txt'

print(data.iloc[7,3])
# for sample in data['abstract']:
words = [sample for sample in data['abstract']]
# print(words[0].split())

cv = CountVectorizer(stop_words='english')
cv_fit = cv.fit_transform(words)
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
