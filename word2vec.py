# %%
from tensorflow.keras.utils import to_categorical
from nltk import word_tokenize
from sklearn.utils import validation  # 以空格形式实现分词
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Embedding
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import LineSentence
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import datetime
from numpy.core.fromnumeric import size
import pandas as pd
import os
from bs4 import BeautifulSoup  # 用于解析网页
from urllib.request import urlopen  # 用于获取网页
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from numpy import seterrobj
from sklearn import datasets
import re

# %%
# 文本初步处理

# 停用词
stop_words = stopwords.words('english')
# 数字


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))
# 特殊符号


def isSymbol(inputString):
    return bool(re.match(r'[^\w]', inputString))


"""
正则表达式，\w表示单词字符[A-Za-z0-9_]
[^\w]表示取反==\W非单词字符
全是特殊符号时返回true
"""
# 词性归并
wordnet_lemmatizer = WordNetLemmatizer()


def check(word):
    """
    如果需要这个单词，则True
    如果应该去除，则False
    """
    word = word.lower()
    if word in stop_words:
        return False
    elif hasNumbers(word) or isSymbol(word):
        return False
    else:
        return True


def preprocessing(sen):
    res = []
    for word in sen:
        if check(word):
            # 这一段的用处仅仅是去除python里面byte存str时候留下的标识。之前数据没处理好，其他case里不会有这个情况
            word = word.lower().replace("b'", '').replace(
                'b"', '').replace('"', '').replace("'", '')
            res.append(wordnet_lemmatizer.lemmatize(word))
    return res


path = "./file"  # 文件夹目录
files = os.listdir(path)  # 得到文件夹下的所有文件名称
print(files)
s = []
for file in files:  # 遍历文件夹
    sentences = []
    if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
        f = open(path+"/"+file, encoding='UTF-8')  # 打开文件
        ff = f.readlines()
        for line in ff:
            sentences.append(line.rstrip("\n"))
        while '' in sentences:
            sentences.remove('')
    s.append(sentences)

separated_words = []
for article in s:
    set = []
    for sentence in article:
        sen = word_tokenize(sentence)
        sen_processed = preprocessing(sen)
        for w in sen_processed:
            set.append(w)
    separated_words.append(set)

# %%
wd = 10
mc = 5
hs = 1
worker = 8
model = Word2Vec(separated_words, vector_size=1000, window=wd,
                 min_count=mc, hs=hs, workers=worker)

# %%
batch = np.zeros(1, len(s))
for article in s:
    word_vec_len = len(model.wv.index_to_key)
