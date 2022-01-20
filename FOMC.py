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


# %%
import re
# 爬FOMC网页上的article


def filterurl(result):
    urlptn = r"(.+)(monetarypolicy)(.+)"  # 匹配模式: 所有http://开头的链接
    urls = [re.match(urlptn, str(_)) for _ in result]  # 正则筛选
    # print(urls)
    while None in urls:
        urls.remove(None)  # 移除表中空元素
    urls = [_.group() for _ in urls]  # group方法获得re.match()返回值中的字符
    # print(urls)
    return urls


html = urlopen('https://www.federalreserve.gov/monetarypolicy/mpr_default.htm')
bsObj = BeautifulSoup(html, 'html.parser')
t1 = bsObj.find_all('a')
L = []
for t2 in t1:
    t3 = t2.get('href')
    L.append(t3)
L_filter = filterurl(L)
L_filter = L_filter[13:43]
for i, html in enumerate(L_filter):
    if i in np.arange(1, 10):
        L_filter[i] = "https://www.federalreserve.gov" + L_filter[i]
L_filter


for i, HTML in enumerate(L_filter[0:1]):
    html = urlopen(HTML)
    bs = BeautifulSoup(html, 'html.parser')
    t1 = bs.find_all(id='article')
    for t2 in t1:
        # .txt可以不自己新建,代码会自动新建
        with open('./file/' + L_filter[i][46:] + '.txt', 'w', encoding='UTF-8') as file_handle:
            file_handle.write(t2.get_text())
    # file_handle.write('\n')


# %%

# 初步测试：每年2月和7月第一周涨跌幅作为label
pd.set_option('display.max.rows', None)
df = pd.read_excel('Signal.xlsx')
for i in [1, 4, 12]:
    df[i] = df['000300.SH'].rolling(i).sum()
df_indexed = df.set_index('date')
judge = df_indexed > 0
judge['M'] = pd.to_datetime(df['date']).apply(
    lambda x: datetime.datetime.strftime(x, "%Y-%m")).values

judge_indexed = judge.set_index('M')
judge_indexed
t = judge_indexed[1][judge['M'].apply(
    lambda x: (x[5:7] == '02') or (x[5:7] == '07')).values]
sig = t.groupby('M').apply(lambda x: x[0])


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
sentences_for_article = []
num_of_sentences_for_article = []
for article in s:
    set = []
    sentences_for_article.append(article)
    num_of_sentences_for_article.append(len(article))
    for sentence in article:
        sen = word_tokenize(sentence)
        sen_processed = preprocessing(sen)
        for w in sen_processed:
            set.append(w)
    separated_words.append(set)


# %%
# 对于每一篇文章统计句子的个数和长度
sen_length = []
sen_count = 0
for sens in sentences_for_article:
    for sen in sens:
        ws = sen.split()
        sen_length.append(len(ws))
        sen_count += 1

# %%
# 生成每一个句子的信号（0|1）
input_sig = np.zeros([1, sen_count])
count = 0
for m, n in enumerate(num_of_sentences_for_article):
    for i in np.arange(0, n):
        input_sig[0, count] = sig[m]
        count += 1

input_sig

# %%
# Word2vec生成词向量
Size = round(np.percentile(sen_length, 95, axis=None, out=None,
             overwrite_input=False, interpolation='linear', keepdims=False))
wd = 5
mc = 5
hs = 1
worker = 4
model = Word2Vec(separated_words, vector_size=Size, window=wd,
                 min_count=mc, hs=hs, workers=worker)

# 对词向量求平均得到句向量


def build_sentence_vector(sentence, size, w2v_model):
    sen_vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in sentence:
        try:
            pos = model.wv.key_to_index[word]
            sen_vec += model.wv.vectors[pos].reshape((1, size))
            count += 1
        except KeyError:
            continue
    if count != 0:
        sen_vec /= count
    return sen_vec


sen_vecs = np.zeros([Size, sen_count])
count = 0
for sens in sentences_for_article:
    for sen in sens:
        ws = sen.split()
        sen_vec = build_sentence_vector(ws, Size, model)
        sen_vecs[:, count] = sen_vec
        count += 1

# %%
# CNN
X_train, X_test, y_train, y_test = train_test_split(sen_vecs.T, input_sig.T,
                                                    test_size=0.2)
X_train_expand = np.expand_dims(X_train.T, 0)
X_test_expand = np.expand_dims(X_test.T, 0)
# y_train = to_categorical(y_train.T)
# y_test = to_categorical(y_test.T)

filter = 1  # 卷积核数量
ks = 5  # 卷积核宽度
sd = 1  # 卷积移动步长
maxpool = 4

CNN = Sequential()
CNN.add(Conv1D(filter, ks, input_shape=(
    X_train_expand.shape[1], X_train_expand.shape[2])))
# CNN.add(MaxPooling1D(maxpool))
CNN.add(AveragePooling1D(maxpool))
CNN.add(Flatten())
# CNN.add(Dense(X_train_expand.shape[2], activation='relu'))
# CNN.add(Dense(X_train_expand.shape[2], activation='softmax'))
CNN.add(Dropout(0.2))
CNN.add(Dense(234, activation='softmax'))
adam = Adam(learning_rate=1e-3, beta_1=0.9,
            beta_2=0.999, epsilon=1e-08, decay=0.0)

CNN.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
CNN.fit(X_train_expand, y_train.T, epochs=100)
# CNN.predict(X_test_expand)
# CNN.summary()

# %%
