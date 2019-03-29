#coding=utf-8
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import io
import jieba
from jieba import posseg
import numpy as np
text = []

'''
数据格式说明：
CountVectorizer 词频向量，输入的是一个一维的数组，每句话切割后作为一个元素放入，而不是二维数据每个单词作为一个元素
lda模型的输出有两个，一个是训练后返回的结果，是一个doc-topic的概率矩阵，第二个是lda.components_ 是一个topic-word的矩阵
'''
f = io.open("mytest.txt","rb+")
line = f.readline()

while line:
    line = f.readline()
    posseg_words = jieba.posseg.cut(line)
    sequence = ""
    for word in posseg_words:
        if word.flag in ['an','i','n','nr','ns','nt','nz','t']:
            sequence = sequence+" "+word.word
    text.append(sequence)

countV = CountVectorizer()
b = countV.fit_transform(text)

lda = LatentDirichletAllocation(n_topics=2,doc_topic_prior=0.1,topic_word_prior=0.1)
doc = lda.fit_transform(b)

def print_top_words(model, feature_names, n_top_words):

    for topic_idx, topic in enumerate(model.components_):
        print "Topic #%d:" % topic_idx
        print " ".join([feature_names[i].encode('utf-8')
                        for i in topic.argsort()[:-n_top_words - 1:-1]])
feature_names = np.array(countV.get_feature_names())

print print_top_words(lda,feature_names,10)
