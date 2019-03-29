#coding=utf-8
import pandas as pd
import numpy as np
import jieba
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.layers import LSTM,Embedding,Dropout
from keras.utils import np_utils


def loadfile():
    neg=pd.read_excel('neg.xls',header=None,index=None)
    pos=pd.read_excel('pos.xls',header=None,index=None)

    combined=np.concatenate((pos[0], neg[0]))

    y=np.concatenate((np.ones(len(pos),dtype=int),np.zeros(len(neg),dtype=int)))
    return combined,y

def toker(text):
    text=[]
    for document in combined:
        text.append(jieba.lcut(document.replace('\n','')))
    return text

def create_dictionaries(model=None,
                        combined=None):

    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量

        def parse_dataset(combined):
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=140)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec,combined
    else:
        print 'No data provided...'



def Word_em(combined):
    model = Word2Vec(combined,
                     size=100,
                     min_count=5,
                     window=5)

    model.save('Word2vec_model.pkl')
    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    print combined.shape[0],combined.shape[1]
    return   index_dict, word_vectors,combined


def get_data(index_dict,word_vectors,combined,y):
    n_symbols=len(index_dict)+1
    embedding_weights=np.zeros((n_symbols,100))
    for word,index in index_dict.items():
        embedding_weights[index,:]=word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test

def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
    model=Sequential()
    model.add(Embedding(output_dim=100,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=140))
    model.add(LSTM(50,activation='sigmoid'))
    model.add(Dropout(0.7))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])
    model.fit(x_train,y_train,batch_size=5,epochs=50,verbose=2,validation_data=(x_test, y_test))
    score=model.evaluate(x_test,y_test,batch_size=5)
    print("Model Accuracy: %.2f%%" % (score[1]*100))


    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    print index
combined,y=loadfile()
text=toker(combined)
index_dict, word_vectors,combined=Word_em(text)
n_symbols,embedding_weights,x_train,y_train,x_test,y_test=get_data(index_dict, word_vectors,combined,y)
train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test)
