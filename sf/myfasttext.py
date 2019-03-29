#encoding=utf-8
import fasttext
import pandas as pd
import numpy as np
import io
import jieba
#pandas的read_excel返回一个df,df通过index可返回索引,values可得到值，df.tolist()变为List,但是要得到string（编码类型）通过df.ix[i,j]，如果ix[]只填行或者列返回的是一个serise类型

def loadfile():
    neg=pd.read_excel('neg.xls',header=None,index=None)
    pos=pd.read_excel('pos.xls',header=None,index=None)

    print neg,pos

    file =io.open("mydata.txt","wb+")
    for i in neg.index.values:
        file.write(neg.ix[i,0].encode('utf-8')+" "+"__label__cp"+"\n")
        if i>3000:
            file.write(pos.ix[i,0].encode('utf-8')+" "+"__label__hp"+"\n")

    frames = [neg,pos]
    result = pd.concat(frames)

    combined=np.concatenate((pos[0], neg[0]))

    y=np.concatenate((np.ones(len(pos),dtype=int),np.zeros(len(neg),dtype=int)))

    return combined,y


def toker():
    f = open("mydata.txt")             # 返回一个文件对象
    w = open("mydata2.txt","wb")
    line = f.readline()             # 调用文件的 readline()方法
    res = jieba.cut(line)
    for itme in res:
        print itme.encode('utf-8')
    while line:
                         # 后面跟 ',' 将忽略换行符
        line = f.readline()

        res = jieba.cut(line)
        for itme in res:
            if itme == " ":
                continue
            if itme in ["label","__","cp","hp"]:
                w.write(itme.encode('utf-8'))
            else:
                w.write(itme.encode('utf-8')+" ")
        w.write("\n")
    f.close()
    w.close()
loadfile()
toker()
