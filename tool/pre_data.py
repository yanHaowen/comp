#coding=utf-8
import numpy as np
import pandas as pd
from sklearn.preprocessing import Binarizer,MinMaxScaler,Normalizer,StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

class Pretreat:
    '''调用时统一使用df[["属性"]]'''

    '''二值化，thre是阈值，大于阈值为1,小于等于为0'''
    def binarizer_(self,X,thre):

        bin = Binarizer(threshold=0.0)
        bin.fit(X)
        return bin.transform(X)

    '''最大最小值缩放，根据最大值和最小值进行缩放'''
    def min_max_scaler(self,X,range_):
        mm = MinMaxScaler(feature_range=range_)
        mm.fit(X)
        return mm.transform(X)

    '''正则化，使数据服从标准正太分布'''
    def normal_(slef,X,norn):
        nor = Normalizer(norm=norn)
        nor.fit(X)
        return nor.transform(X)

    '''标准号,使所有数据都在0附近，方差为1'''
    def Z_score(self,X):
        st = StandardScaler()
        st.fit(X)
        return st.transform(X)

    def Iforest(self,X,num):
        clf = IsolationForest(n_estimators=num)
        clf.fit(X)
        pre = clf.predict(X)
        for i in range(len(pre)):
        if pre[i] == -1:
            wring_index.append(i)
        else:
            right_index.append(i)
        data = X.ix[right_index]
        w_data =X.is[wring_index]
        return data,w_data

    def Lfactor():
        clf = LocalOutlierFactor()
        y_pre = clf.fit_predict(X)
        right_index = []
        wring_index = []
        for i in range(len(y_pre)):
            if y_pre[i] == -1:
                wring_index.append(i)
            else:
                right_index.append(i)

        data = X.ix[right_index]
        w_data = X.ix[wring_index]
        return data,w_data
df = pd.read_csv("/home/yhw/桌面/compa/光伏功率预测/data/train_1.csv")
pre = Pretreat()
print pre.min_max_scaler(df[["实发辐照度","温度"]],(-1,1))
