#coding=utf-8
from pyecharts import Scatter,Bar,Boxplot,Page
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2,SelectKBest,SelectFromModel
import operator
from sklearn import ensemble

class Select:
    '''此处的category只描述y的数据类型,0是标称数据，1是连续数据'''
    def __init__(self,category):
        self.category = category

    ''' 函数用统计方法（卡方）对特征值进行排序，填入的数据分标称数据'''
    def feature_sort_statistics_b(self,X_b,y):
        sort = dict()
        feature_name_b = X_b.columns.values
        feature_name_y = y.columns.values[0]

        if self.category == 0:
            data_y,p_y = self.Chi_squared(X_b.values,y.values)
            dic = dict()
            for core,name in zip(data_y,feature_name_b):
                dic.update({name:core})
            dic = sorted(dic.items(),key=operator.itemgetter(1),reverse=True)
            sort.update({'所有特征对'+feature_name_y:dic})

        dics = dict()
        for i in range(len(feature_name_b)):
            name_ = feature_name_b[i]
            lab = [feature_name_b[k] for k in range(i+1,len(feature_name_b))]
            if lab == []:
                break
            data_,p_ = self.Chi_squared(X_b[lab].values,X_b[name_].values)
            for core,name in zip(data_,lab):
                dics.update({name_+"-"+name:core})
        dics = sorted(dics.items(),key=operator.itemgetter(1),reverse=True)
        sort.update({'特征之间':dics})

        return sort

        '''使用皮尔逊系数计算两个数值属性间的相关性'''
    def feature_sort_statistics_s(self,X_s):
        sort = dict()
        dic,dic_y = self.person(X_s)
        if self.category == 1:
            sort.update({"其他属性对y的相关系数":dic_y})
        else:
            dic = dict(dic,**dic_y)
            dic = sorted(dic.items(),key=operator.itemgetter(1),reverse=True)

            sort.update({"其他属性之间的相关系数":dic})
        return sort

        '''利用树模型对特征进行筛选'''
    def feature_sort_tree(self,model,X,y):
        sort = dict()
        dic = dict()
        model.fit(X,y)
        feature_name = X.columns.values
        feature_importances = model.feature_importances_
        print feature_importances
        for i in range(len(feature_name)):
            dic.update({feature_name[i]:feature_importances[i]})

        dic = sorted(dic.items(),key=operator.itemgetter(1),reverse=True)

        tp = type(model)
        sort.update({str(tp)+"模性排序":dic})
        return sort


    def feature_sort_linear(self,model,X,y):
        sort = dict()
        dic = dict()
        feature_name = X.columns.values
        model.fit(X,y)
        coef = model.coef_
        for i in range(len(coef)):
            dic.update({feature_name[i]:coef[i]})
        dic = sorted(dic.items(),key=operator.itemgetter(1),reverse=True)
        tp = type(model)
        sort.update({str(tp)+"模型特征选择":dic})
        return sort

#-------------------------------------------------------------help function-----------------------------------------------------#
    '''相关系数的获取，只对数值型数据进行相关性排序，如果y也是数值类型，那么默认y在第一列'''
    def person(self,X1):
        feature_name_s = X1.columns.values
        n = X1.shape[1]
        corr1 = X1.corr()
        dic = dict()
        dic_y = dict()

        for i in range(n):
            name_ = feature_name_s[i]
            for j in range(i+1,n):
                name_2 = feature_name_s[j]
                if i == 0:
                    dic_y.update({name_+"-"+name_2:corr1[name_][name_2]})
                else:
                    dic.update({name_+"-"+name_2:corr1[name_][name_2]})

        return dic,dic_y

    '''该方法使用卡方验证验证两标称数据间相关性，且注意输入值不能为负数'''
    def Chi_squared(self,X1,X2):
        n = X1.shape[1]
        if n == 1:
            X1 = X1.reshape(X1.shape[0],1)
        model = SelectKBest(chi2,k=n)

        model.fit(X1,X2)
        data = model.scores_
        p = model.pvalues_
        return data,p

    def printsort(self,dic):
        '''附带输出'''
        for key in dic.keys():
            print key
            for itme in dic[key]:
                print itme[0],itme[1]
            print "---"


#test
# df = pd.read_csv("/home/yhw/桌面/compa/光伏功率预测/data/train_1.csv")
# month = []
# hour = []
# for sj in df["时间"]:
#     month.append(sj[5:7])
#     hour.append(sj[10:13]+sj[14:16])
# df.insert(0,"月份",month)
# df.insert(0,"小时",hour)
#
# select = Select(0)
#
# #dic = select.feature_sort_statistics_b(df[["月份","风向"]],df[["小时"]])
# #select.printsort(dic)
# #dic = select.feature_sort_statistics_s(df[["温度","湿度","压强","风速"]])
# # from sklearn.ensemble import RandomForestRegressor
# # rf = RandomForestRegressor()
# X = df[["温度","压强","湿度"]]
# y = df[["风速"]]
# # dic = select.feature_sort_tree(rf,X,y)
#
# from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV,RidgeCV
# l = LassoLarsCV([0.1,0.2,0.3])
# dic = select.feature_sort_linear(l,X,y)
# select.printsort(dic)
