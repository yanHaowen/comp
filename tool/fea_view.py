#coding=utf-8
from pyecharts import Scatter,Bar,Boxplot,Page
import numpy as np
import pandas as pd
class View:

    '''category:变量，描述数据是 离散 结果还是 连续 结果，0是标称数据，1是连续数据'''
    def __init__(self,category):
        self.category = category

    '''每个特征值的分布情况'''
    '''默认传入的都是pandas对象'''
    def feature_view_bar(self,X,n):
        feature_name = X.columns.values
        page = Page()
        for f in feature_name:
            if self.category[f] == 1:
                range,data = self.split_data(X[f],n)
            else:
                range,data = self.classification_split(X[f])
            bar = Bar(f)
            bar.add("",range,data,is_stack=True)
            page.add(bar)

        page.render("feature_view_bar.html")

    '''Boxplot来进行分布观察'''
    def feature_view_box(self,X):
        feature_name = X.columns.values
        page = Page()
        boxp = Boxplot()
        for f in feature_name:
            if self.category[f] == 1:
                boxp.add("",f,boxp.prepare_data([X[f].values]))
        page.add(boxp)
        page.render("/home/yhw/桌面/compa/光伏功率预测/feature_view_box.html")

    '''对坐标变量进行散点图绘制，对于y是离散那么绘制特征值两两间散点图，如果y是连续型绘制特征与y的散点图'''
    '''数据量太大建议对数据进行分段抽取'''
    def featur_view_scatter(self,X,y):
        feature_name = X.columns.values
        feature_name_y = y.columns.values[0]
        X.insert(0,feature_name_y,y)
        data = X
        page = Page()
        for f in feature_name:
            if self.category[feature_name_y] == 1 and self.category[f] == 1:
                scatter = Scatter(f)
                scatter.add("",X[f].values,y.values)
                page.add(scatter)
            if self.category[feature_name_y] == 0 and self.category[f] == 1:
                for f2 in feature_name:
                    if f != f2 and self.category[f2] == 1:
                        scatter = Scatter(f,width=1000,height=1000)
                        y_value = y.values.reshape([y.values.shape[0]])
                        rangs,ff = self.classification_split(y_value)
                        for r in rangs:
                            scatter.add(r,data[data["y"] == r][f],data[data["y"] == r][f2])
                        page.add(scatter)
        page.render("/home/yhw/桌面/compa/光伏功率预测/featur_view_scatter.html")
    #------------------------------------------------help_function-------------------------------------------------------#

    '''对离散数据进行分解，把具体类别和出现次数进行返回'''
    def classification_split(self,X):
        data = dict()
        for itme in X:
            if not data.has_key(itme):
                data.update({itme:0})
            else:
                nx = data.get(itme)
                data.update({itme:nx+1})
        rangs = data.keys()
        data = data.values()
        return rangs,data

    '''对连续数据离散化，n代表分装为多少个袋'''
    def split_data(self,X,n):
        min = X.min()
        max = X.max()
        med = (max - min)/n

        data = np.zeros(shape=[n,])
        for itme in X:
            for i in range(0,n):
                if itme >= med*i and itme < med*(i+1):
                    nx = data[i]
                    data[i] = nx+1
                    break
        ranges = [str(med*i)+"-"+str(med*(i+1)) for i in range(n)]
        return ranges,data

#测试数据
'''
dic = dict({"风向":0,"风速":1,"y":0,"温度":1})
view = View(dic)
df = pd.read_csv("/home/yhw/桌面/compa/光伏功率预测/data/train_1.csv")
#view.feature_view_box(df[["风向","风速"]])
y=np.ones(shape=[66859])
y = pd.DataFrame({"y":y})
view.featur_view_scatter(df[["温度","风速"]],y)
