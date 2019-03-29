#coding=utf-8
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split,cross_val_score


class TSVM:
    def initial(self,kernel = 'linear'):
        #初始化 传入核函数
        self.Cl,self.Cu = 1.5,0.001
        self.kerner = kernel
        #C惩罚系数
        self.clf = svm.SVC(C = 3,kernel=kernel)

    def load(self,model_path='./TSVM.model'):

        #加载训练好的模型
        self.clf = joblib.load(model_path)

    def train(self,X1,X2,Y1):
        #X1 Y1 有标签的数据集合
        #X2 无标签的数据集合

        N = len(X1) + len(X2)
        sample_weight = np.ones(N)
        sample_weight[len(X1):] = self.Cu

        #用标签数据训练一个SVM

        self.clf.fit(X1,Y1)
        Y2 = self.clf.predict(X2)

        Y2 = np.expand_dims(Y2 , axis=1)
        X2_id = np.arange(len(X2))

        X3 = np.vstack([X1, X2])
        Y3 = np.vstack([Y1, Y2])

        while self.Cu < self.Cl:
            self.clf.fit(X3 , Y3 ,sample_weight=sample_weight)
            while True:
                #返回到超平面的距离，也就是x*w+b的 值
                y2_d = self.clf.decision_function(X2)
                y2 = Y2.reshape(-1)
                epsilon = 1 - y2 * y2_d
                #epsilon越大表名出错的可能性越大
                positive_set, positive_id = epsilon[y2>0], X2_id[y2>0]
                negative_set, negative_id = epsilon[y2 == 0], X2_id[y2 == 0]
                positive_max_id = positive_id[np.argmax(positive_set)]
                negative_max_id = negative_id[np.argmax(negative_set)]
                a, b =epsilon[positive_max_id],epsilon[negative_max_id]

                if a>0 and b>0 and a+b >2.0:
                    y2[positive_max_id] = y2[negative_max_id] * -1
                    y2[negative_max_id] = y2[positive_max_id] * -1
                    y2 = np.expand_dims(y2,1)
                    y3 = np.vstack([Y1,Y2])
                    self.clf.fit(X3,Y3,sample_weight=sample_weight)
                else:
                    break
                self.Cu = min(2*self.Cu,self.Cl)
                sample_weight[len(X1):] = self.Cu

    def score(self,X,Y):
        #返回一个准确率
        return self.clf.score(X,Y)

    def predict(self,X):
        #预测X的标签
        return self.clf.predict(X)

    def save(self,path='./TSVM.model'):
        joblib.dump(self.clf,path)

if __name__ == '__main__':
    model = TSVM()
    model.initial()

    np.random.seed(100)
    X1_R = 0.8 * np.random.randint(1,100,(100,4))
    X1_N = 2 * np.random.randint(1,100,(100,4))
    Y1_r = np.ones((100,1))
    Y1_n = np.zeros((100,1))
    X1 = np.vstack([X1_R,X1_N])
    Y1 = np.vstack([Y1_r,Y1_n])
    X2_R = 0.8 * np.random.randint(1,100,(100,4))
    X2_N = 2 * np.random.randint(1,100,(100,4))
    X2 = np.vstack([X2_R,X2_N])

    model.train(X1[:200],X2[:200],Y1[:200])
    score = model.score(X1[180:],Y1[180:])
    print(score)
