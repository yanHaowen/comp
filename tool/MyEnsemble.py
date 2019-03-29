#coding=utf-8
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,StratifiedKFold,ShuffleSplit,train_test_split
import xgboost
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
from sklearn.model_selection import GridSearchCV
import time
from sklearn import ensemble
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV   # Lasso回归,LassoCV交叉验证实现alpha的选取，LassoLarsCV基于最小角回归交叉验证实现alpha的选取
from sklearn.ensemble import IsolationForest

class myEnsemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T,number):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(self.n_folds, shuffle=True, random_state=2016).split(X,y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], len(folds)))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                print(number,"------------")
                print(explained_variance_score(np.array(y_holdout),y_pred))
                print(mean_squared_error(np.array(y_holdout),y_pred))
                print(mean_absolute_error(np.array(y_holdout),y_pred))

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]

            S_test[:, i] = S_test_i.mean(1)
        if self.stacker == None:
            return S_test.mean(1)
        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict(S_test)[:]
        return y_pred
