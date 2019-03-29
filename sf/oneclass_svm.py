import numpy as np
from sklearn import svm

xx, yy = np.meshgrid(np.linspace(-5,5,500),np.linspace(-5,5,500))

X_train = 0.3 * np.random.randn(100,4)
X_test = 0.3 * np.random.randn(100,4)
#nu设定训练误差（0， 1]
clf = svm.OneClassSVM(nu=0.1,kernel='rbf',gamma=0.1)
clf.fit(X_train)

y_pred = clf.predict(X_train)
n_error = y_pred[y_pred == -1].size

y_pred_t = clf.predict(X_test)
n_error_t = y_pred_t[y_pred_t == -1].size
print n_error,n_error_t
