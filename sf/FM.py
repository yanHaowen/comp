from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
import numpy as np
'''
train = [
    {"user": "1", "item": "5", "age": 19},
	{"user": "2", "item": "43", "age": 33},
	{"user": "3", "item": "20", "age": 55},
	{"user": "4", "item": "10", "age": 20},
]

v = DictVectorizer()
x = v.fit_transform(train)

y = np.repeat(1.0,x.shape[0])
fm = pylibfm.FM()
fm.fit(x,y)
p = fm.predict(v.transform({"user": "1", "item": "10", "age": 24}))
print p
'''

def loadData(filename,path="ml-100k/"):
    data = []
    y = []
    users = set()
    items = set()
    with open(path+filename) as f:
        for line in f:
            (user,movieid,rating,ts) = line.split('\t')
            data.append({"user_id":str(user),"movie_id":str(movieid)})
            y.append(float(rating))
            users.add(user)
            items.add(movieid)

        return (data,np.array(y),users,items)

(train_data, y_train, train_users, train_items) = loadData("ua.base")
(test_data, y_test, test_users, test_items) = loadData("ua.test")
v = DictVectorizer()
x_train = v.fit_transform(train_data)


x_test = v.fit_transform(test_data)

fm = pylibfm.FM(num_factors=10,num_iter=20,verbose=True,task='regression')

fm.fit(x_train,y_train)
print fm.predict(v.transform({"user_id":'1',"movie_id":'1'}))
