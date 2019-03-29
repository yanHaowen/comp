#coding=utf-8
import numpy as np
import pandas as pd

def data_create():
    data = []
    for i in range(100):
        data.append(np.random.randint(1,10,(1,4))[0])


def delete(k_suport,k_data):

    delete = []
    temp_suport = k_suport
    for key in temp_suport:
        value = temp_suport.get(key)

        if  value/float(sum) <theta:

            delete.append(k_data[key])
    for j in delete:
        k_data.remove(j)



def combination(k,data):
    keys = data

    data = {}
    new_keys = []

    index = 1
    for itme in keys:
        for i in range(index,len(keys)):
            if k == 2:
                new_keys.append(list(set([itme,keys[i]])))
            else:
                ls = sorted(list(set(itme+keys[i])))
                if ls in new_keys:
                    continue
                new_keys.append(ls)
        index = index+1

    delete = []
    for itme in new_keys:
        if len(itme) != k:
            delete.append(itme)
    for dele in delete:
        new_keys.remove(dele)
    return new_keys

def Apropr(theta,data):
    theta = 0.12
    k_data = []
    k_suport = {}
    k_old = {}
    k_return = {}
    n = 0
    sum = 0
    for both in data:
        for itme in both:
            sum = sum+1
            if itme not in k_data:
                k_data.append(itme)
                k_suport.update({n:1})
                n = n+1
            else:
                index_ = k_data.index(itme)
                IF = k_suport.get(index_)
                k_suport.update({index_:IF+1})


    k_old = k_suport
    delete(k_suport,k_data)
    print k_data

    #几元组
    k=1
    while True:
        k = k+1
        if k_data == []:
            k_return = k_old
            break
        if len(k_data) == 1:
            k_return == k_data
            break
        print k,k_data
        k_data = combination(k,k_data)
        k_suport = {}
        for itme in k_data:
            for min in data:
                if itme in min:
                    index_ = k_data.index(itme)
                    if index_ not in k_suport:
                        k_suport.update({index_:1})
                    else:
                        n_k = k_suport.get(index_)
                        k_suport.update({index_:n_k+1})
        delete(k_suport,k_data)
        k_old = k_data
