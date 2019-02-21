#!-*- coding:utf-8 -*-
from sys import path
from data28 import Dataset #data28是文件data28.py，从data28导入Dataset类
import pickle
import numpy as np
path.append('C:/Users/Administrator/eclipse-workspace/first/album/data') #将编译路径添加进去
haha = Dataset() #实例化
dataset = haha.getdataset() #通过调用实例化之后的getdataset方法获得dataset数据集

f = open('C:/Users/XU/eclipse-workspace/pythonlearning/album/data/dataset', 'wb') #先打开一个文件，如果这个文件不存在，就新建这个文件，二进制可写
pickle.dump(dataset, f) #使用pickle的dump将获得的dataset数据倒入到f文件中
f.close()

f = open('data/dataset', "rb") #二进制可读
dataset = pickle.load(f) #使用pickle.load()读取文件中的数据
print(dataset.shape[1])
f.close()

#将标签为0的全部删除
count = 0
dataset1 = np.zeros((1,dataset.shape[1]), np.uint8)
for i in range(dataset.shape[0]):
    #将dataset1与标签非0的行拼接
    if dataset[i][3072] != 0:
        dataset1 = np.concatenate((dataset1, dataset[i].reshape(1, dataset.shape[1])), axis = 0)
#print(dataset1)   
#删除第一行，因为一开始创建的dataset1的二维数组的第一行没用     
dataset1 = np.delete(dataset1, 0, axis = 0)  #np.delete()中dataset是需要做处理的
print(dataset1)
