#!-*- coding:utf-8 -*-
import numpy as np
import os, glob, cv2
from PIL import Image #用于处理图片的模块
import pandas as pd
import scipy.io as sio #用于读取.mat数据的模块
#Dataset()是为了获得所有图像的像素数组，而getdataset是为了将各张图片的像素标签放入对应的数组中
#参考网页https://blog.csdn.net/baidu_36161077/article/details/73864519
class Dataset():
    def __init__(self):
        self.classes = ['apple', 'orange', 'peach'] # 文件夹的顺序要预先排列好，因为下面的遍历顺序是按照classes列表中的顺序进行遍历的
        self.cwd = os.getcwd() #得到的主目录,比如我的主目录是‘C:/Users/Administrator/eclipse-workspace/first/album’，那么self.cwd就等于'C:/Users/Administrator/eclipse-workspace/first/album'
        self.arr = [[]]
        self.a = []

    def img2array(self):
        #name为class的类别, glob.glob('c:/pic*.txt')获得C盘pic文件夹下的所有txt格式的文件,返回的是列表格式，该列表的元素为txt格式文件的绝对路径
        #print(glob.glob(self.cwd + '/data/' + 'cat'+'/' + '*.jpg'))
        for index, name in enumerate(self.classes): 
            class_path = self.cwd + '/data/' + name + '/'#这是我的图片文件夹的存放路径
            for infile in glob.glob(class_path + '*.jpg'): # 遍历所有文件夹下的jpg格式的图片
                file,ext = os.path.splitext(infile) #将文件名字和格式进行拆分
                img = Image.open(infile) #使用Image模块打开文件,输入的参数为图片的路径

                #利用cv.2对图像进行尺寸变换，再转换成PIL的图片格式
                #image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                #image = cv2.resize(image, (32, 32))
                #img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                img = img.resize([32, 32]) #将图片大小都设置成 32 * 32 大小的
                
                r,g,b = img.split() #将彩色图片拆分为三色通道，r,g,b分别为red,green,black
                r_array = np.array(r).reshape([1024]) #利用numpy将三色通道中的图片信息数据化。
                g_array = np.array(g).reshape([1024]) # 将32 * 32的三色通道的图像都reshape成一行
                b_array = np.array(b).reshape([1024])
                merge_array = np.concatenate((r_array,g_array,b_array)) #将三色通道的图片信息拼接，这就是一张图片的所有信息, shape = [1,1024 * 3]
                if self.arr == [[]]:
                    self.arr = [merge_array] #第一张图片放到列表中，后面经过numpy 中的concatenate会变成ndarray格式的
                    continue
                self.arr = np.concatenate((self.arr, [merge_array]),axis = 0) #将所有文件夹下的所有的jpg格式的图片信息拼接成一个数组
        return self.arr
        
    def getdataset(self):
        img_info = self.img2array() # 通过调用 img2array方法来得到图片信息       
        #f = open('/home/nw/data/dmos.mat','rb') #将相应文件夹下的标签信息读取出来
        #因为文件夹下的标签矩阵是mat格式的，使用scipy下的io模块中的loadmat可以直接将标签矩阵读写出来
        #注意上面classes中的文件夹的顺序要准确，要和标签的顺序一一对应。文件夹和标签的对应顺序要预先明确
        #labelset = sio.loadmat(f)
        #labelset = labelset['dmos'] #load出来的是字典的形式。使用索引将标签信息取出来
        #labelset = np.reshape(labelset,[982,1]) #将标签处理成shape = [样本数，1]
        A = np.zeros((30, 1), np.uint8) #A为苹果的标签
        B = np.ones((30,1), np.uint8) #B为橘子的标签
        C = 2 * B #C为桃子的标签
        W = np.concatenate((A, B), axis = 0)
        W1 = np.concatenate((W, C), axis = 0)
        #W1的大小是90x1的, np.concatenate()的axis=0为行拼接
        dataset = np.concatenate((img_info,W1), axis = 1) #将图片信息和标签信息拼接起来
        return dataset

Dataset().img2array()
    