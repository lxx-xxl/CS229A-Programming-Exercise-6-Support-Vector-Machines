# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 23:06:11 2022

@author: taylo
"""

import os
import scipy.io as scio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.svm import SVC
from numpy import random

#example 1 (看c对decision boundary的影响)用linear kernel
os.chdir(r'C:\Users\taylo\Desktop\ml\programming_exercise_for_ml\ml_ng\6-svm\data')
data = scio.loadmat('ex6data1.mat') #读取出来的data是字典格式

X = data['X']
y = data['y']

#plot 2D dataset
def plot_data(X,y):
    plt.scatter(X[:,0], X[:,1], c = y.flatten(), cmap = 'jet')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
#C=1
svc1 = SVC(C=1, kernel='linear')
#导入数据进行训练
svc1.fit(X,y.flatten())
print(svc1.score(X,y.flatten()))

#plot
def plot_boundary(model):
    x_min,x_max=-0.5,4.5
    y_min,y_max=1.3,5
    #linspace是用来生成俩500*1的list,meshgrid是用来生成网格，xx_1与xx_2里相同位置的两个数字组合成为（x1，x2）
    xx_1,xx_2=np.meshgrid(np.linspace(x_min,x_max,500),
                      np.linspace(y_min,y_max,500))
    #xx_1=(500,500),xx_2=(500,500)
    #xx_1.flatten()=(250000,1);np.c_=(250000,2)<这就跟组了一个假的x一样> z=(250000,1)
    z=model.predict(np.c_[xx_1.flatten(),xx_2.flatten()]) #np.c_按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等
    zz=z.reshape(xx_1.shape) #(500,500)
    plt.contour(xx_1,xx_2,zz)
#plot_boundary(svc1)
#plot_data(X,y)

#C=100
svc100=SVC(C=100,kernel='linear')
svc100.fit(X,y.flatten())
print(svc100.score(X,y.flatten()))
#plot_boundary(svc100)
#plot_data(X,y)

#example 2,看sigma对decision boundary的影响,用Gaussian kernel
data_2 = scio.loadmat('ex6data2.mat')
X_2=data_2['X']
y_2=data_2['y']
#plot_data(X_2,y_2)

#rbf是Gaussian kernel的意思，gamma就是sigma
svc1_kernel=SVC(C=1,kernel='rbf',gamma=1)
svc50_kernel=SVC(C=1,kernel='rbf',gamma=50)
svc1000_kernel=SVC(C=1,kernel='rbf',gamma=1000)
svc1_kernel.fit(X_2,y_2.flatten())
svc50_kernel.fit(X_2,y_2.flatten())
svc1000_kernel.fit(X_2,y_2.flatten())
print(svc1_kernel.score(X_2,y_2.flatten()))
print(svc50_kernel.score(X_2,y_2.flatten()))
print(svc1000_kernel.score(X_2,y_2.flatten()))

def plot_boundary_kernel(model):
    x_min,x_max=0,1
    y_min,y_max=0.4,1
    #linspace是用来生成俩500*1的list,meshgrid是用来生成网格，xx_1与xx_2里相同位置的两个数字组合成为（x1，x2）
    xx_1,xx_2=np.meshgrid(np.linspace(x_min,x_max,500),
                      np.linspace(y_min,y_max,500))
    #xx_1=(500,500),xx_2=(500,500)
    #xx_1.flatten()=(250000,1);np.c_=(250000,2)<这就跟组了一个假的x一样> z=(250000,1)
    z=model.predict(np.c_[xx_1.flatten(),xx_2.flatten()]) #np.c_按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等
    zz=z.reshape(xx_1.shape) #(500,500)
    plt.contour(xx_1,xx_2,zz)

#plot_boundary_kernel(svc1_kernel)
#plot_boundary_kernel(svc50_kernel)
#plot_boundary_kernel(svc1000_kernel)
#plot_data(X_2,y_2)

#example3 find the best C and sigma
data_3 = scio.loadmat('ex6data3.mat')
X_3=data_3['X']
y_3=data_3['y']
Xval_3=data_3['Xval']
yval_3=data_3['yval']
#plot_data(X_3,y_3)

#注意：获取到的最优参数组合不只有一组，更改候选值的顺序，最佳参数组合及其对应的决策边界也会改变
C_3=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30,100]
gamma_3=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30,100]
best_score=0
best_params=(0,0)
for c in C_3:
    for gamma in gamma_3:
        svc3_kernel=SVC(C=c,kernel='rbf',gamma=gamma)
        svc3_kernel.fit(X_3,y_3.flatten())  #use x y to fit
        score=svc3_kernel.score(Xval_3,yval_3.flatten())    #use xval yval to test
        if score>best_score:
            best_score=score
            best_params=(c,gamma)
print(best_score,best_params)
c_3,gamma_3=best_params
svc3_kernel=SVC(C=c_3,kernel='rbf',gamma=gamma_3)
svc3_kernel.fit(X_3,y_3.flatten())
score_3=svc3_kernel.score(Xval_3,yval_3.flatten())
def plot_boundary3_kernel(model):
    x_min,x_max=-0.6,0.4
    y_min,y_max=-0.7,0.6
    #linspace是用来生成俩500*1的list,meshgrid是用来生成网格，xx_1与xx_2里相同位置的两个数字组合成为（x1，x2）
    xx_1,xx_2=np.meshgrid(np.linspace(x_min,x_max,500),
                      np.linspace(y_min,y_max,500))
    #xx_1=(500,500),xx_2=(500,500)
    #xx_1.flatten()=(250000,1);np.c_=(250000,2)<这就跟组了一个假的x一样> z=(250000,1)
    z=model.predict(np.c_[xx_1.flatten(),xx_2.flatten()]) #np.c_按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等
    zz=z.reshape(xx_1.shape) #(500,500)
    plt.contour(xx_1,xx_2,zz)

#plot_boundary3_kernel(svc3_kernel)
#plot_data(X_3,y_3)

#spam classifier 预处理没做
spamTrain = scio.loadmat('spamTrain') #读取出来的data是字典格式
Xtrain=spamTrain['X']   #(4000,1899) 1899个词，有出现垃圾词就1，没就0； 4000个example
ytrain=spamTrain['y']   #(4000,1) 垃圾邮件为1，非垃圾0

spamTest = scio.loadmat('spamTest')
Xtest=spamTest['Xtest'] #(1000,1899)
ytest=spamTest['ytest'] #(1000,1)

best_score_spam=0
best_param_spam=0
for c_spam in C_3:
    svm_spamtest=SVC(C=c_spam,kernel='linear')
    svm_spamtest.fit(Xtrain,ytrain.flatten())
    score_spamtest=svm_spamtest.score(Xtest,ytest.flatten())
    if score_spamtest>best_score_spam:
        best_score_spam=score_spamtest
        best_param_spam=c_spam
        print(best_score_spam,best_param_spam)

svc_spam=SVC(C=best_param_spam,kernel='linear')
svc_spam.fit(Xtrain,ytrain.flatten())

score_spamtest=svc_spam.score(Xtest,ytest.flatten())
score_spamtrain=svc_spam.score(Xtrain,ytrain.flatten())
print('train: ',score_spamtrain,'; test: ',score_spamtest)
