#引入numpy高效矩阵计算库
import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import os
import math

class Linear_regression:
    def __init__(self,regularization='L2',lam=0.0):
        
        self.weight=None #就是w
         
        self.train_data=None
        self.train_label=None
        self.test_data=None
        self.test_label=None

        #正则化项，我们这里实现的是L2正则化
        self.reg=regularization
        #正则化项的系数
        self.lam=lam

    #训练w和b
    def train(self,train_data,train_label):
        # self.train_data  n*m矩阵  输入的训练数据集，n表示数据样本数 ，m代表的是每一个样本的维度
        self.train_data=train_data
        self.train_label=train_label
        #在数据的特征维度上扩充一维，即n*m变为n*(m+1)
        X=self.extend_dim(self.train_data)
        y=self.train_label

        #单位矩阵
        n,m_1=X.shape
        I=np.eye(m_1)

        #下面三行就是公式
        if self.reg == 'L2':
            XT_X=np.linalg.inv(np.matmul(X.T,X)+self.lam*I)#求逆
            XT_y=np.matmul(X.T,y)   
        self.weight=np.matmul(XT_X,XT_y)

    #拟合阶段
    def predict(self,test_data):
        self.test_data=test_data
        X=self.extend_dim(self.test_data) #与训练的时候一样
        result=np.matmul(X,self.weight)
        self.test_label=result
        return result

    #这个函数判断是否可以简单可视化，如果是一维就可以画出一个二维的图
    def is_visualize(self):
        return self.train_data is not None and self.train_data.shape[1]==1

    #将线性回归可视化
    def draw(self):
            #判断是否可以可视化
            if self.is_visualize():
                #画出traindata和testdata的散点图，颜色分别是：绿色和红色
                plt.scatter(self.train_data.reshape(-1),self.train_label.reshape(-1),c='g',label="train data")
                plt.scatter(self.test_data.reshape(-1),self.test_label.reshape(-1),c='r',label="predict_data")
                plt.legend(loc='best')

                #生成x的list
                line_x=np.array([item for item in range(self.train_data.min(),self.train_data.max()+1)])
                #线性拟合，算出相应的y
                line_y=np.matmul(self.extend_dim(line_x.reshape(-1,1)),self.weight)

                #画出拟合的直线
                plt.plot(line_x,line_y,c='y',linestyle='--')
            else:
                print("Could not be visilized!")
            
    #在数据的特征维度上扩充一维，即n*m变为n*(m+1)         
    def extend_dim(self,data):
        _One=np.ones([data.shape[0]]).reshape(-1,1)
        return np.concatenate((data,_One),axis=1)

    #相应的数据清零，可以进行下一次拟合
    def Zero_example(self):
        self.train_data=None
        self.train_label=None
        self.test_data=None
        self.test_label=None
        self.weight=None


#我们下面使用波士顿房价数据集，使用sklearn标准库中的线性回归与自己的实现进行对比，以此来验证实现是否正确。
if __name__=='__main__':
        #加载数据集
        path=os.getcwd()+"/"+"boston_housing_data.csv"
        boston_housing=pd.read_csv(path)
        #去除当中的Nan
        boston_housing.dropna(inplace=True)

        #获取我们拟合的目标
        y=np.array(boston_housing['MEDV'])
        #除去MEDV，其他属性都是x
        X=np.array(boston_housing.drop(['MEDV'],axis=1))

        #划分训练集和测试集
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=10)

        #对数据进行MinMax归一化
        MinMax_Scale=preprocessing.MinMaxScaler()
        X_train=MinMax_Scale.fit_transform(X_train)
        X_test=MinMax_Scale.fit_transform(X_test)
        y_train=MinMax_Scale.fit_transform(y_train.reshape(-1,1))
        y_test=MinMax_Scale.fit_transform(y_test.reshape(-1,1))

        #定义自己的模型
        myLR=Linear_regression()
        myLR.train(X_train,y_train)
        myprediction=myLR.predict(X_test)
        
        #使用官方模型
        LR = LinearRegression()
        # 使用训练数据进行参数估计
        LR.fit(X_train, y_train)
        # 使用测试数据进行回归预测
        prediction = LR.predict(X_test)
        
        
        number,_=prediction.shape

        #下面验证是否输出一致
        for index in range(number):
            if not math.isclose(prediction[index][0],myprediction[index][0]):
                print("My LinearRegression model doesn't achieve the standard model.")
        print("My LinearRegression model achieve the standard model.")
        
        
        
