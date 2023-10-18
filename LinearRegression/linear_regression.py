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
        
        self.weight=None
         
        self.train_data=None
        self.train_label=None
        self.test_data=None
        self.test_label=None
        
        self.reg=regularization
        self.lam=lam
        
    def train(self,train_data,train_label):
        # self.train_data  n*m矩阵  输入的训练数据集，n表示数据样本数 ，m代表的是每一个样本的维度
        self.train_data=train_data
        self.train_label=train_label
        X=self.extend_dim(self.train_data)
        y=self.train_label
        
        n,m_1=X.shape
        I=np.eye(m_1)
        
        if self.reg == 'L2':
            XT_X=np.linalg.inv(np.matmul(X.T,X)+self.lam*I)
            XT_y=np.matmul(X.T,y)
            
        self.weight=np.matmul(XT_X,XT_y)
    def predict(self,test_data):
        
        self.test_data=test_data
        X=self.extend_dim(self.test_data)
        
        result=np.matmul(X,self.weight)
        self.test_label=result
        return result
    
    def is_visualize(self):
        return self.train_data is not None and self.train_data.shape[1]==1
    def draw(self):
            if self.is_visualize():      
                plt.scatter(self.train_data.reshape(-1),self.train_label.reshape(-1),c='g',label="train data")
                plt.scatter(self.test_data.reshape(-1),self.test_label.reshape(-1),c='r',label="predict_data")
                plt.legend(loc='best')

                line_x=np.array([item for item in range(self.train_data.min(),self.train_data.max()+1)])
                
                line_y=np.matmul(self.extend_dim(line_x.reshape(-1,1)),self.weight)
                
                plt.plot(line_x,line_y,c='y',linestyle='--')
            else:
                print("Could not be visilized!")
            
            
    def extend_dim(self,data):
        _One=np.ones([data.shape[0]]).reshape(-1,1)
        return np.concatenate((data,_One),axis=1)
    def Zero_example(self):
        self.train_data=None
        self.train_label=None
        self.test_data=None
        self.test_label=None
    
if __name__=='__main__':
        path=os.getcwd()+"/"+"boston_housing_data.csv"
        boston_housing=pd.read_csv(path)
        boston_housing.dropna(inplace=True)
        
        y=np.array(boston_housing['MEDV'])
        X=np.array(boston_housing.drop(['MEDV'],axis=1))
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=10)
        
        MinMax_Scale=preprocessing.MinMaxScaler()
        X_train=MinMax_Scale.fit_transform(X_train)
        X_test=MinMax_Scale.fit_transform(X_test)
        y_train=MinMax_Scale.fit_transform(y_train.reshape(-1,1))
        y_test=MinMax_Scale.fit_transform(y_test.reshape(-1,1))
        
        myLR=Linear_regression()
        myLR.train(X_train,y_train)
        myprediction=myLR.predict(X_test)
        
        
        LR = LinearRegression()
        # 使用训练数据进行参数估计
        LR.fit(X_train, y_train)
        # 使用测试数据进行回归预测
        prediction = LR.predict(X_test)
        
        
        number,_=prediction.shape
        for index in range(number):
            if not math.isclose(prediction[index][0],myprediction[index][0]):
                print("My LinearRegression model doesn't achieve the standard model.")
        print("My LinearRegression model achieve the standard model.")
        
        
        