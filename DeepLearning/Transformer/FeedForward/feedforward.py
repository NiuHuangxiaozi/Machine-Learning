import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt



#这里实现的Transformer里面的 FeedForward 层


class FeedForward(nn.Module):
    def __init__(self,embedding_size,hiddensize,dropout=0.1):
        super(FeedForward,self).__init__()
        
        #输入的隐层
        self.embedding_size=embedding_size
        
        #中间的隐层
        self.hiddensize=hiddensize
        
        #冻结因子，我们冻结某些节点的梯度计算
        self.dropout=dropout
        
        #第一个线性层
        self.linear_1=nn.Linear(in_features=self.embedding_size,out_features=self.hiddensize)
        
        #激活函数
        self.relu1=nn.ReLU()
        
        #冻结算子
        self.drop1=nn.Dropout(self.dropout)
        
        #第二个线性层
        self.linear_2=nn.Linear(in_features=self.hiddensize,out_features=self.embedding_size)
        
    def forward(self,x):
        
        #分解形状
        B,L,C=x.shape
        
        #除了B这个维度，其他铺平
        x=x.view(B,-1)
        
        x=self.linear_1(x)
        x=self.relu1(x)
        x= self.drop1(x)
        x=self.linear_2(x)
        
        #重新恢复[B,L,C]的形状
        x=x.view(B,L,C)
        return x
    
if __name__=='__main__':
    #定义
    feedforward=FeedForward(512*10,4*512*10)
    
    #数据
    data=torch.tensor(np.random.normal(0,0.1,(64,10,512))).float()
    
    #forward过程
    result=feedforward(data)
    
    #打印之后的维度
    print(result.shape)