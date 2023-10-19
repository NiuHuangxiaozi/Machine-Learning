#本模块实现了transformer当中的LayerNomalization模块
import numpy as np
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    
    def __init__(self,eps=1e-05,affine=True):
        super(LayerNorm,self).__init__()
        self.eps =eps  #防止方差在分母的时候取0
        self.affine=affine # _y _b是否计算梯度，_y,_b在pytorch的源码中是self.weight和self.bias
        self._y=None # 形状为 [L,C] ，L,C在下面forward介绍
        self._b=None # 形状为 [L,C] ，L,C在下面forward介绍
        
    def forward(self,input):
        '''
            input 的形状是：[B,L,C] 
            B: Batchsize 每一批的大小
            L: Length 时间序列的长度或者NLP中语句的长度
            C: 每一个时间点的向量表示或者在NLP中每一个词元的向量表示
        '''
        
        B,L,C=input.shape 
        
        #初始化_y和_b
        self.initial_y_b(L,C) 
        
        #计算每一小批的均值，每一小批就是L*C个元素，这里计算L*C个元素的均值
        mean=torch.mean(input,keepdim=True,dim=[-2,-1])
        
        #计算每一小批的方差，每一小批就是L*C个元素，这里计算L*C个元素的方差
        std=torch.mean((input - mean) ** 2, dim=[-2,-1], keepdim=True)+self.eps
        
        #概率论中学过，正态分布的归一化
        result=(input-mean)/torch.sqrt(std)*self._y+self._b
        
        return result
    
    #初始化_y和_b
    def initial_y_b(self,L,C):
        
        # requires_grad在这里设置weight和bias是否计算梯度
        if self.affine:
            self._y=torch.ones((L,C),requires_grad=True)
            self._b=torch.zeros((L,C),requires_grad=True)
        else:
            self._y=torch.ones((L,C),requires_grad=False)
            self._b=torch.zeros((L,C),requires_grad=False)
    
    
 #下面这个是测试与pytorch标准库是否一致

if __name__=='__main__':
    
    #定义一个时间序列，形状为[64,10,512]，允许计算梯度
    data=torch.normal(mean=0,std=0.1,size=(64,10,512),requires_grad=True)
    
    #定义自己写的层归一化
    my_Layer_Normalization=LayerNorm(eps=1e-05,affine=True)
    
    #使用官方层归一化
    official_Layer_Normalization=nn.LayerNorm([10,512],eps=1e-05,elementwise_affine=True)
    
    #分别进行归一化
    my_result=my_Layer_Normalization(data)
    official_result=official_Layer_Normalization(data)
    
    #进行反向梯度传播
    my_result.sum().backward()
    official_result.sum().backward()
    
    #打印对应结果
    print(my_result)
    print(official_result)
    
    #打印对应梯度
    print(my_Layer_Normalization._y.grad)
    print(my_Layer_Normalization._b.grad)
    print(official_Layer_Normalization.weight.grad)
    print(official_Layer_Normalization.bias.grad)
    