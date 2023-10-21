import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

#这个库用来画热力图
import seaborn as seab

#这里实现的Transformer里面的自注意力层机制层


class Self_Attention(nn.Module):
    
    def __init__(self,atten_mask,input_size=512,hidden_size=512):
        super(Self_Attention,self).__init__()
        
        
        #Query的变换
        self.Q=nn.Linear(in_features=input_size,out_features=hidden_size)
        
        #Key的变换
        self.K=nn.Linear(in_features=input_size,out_features=hidden_size)
        
        #Value的变换
        self.V=nn.Linear(in_features=input_size,out_features=hidden_size)
        
        #MASK矩阵
        self.atten_mask=atten_mask
        
        #输入的每一个的向量维度
        self.input_size=input_size
        
        #映射完后每一个点向量的维度
        self.hidden_size=hidden_size
    def forward(self,x):
        '''
        输入x的形状是 [B,L,C]
        input_size就是channel，可以想到每一个时间点都有一个向量表示%gui
        
        '''
        '''
        首先让我们定义Q,K,V的矩阵，这些矩阵的形状都是[input_size,hidden_size]
        将输入空间映射到输出空间
        '''
        #获取三个维度的大小
        B,L,C=x.shape
        
        #做QKV的变换
        Q=self.Q(x)
        K=self.K(x)
        V=self.V(x)
        
        '''
            torch.matmul(Q,K.transpose(1,2))表示的是Q矩阵和K矩阵的矩阵乘法
            Q的维度为 ：[B,L,C]
            K.transpose(1,2)维度为:[B,C,L]
            乘完之后就是[B,L,L]    L*L就是注意力矩阵了
            
            torch.sqrt(torch.full((L,L),self.hidden_size)) 会生成一个L * L 的矩阵，里面的值都是self.hidden_size。
            torch.sqrt取根号，这里做关于尺度的缩放，是为了消除向量表征长度对于注意力的影响
            
        '''
        q_k=torch.matmul(Q,K.transpose(1,2))/torch.sqrt(torch.full((L,L),self.hidden_size))
        
        #进行掩码的mask
        '''
        我们这里对于mask矩阵的要求是：如果你要mask这一个点，那么这个点就填充为 -inf ，否则填充为0
        '''
        mask_q_k=torch.where(self.atten_mask==1,-float('inf'),q_k)
        
        #经过softmax -inf变为0
        atten_matrix=F.softmax(mask_q_k,dim=2)
        
        #画出注意力热力图
        seab.heatmap(atten_matrix.detach().cpu().numpy()[0],fmt='1f',cmap='rainbow')
        
        #用注意力矩阵乘以Value举证得出新的序列表示
        output=torch.matmul(atten_matrix,V)
        return output
def generate_mask(x):
        #获取时间序列步长
        _,L,_ =x.shape
        
        #生成一个上三角矩阵，但是对角线为0
        '''
            [0,1,1
             0,0,1
             0,0,0]
             1代表遮盖
             0代表不遮盖
        '''
        mask=torch.triu(torch.full((10,10),1),diagonal=1)
        
        return mask
if __name__=='__main__':
    
    #生成数据
    data=np.random.normal(0,0.1,(32,10,512))
    data=torch.tensor(data).float()
    
    #进行实验
    model=Self_Attention(generate_mask(data),512,512)
    answer=model(data)
    print(answer[0])
        