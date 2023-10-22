
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

#tqdm库可以显示进度条
from tqdm import *
import os

class Self_Attention(nn.Module):
    
    def __init__(self,atten_mask,input_size=512,hidden_size=512,heads=8):
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
        
        #多头注意力的头的个数
        self.heads=heads
        
        #加载到哪个设备上
        self.device=None
        
    def forward(self,x):
        '''
        输入x的形状是 [B,L,C]
        input_size就是channel，可以想到每一个时间点都有一个向量表示%gui
        
        '''
        '''
        首先让我们定义Q,K,V的矩阵，这些矩阵的形状都是[input_size,hidden_size]
        将输入空间映射到输出空间
        '''
        
        #获取输入数据所在的设备
        self.device=x.device
        
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
        assert(self.hidden_size%self.heads==0)
        
        
        each_hidden_dim=int(self.hidden_size/self.heads)
        
        multi_Q=list(Q.split(each_hidden_dim,dim=2))
        multi_K=list(K.split(each_hidden_dim,dim=2))
        multi_V=list(V.split(each_hidden_dim,dim=2))
        
        #缩放矩阵，要被除
        scale_matrix=torch.sqrt(torch.full((L,L),each_hidden_dim)).to(self.device)
        
        output=[]
        for index in range(len(multi_Q)):

            #自注意力的过程
            mini_q_k=torch.matmul(multi_Q[index],multi_K[index].transpose(1,2))/scale_matrix

            #进行mask的过程
            if self.atten_mask is not None:
                mask_mini_q_k=torch.where(self.atten_mask==1,-float('inf'),mini_q_k)
            else:
                mask_mini_q_k=mini_q_k
            
            #经过softmax -inf变为0
            mini_atten_matrix=F.softmax(mask_mini_q_k,dim=2)

            #注意力矩阵乘以V矩阵
            mini_output=torch.matmul(mini_atten_matrix,multi_V[index])
            
            output.append(mini_output)
            
        #多头注意力机制，最后将所有的头拼接在一起
        result=torch.cat(output,dim=2)
        
        return result
    
class LayerNorm(nn.Module):
    
    def __init__(self,eps=1e-05,affine=True):
        super(LayerNorm,self).__init__()
        self.eps =eps  #防止方差在分母的时候取0
        self.affine=affine # _y _b是否计算梯度，_y,_b在pytorch的源码中是self.weight和self.bias
        self._y=None # 形状为 [L,C] ，L,C在下面forward介绍
        self._b=None # 形状为 [L,C] ，L,C在下面forward介绍
        
        self.device=None #记录输入的数据是在gpu还是cpu上
        
    def forward(self,input):
        '''
            input 的形状是：[B,L,C] 
            B: Batchsize 每一批的大小
            L: Length 时间序列的长度或者NLP中语句的长度
            C: 每一个时间点的向量表示或者在NLP中每一个词元的向量表示
        '''
        #记录设备
        self.device=input.device
        
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
            self._y=torch.ones((L,C),requires_grad=True).to(self.device)
            self._b=torch.zeros((L,C),requires_grad=True).to(self.device)
        else:
            self._y=torch.ones((L,C),requires_grad=False).to(self.device)
            self._b=torch.zeros((L,C),requires_grad=False).to(self.device)
            
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
        x=self.linear_1(x)
        x=self.relu1(x)
        x= self.drop1(x)
        x=self.linear_2(x)
        return x

#这个类表示的是一个transformer的encoderlayer的设计，后面的encoder就是将这个模块不断的堆叠咋在一起
class EncoderLayer(nn.Module):
    
    def __init__(self,input_dim,output_dim,heads):
        super(EncoderLayer,self).__init__()
        
        # atten_mask,input_size=512 , hidden_size=512,heads=8
        #子注意力模块
        self.self_attention=Self_Attention(None,input_size=input_dim,hidden_size=output_dim,heads=heads)
        #层归一化层
        self.layernormalization1=LayerNorm()
        #feedforward层
        self.feed= FeedForward(output_dim,output_dim*4)
        self.layernormalization2=LayerNorm()
        
    def forward(self,x):
        #这个过程一看论文就知道了
        x=self.layernormalization1(x+self.self_attention(x))
        x=self.layernormalization2(x+self.feed(x))    
        return x

#encoder，堆叠多个encoderlayer
class Encoder(nn.Module):
    
    def __init__(self,input_dim,hidden_dim,heads,Layer_number,Seq_length=28,category=10):
            super(Encoder,self).__init__()
            
            self.device='cuda:0'if torch.cuda.is_available() else 'cpu'
            self.input_dim=input_dim
            self.hidden_dim=hidden_dim
            self.Layer_number=Layer_number
            self.Seq_length=Seq_length
            
            self.linear=nn.Linear(in_features=self.input_dim,out_features=self.hidden_dim)
            self.Encoderlayers=nn.ModuleList([EncoderLayer(self.hidden_dim,self.hidden_dim,heads) for _ in range(self.Layer_number)])
            
            self.linear2=nn.Linear(in_features=self.Seq_length*self.hidden_dim,out_features=category)
            
    def forward(self,x):
            '''
                首先第一个linear层主要的作用就是将原来的embedding升维到模型里面的隐层
                维度的转化过程就是：[B,L,C] -----> [ B,L,H ] ( [16,10,28]-----[16,10,512] )
                Transformer 里面的维度一般设置的是512
            '''
            x=self.linear(x)
            '''
                通过一个for循环不断遍历每一层的encoderlayer
            '''
            for index in range(self.Layer_number):
                x=self.Encoderlayers[index](x)

            #维度变化是 [B，L,C]---->[B,L*C] 这样就可以用linear层进行分类里了
            x=x.view(x.shape[0],-1)
            x=self.linear2(x)
        
            #用softmax进行分类
            x=F.softmax(x,dim=1)
            
            return x
            
        




if __name__=="__main__":
    #设备
    device='cuda:0'if torch.cuda.is_available() else 'cpu'


    #加载数据集
    cwd_path =os.getcwd()
    data_path=cwd_path+'/data/'
    print(data_path)

    #对下载下来的图片准备进行什么操作，[]里面是操作列表，比如:ToTensor()是将一个对象转化为tensor；Normalize是将每一个像素进行归一化
    transfor=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize(mean=[0.5],std=[0.5])])

    '''
        torchvision里面集成了MNIST数据集，我们直接在datasets里面找
    '''
    train_data=torchvision.datasets.MNIST(data_path,train=True,transform=transfor,download=True)
    test_data=torchvision.datasets.MNIST(data_path,train=False,transform=transfor)

    BATCH_SIZE=256
    train_loader=torch.utils.data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
    test_loader=torch.utils.data.DataLoader(dataset=test_data,batch_size=BATCH_SIZE)
    print(len(train_data))
    
    
    model=Encoder(input_dim=28,hidden_dim=128,heads=8,Layer_number=2,Seq_length=28)
    model=model.to(device)
    
    
    
    loss_func=torch.nn.CrossEntropyLoss()
    optim=torch.optim.Adam(model.parameters(),lr=0.001)
    Epoch=10
    history = {'Test Loss':[],'Test Accuracy':[]}


    for epoch in range(1,Epoch+1):
        #训练部分
        processBar = tqdm(enumerate(train_loader), total=len(train_loader))
        model.train()
    
        for index,(data,label) in processBar:
            data=data.reshape(-1,28,28).to(device)
            label=label.to(device)
        
            outputs=model(data)
            prediction=torch.argmax(outputs,dim=1)
            acc=torch.sum(prediction==label)/data.shape[0]
            loss=loss_func(outputs,label)
        
            optim.zero_grad()
            loss.backward()
            optim.step()
        
            processBar.set_description("[%d/%d] Loss : %.8f Acc：%.8f" %(epoch,Epoch,loss,acc))
        
            if index==len(processBar)-1:
                model.eval()
                with torch.no_grad():
                    total_loss=0.
                    total_right=0
                    for index,(t_data,t_label) in enumerate(test_loader):
                        t_data=t_data.reshape(-1,28,28).to(device)
                        t_label=t_label.to(device)

                        t_outputs=model(t_data)

                        loss=loss_func(t_outputs,t_label)
                        t_prediction=torch.argmax(t_outputs,dim=1)

                        total_loss+=loss
                        total_right+=torch.sum(t_prediction==t_label)
                    average_loss=total_loss/len(test_data)
                    total_acc=total_right/len(test_data)
                    #print(average_loss.item())
                    history['Test Loss'].append(average_loss.item())
                    history['Test Accuracy'].append(total_acc.item())
                    processBar.set_description("[%d/%d] TEST: average_loss %.8f Total_acc %.8f" %(epoch,Epoch,average_loss,total_acc))
            
            
        processBar.close()     
