import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from string import punctuation
from collections import Counter
#nltk是一个自然语言处理库
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import os
import torch.nn.functional as F


#多头注意力机制
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
        
    def forward(self,q,k,v):
        '''
        q,k的形状一定是一样的，v则有可能不一样，但是一般形状也是一样的
        输入q的形状是 [B,L,C]
        input_size就是channel，可以想到每一个时间点都有一个向量表示%gui
        
        '''
        '''
        首先让我们定义Q,K,V的矩阵，这些矩阵的形状都是[input_size,hidden_size]
        将输入空间映射到输出空间
        '''
        
        #获取输入数据所在的设备
        self.device=q.device
        
        #获取三个维度的大小
        B,L,C=q.shape
        
        #做QKV的变换
        Q=self.Q(q)
        K=self.K(k)
        V=self.V(v)
        
        
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

#层归一化的实现
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


#FNN层，就是两层线性层
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
    
    def __init__(self,input_dim,hidden_dim,heads):
        super(EncoderLayer,self).__init__()
        
        # atten_mask,input_size=512 , hidden_size=512,heads=8
        #子注意力模块
        self.self_attention=Self_Attention(None,input_size=input_dim,hidden_size=hidden_dim,heads=heads)
        #归一化层
        self.layernormalization1=LayerNorm()
        #feedforward层
        self.feed= FeedForward(hidden_dim,hidden_dim*4)
        
        self.layernormalization2=LayerNorm()
        
    def forward(self,x):
        #这个过程一看论文就知道了
        x=self.layernormalization1(x+self.self_attention(x,x,x))
        x=self.layernormalization2(x+self.feed(x))    
        return x
    

class DecoderLayer(nn.Module):
    def __init__(self,mask_metrix,input_dim,hidden_dim,heads):
        super(DecoderLayer,self).__init__()
        
        self.mask_metrix=mask_metrix #这是mask矩阵
        self.input_dim=input_dim #这是这一层输入embedding的大小
        self.hidden_dim=hidden_dim #这是这一层输入hidden embedding的大小，也是这一层输出的大小
        self.heads=heads
        
        
        self.mask_self_attention=Self_Attention(self.mask_metrix,input_size=self.input_dim,hidden_size=self.hidden_dim,heads= self.heads)
        
        
        self.mask_self_attention_1=Self_Attention(None,input_size=self.input_dim,hidden_size=self.hidden_dim,heads= self.heads)
        
        
        self.feed= FeedForward(self.hidden_dim,self.hidden_dim*4)
        
        #归一化层
        self.layernormalization1=LayerNorm()
        self.layernormalization2=LayerNorm()
        self.layernormalization3=LayerNorm()
        
    def forward(self,de_input,en_input):
        de_input=self.layernormalization1(de_input+self.mask_self_attention(de_input,de_input,de_input))
        de_input=self.layernormalization2(de_input+self.mask_self_attention_1(en_input,en_input,de_input))
        de_input=self.layernormalization3(de_input+self.feed(de_input))    
        return de_input

#这是我自己实现的Transformer模型
class Transformer(nn.Module):
    def __init__(self,LayerNumber,HiddenSize,Heads,Mask_matrix):
        super(Transformer,self).__init__()
        
        self.LayerNumber=LayerNumber #这个参数表示的意思是一共有几层encoder和decoder
        
        self.HiddenSize=HiddenSize #Transformer里面的隐藏层是多少
        
        self.Heads=Heads #多头注意力的头数
        
        self.Mask_matrix=Mask_matrix #这个矩阵是为了在decoder当中进行mask
        
        self.Encoders=nn.ModuleList([EncoderLayer(self.HiddenSize,self.HiddenSize,self.Heads) for _ in range(self.LayerNumber)])
        self.Decoders=nn.ModuleList([DecoderLayer(self.Mask_matrix,self.HiddenSize,self.HiddenSize,self.Heads) for _ in range(self.LayerNumber)])
        
        self.device=None
    def forward(self,inputs,sr_outputs):

        #输入的inputs在哪个设备上，我们就将模型内部的局部变量加载到相应设备上
        self.device=inputs.device

        #添加位置编码
        inputs=inputs+self.Generate_positional_matrix(inputs)
        sr_outputs=sr_outputs+self.Generate_positional_matrix(sr_outputs)

        #多层EncoderLayer和Dencoderlayer的叠加
        en_vals=inputs
        de_vals=sr_outputs
        for layer in range(self.LayerNumber):
            en_vals=self.Encoders[layer](en_vals)
            de_vals=self.Decoders[layer](de_vals,en_vals)
        
        return de_vals
    
    #生成位置编码的函数
    def Generate_positional_matrix(self,x):
        _,L,C=x.shape
        positional_matrix=torch.Tensor(L,C)
        for pos in range(L):
            for i in range(int(C/2)):
                deno=np.power(10000,(2*i/C))
                positional_matrix[pos][2*i]=np.sin(pos/deno)
                positional_matrix[pos][2*i+1]=np.cos(pos/deno)
                
        return positional_matrix.to(self.device)
        #画出图像
        #cax = plt.matshow(positional_matrix)
        #plt.gcf().colorbar(cax)

def generate_mask(L):
        #生成一个上三角矩阵，但是对角线为0
        '''
            [0,1,1
             0,0,1
             0,0,0]
             1代表遮盖
             0代表不遮盖
        '''
        mask=torch.triu(torch.full((L,L),1),diagonal=1)
        return mask
    

#为了适用本次的任务设计的模型，在Transformer后面加上了softmax层
class Sen_model(nn.Module):
    def __init__(self,input_dim,hidden_dim,transformer_layer,heads,mask,classification,seq_length):
        super(Sen_model,self).__init__()
        
        self.input_dim=input_dim #输入词向量的维度
        self.hidden_dim=hidden_dim #transformer里面隐层的大小
        self.seq_length=seq_length #句子的长度
        self.layer=transformer_layer #transformer里面encoder和decoder叠加的层数
        self.heads=heads #多头注意力机制的头数
        self.mask=mask  #decoder使用的mask矩阵
        self.classification=classification #最后softmax分类的个数，这里是5


        
        #首先一个线性层，将词向量的维度映射到transformer里面隐层的大小
        self.linear1=nn.Linear(in_features=self.input_dim,out_features=self.hidden_dim)
        
        self.transformer=Transformer(LayerNumber=self.layer,HiddenSize=self.hidden_dim,Heads=self.heads,Mask_matrix=self.mask)
        #进行分类的linear
        self.linear2=nn.Linear(in_features=self.hidden_dim*self.seq_length,out_features=self.classification)

    
    def forward(self,x):
        x=self.linear1(x)
        x=self.transformer(x,x)
        x=x.view(x.shape[0],-1)
        x=self.linear2(x)
        x=F.softmax(x,dim=1)
        return x
    


'''
这个类是训练数据的准备类，用于产生训练集、测试机和验证集
这个类的实现方法在Kaggle中 ，网址为https://www.kaggle.com/code/vikkach/sentiment-analysis-lstm-pytorch 
可以去看他的解释，比较详细。我只是搬运理解的打工人。
'''
class Data_prepare:
    def __init__(self,dir_path,train_batch,test_batch):
        
        self.dir_path=dir_path #csv文件在的文件夹路径
        self.category=5 #最后分类的个数
        self.train_batch=train_batch #训练batchsize
        self.test_batch=test_batch #测试batchsize
        
        self.train_data=None #训练数据集
        self.valid_data=None #验证数据集
        self.test_data=None #测试数据集
        self.train_loader =None #训练数据集的数据加载器
        self.valid_loader =None #训练数据集的数据加载器
        self.test_loader=None #训练数据集的数据加载器
        
        
        
        self.encoded_voc=None 
        self.train_review_lens=None
        self.test_zero_idx=None
        
        
        self.Prepare()
        
        
    def Prepare(self):
        #使用pandas库的read_csv读取文件
        train_data = pd.read_csv(self.dir_path+'/train.tsv', sep = '\t')
        test_data = pd.read_csv(self.dir_path+'/test.tsv', sep = '\t')
        #去除有没有Nan的值
        test_data['Phrase'] = test_data['Phrase'].fillna(" ")
        
        
        #对输入的训练数据和测试数据集进行预处理，主要做的就是 ：转化为小写；提取单词主干和去除停用词
        train_data_pp = self.pre_process(train_data)
        test_data_pp = self.pre_process(test_data)
        print('Phrase before pre-processing: ', train_data['Phrase'][0])
        print('Phrase after pre-processing: ', train_data_pp[0])

        
        #为所有的单词确定一个唯一的序号，依据是：词语频率
        self.encoded_voc = self.encode_words(train_data_pp + test_data_pp)
        
        
        将原来的语句单词转化为数字的表示形式，依靠前面train_data_pp和test_data_pp字典
        train_reviews_ints = self.encode_data(train_data_pp)
        test_reviews_ints = self.encode_data(test_data_pp)
        print('Example of encoded train data: ', train_reviews_ints[0])
        print('Example of encoded test data: ', test_reviews_ints[0])
        

        #由1，2，3变为one-hot编码
        y_target = self.to_categorical(train_data['Sentiment'],self.category)
        print('Example of target: ', y_target[0])
        

        #这里所有句子长度的一个数量统计，返回一个字典，主要是后面为了确定最大的句子长度，用来padding
        self.train_review_lens = Counter([len(x) for x in train_reviews_ints])
        test_review_lens = Counter([len(x) for x in test_reviews_ints])


        #在这里我们记录下长度为0的句子的下标，后面在输出情感的时候直接输出：中立
        self.test_zero_idx = [test_data.iloc[ii]['PhraseId'] for ii, review in enumerate(test_reviews_ints) if len(review) == 0]
        # remove reviews with 0 length
        non_zero_idx = [ii for ii, review in enumerate(train_reviews_ints) if len(review) != 0]


        #这里将训练数据集中句子长度已经为0的句子删除
        train_reviews_ints = [train_reviews_ints[ii] for ii in non_zero_idx]
        y_target = np.array([y_target[ii] for ii in non_zero_idx])
        print('Number of reviews after removing outliers: ', len(train_reviews_ints))
        

        #根据最大的句子长度进行padding
        train_features = self.pad_features(train_reviews_ints, max(self.train_review_lens))
        X_test = self.pad_features(test_reviews_ints, max(self.train_review_lens))


        #在train训练集中划分验证集
        X_train,X_val,y_train,y_val = train_test_split(train_features,y_target,test_size = 0.2)
        print(X_train[0])
        print(y_train[0])
        print("X_train",X_train.shape)
        print("X_val",X_val.shape)
        print("X_test",X_test.shape)

        #提取出测试集的index后面测试有用
        ids_test = np.array([t['PhraseId'] for ii, t in test_data.iterrows()])
        print(ids_test)


        #下面就是pytorch经典生成数据加载器的过程
        self.train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        self.valid_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        self.test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(ids_test))
        

        self.train_loader = DataLoader(self.train_data, shuffle=True, batch_size=self.train_batch)
        self.valid_loader = DataLoader(self.valid_data, shuffle=True, batch_size=self.train_batch)
        self.test_loader = DataLoader(self.test_data, batch_size=self.test_batch)
        
        
    '''
    输入：DataFrame  ， 输出：字符串的list。
    作用：自然语言的预处理
    '''
    def pre_process(self,df):
        
        reviews = []
        
        stopwords_set = set(stopwords.words("english")) #set集合里面全部是停用词
        
        ps = PorterStemmer() 
        for p in tqdm(df['Phrase']):
            # 
            p = p.lower()
            # remove punctuation and additional empty strings
            p = ''.join([c for c in p if c not in punctuation])
            reviews_split = p.split()
            reviews_wo_stopwords = [word for word in reviews_split if not word in stopwords_set]
            reviews_stemm = [ps.stem(w) for w in reviews_wo_stopwords]
            p = ' '.join(reviews_stemm)
            reviews.append(p)
        return reviews
    
    '''
    输入：字符串的二维list。， 输出：一个字典，key是单词，value是1一依次增加。
    作用： 为每一个单词确定一个编号，频率最大的编号为1，以此类推。
    '''
    def encode_words(self,data_pp):
        words = []
        for p in data_pp:
            words.extend(p.split())
        counts = Counter(words)
        vocab = sorted(counts, key=counts.get, reverse=True)
        vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
        return vocab_to_int
    
    
    '''
    输入：字符串的二位维list， 输出：二维list，里面存储着单词对应的编号。
    作用： 将原来的语句转化维向量表示模式
    '''
    def encode_data(self,data):
        reviews_ints = []
        for ph in data:
            reviews_ints.append([self.encoded_voc[word] for word in ph.split()]) 
        return reviews_ints 
    
    
    '''
    输入:类别标签[0,1,2,3,4]，输出：一维nparray，one-hot编码。
    作用： 生成one-hot编码
    '''
    def to_categorical(self,y, num_classes):
            """ 1-hot encodes a tensor """
            return np.eye(num_classes, dtype='uint8')[y]
        

    '''
    输入:二位list，已经数字化的数据集，输出：二维nparray数组
    作用：这个就是传入最大句子长度seq_length，短的句子补齐
    ''' 
    def pad_features(self,reviews, seq_length):
            features = np.zeros((len(reviews), seq_length), dtype=int)
            for i, row in enumerate(reviews):
                try:
                    features[i, -len(row):] = np.array(row)[:seq_length]
                except ValueError:
                    continue
            return features


    def Get_train_data(self):
            return self.train_data
    def Get_test_data(self):
            return self.test_data
    def Get_vali_data(self):
            return self.valid_data
        
    def Get_train_loader(self):
            return self.train_loader
    def Get_test_loader(self):
            return self.test_loader
    def Get_vali_loader(self):
            return self.valid_loader

    #获得句子的最大长度
    def Get_max_embedding_size(self):
        return max(self.train_review_lens)
        
    #获得test集合中句子长度为0的index集合返回出来
    def Get_test_zero_index(self):
        return self.test_zero_idx
      
   
    
if __name__=='__main__':
    
    #定义一些超参数
    device =torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu') #训练设备
    train_batch=128 #训练batchsize
    test_batch=128  #测试batchsize
    mask_matrix=generate_mask(30).to(device) #生成decoder的mask矩阵
    learning_rate=1e-3 #学习率
    Epoch=2 #训练轮数
    path=os.getcwd()


    #加载数据
    print("Data preparation begin!")#
    pData=Data_prepare(path,train_batch,test_batch)
    train_loader=pData.Get_train_loader()
    test_loader=pData.Get_test_loader()
    vali_loader=pData.Get_vali_loader()
    print("Data preparation end!")


    
    print("Train model begin")
    #训练准备，定义模型
    model=Sen_model(input_dim=1,hidden_dim=256,transformer_layer=2,heads=8,mask=mask_matrix,classification=5,seq_length=30)
    #将模型记载到相应设备上
    model = model.to(device)
    #定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #定义损失函数
    criterion = nn.CrossEntropyLoss()

    #history记录test阶段的loss和accurary用于plot画图
    history = {'Test Loss':[],'Test Accuracy':[]}

    #开始多轮训练
    for epoch in range(1,Epoch+1):
        '''
            这个写法可以生成进度条，enumerrate会生成（index，value)的组合对
            trainloader对象本来就是（data，label）的组合对
        '''     
        processBar = tqdm(enumerate(train_loader), total=len(train_loader))
        #模型训练之前一定要写
        model.train()

        for index,(data,label) in processBar:
            data=data.reshape(data.shape[0],data.shape[1],1).float().to(device)
            label=label.to(device)

            #模型前向传播
            outputs=model(data)

            #argmax就是按照某一个维度求得这个维度上最大值的下标，如果不想降维，请使用keepdim=True
            prediction=torch.argmax(outputs,dim=1)

            #使用argmax将one-hot变为index类型，便于下面算正确率
            label_index=torch.argmax(label, dim=1)
            
            #sum(prediction==label_index)会生成0-1矩阵，sum求和就是统计为True的过程，再除以本次batch的数量
            acc=torch.sum(prediction==label_index)/data.shape[0]

            #计算损失
            loss=criterion(outputs,label.float())

            '''
                反向传播三件套
                    zero_grad可以将上一次计算得出的梯度清零，因为每次梯度的计算使用的是加法，如果不清0，那么后面梯度的更新就会加入前面计算出来的梯度
                    backward反向传播
                    step更新参数
            '''
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #进度条旁边打印说明
            processBar.set_description("[%d/%d] Loss : %.8f Acc：%.8f" %(epoch,Epoch,loss,acc))

        #下面的部分是在验证集上进行验证，常常结合早停机制
        model.eval()
        with torch.no_grad():
            total_loss=0
            total_right=0
            for index,(t_data,t_label) in enumerate(vali_loader):
                #以下这些和训练的时候一样，可以看上面的训练
                t_data=t_data.reshape(t_data.shape[0],t_data.shape[1],1).float().to(device)

                t_label=t_label.to(device)
  
                t_outputs=model(t_data)

                loss=criterion(t_outputs,t_label.float())
            
                t_prediction=torch.argmax(t_outputs,dim=1)
            
                t_label_index=torch.argmax(t_label,dim=1)

                #将所有的loss加起来，后面打印
                total_loss+=loss
                total_right+=torch.sum(t_prediction==t_label_index)
            
            test_data_length=len(pData.Get_test_data())
            print(test_data_length)

            
            average_loss=total_loss/test_data_length
            total_acc=total_right/test_data_length
            
            history['Test Loss'].append(average_loss.item())
            history['Test Accuracy'].append(total_acc.item())
            
            #打印这一次的vali的loss和acc
            print("Epoch:{}/{} Vali: average_loss{} Total_acc {}".format(epoch,Epoch,average_loss,total_acc))

    #下面的三行是保存模型的参数
    PATH=os.getcwd()+'/state_dict_model.pth'
    torch.save(model.state_dict(),PATH)
    print("Train model end")    



#下面的部分是用训练好的模型进行测试
@torch.no_grad()
def prediction(model,test_loader, device, batch_size,test_zero_idx):
    #创建一个空的DataFrame，
    df = pd.DataFrame(
                      {'PhraseId': pd.Series(dtype='int'),
                      'Sentiment': pd.Series(dtype='int')
                      }
                     )
    model.eval()
    for seq, id_ in tqdm(test_loader):
        #形状变为【B，L，1】 B是testbatch，L是句子长度
        seq=seq.reshape(seq.shape[0],seq.shape[1],1).float().to(device)
        #进行预测
        out= model(seq)
        #one-hot变为index
        out_index=torch.argmax(out,dim=1)

        #下面填写DataFrame
        for i, answer in zip(id_,out_index):
            #如果句子长度为0，就输出中立
            if i in test_zero_idx:
                predicted = 2
            else:
                predicted = answer.item()
            #添加一条记录
            subm = {
                     'PhraseId': int(i), 
                     'Sentiment': predicted
                   }
            df = df._append(subm, ignore_index=True)  
    return df 

#进行预测，返回一个DataFrame对象
submission=prediction(model,test_loader, device, 128, pData.Get_test_zero_index())
#将结果存在csv当中
submission.to_csv('submission.csv', index=False)
