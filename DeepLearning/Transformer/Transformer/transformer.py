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
        
class Transformer(nn.Module):
    def __init__(self,LayerNumber,HiddenSize,Heads,Mask_matrix):
        super(Transformer,self).__init__()
        
        self.LayerNumber=LayerNumber #这个参数表示的意思是一共有几层encoder和decoder
        
        self.HiddenSize=HiddenSize #Transformer里面的隐藏层是多少
        
        self.Heads=Heads #多头注意力的头数
        
        self.Mask_matrix=Mask_matrix #这个矩阵是为了在decoder当中进行mask
        
        self.Encoders=nn.ModuleList([EncoderLayer(self.HiddenSize,self.HiddenSize,self.Heads) for _ in range(self.LayerNumber)])
        self.Decoders=nn.ModuleList([DecoderLayer(self.Mask_matrix,self.HiddenSize,self.HiddenSize,self.Heads) for _ in range(self.LayerNumber)])
        
        
    def forward(self,inputs,sr_outputs):
        
        inputs=inputs+self.Generate_positional_matrix(inputs)
        sr_outputs=sr_outputs+self.Generate_positional_matrix(sr_outputs)
        
        en_vals=inputs
        de_vals=sr_outputs
        for layer in range(self.LayerNumber):
            print(layer)
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
                
        return positional_matrix
        #画出图像
        #cax = plt.matshow(positional_matrix)
        #plt.gcf().colorbar(cax)

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
        mask=torch.triu(torch.full((L,L),1),diagonal=1)
        return mask
        
if __name__=='__main__':
    data=torch.tensor(np.random.normal(0,0.1,(32,100,512))).float()
    
    model=Transformer(LayerNumber=2,HiddenSize=512,Heads=8,Mask_matrix= generate_mask(data))
    answer=model(data,data)
    print(answer.shape)
    