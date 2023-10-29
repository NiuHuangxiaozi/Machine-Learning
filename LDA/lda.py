import numpy as np
import matplotlib.pyplot as plt
import heapq

class Fisher_Analysis:
    def __init__(self,classification=3,tar_dim=1):
        self.S_w=None
        self.B=None
        self.classification=classification
        self.tar_dim=tar_dim
        
        assert(self.tar_dim<=classification-1)
        
        self.w=[]
    def fit(self,train_data,train_label):
        '''
            train_data(type:list)
            train_label(type:list)
        '''
        #求出S_w
        for data in train_data:
            if self.S_w is None:
                self.S_w=np.cov(data,rowvar=False)
            else:
                self.S_w+=np.cov(data,rowvar=False)
           
        #求出u_i
        means=[]
        for data in train_data:
            means.append(np.mean(data,axis=0))
        total_train_data=np.stack(train_data,axis=0)
        total_mean=np.mean(total_train_data.reshape(-1,total_train_data.shape[-1]),axis=0)
        total_mean=total_mean.reshape(-1,1)

        
        #计算B
        for index in range(len(means)):
            means[index]=means[index].reshape(-1,1)
            cov_matrix=np.matmul(means[index]-total_mean,(means[index]-total_mean).T)
            if self.B is None:
                self.B=train_data[index].shape[0]*cov_matrix
            else:
                self.B+=train_data[index].shape[0]*cov_matrix
                
        self.tar_matrix=np.matmul(np.linalg.inv(self.S_w),self.B)
        
        
        eigenvalue,eigenvector=np.linalg.eig(self.tar_matrix)
        eigenvalue=list(eigenvalue)
        eigenvector=eigenvector.T
        
        
        topk_eigen=heapq.nlargest(self.tar_dim, range(len(eigenvalue)), eigenvalue.__getitem__)
        
        for index in topk_eigen:
            self.w.append(eigenvector[index])
        print("Fisher Analysis End!")
        
    #目前只支持一维
    def draw(self):
        #首先画出新的基准
        print(self.w)
        k=self.w[0][1]/self.w[0][0]
        
        xline=np.linspace(0.2,0.3,50)
        yline=[k*item for item in xline]
        plt.plot(xline,yline)
        
        
if __name__=='__main__':
    train_data_1=np.random.normal(0.28,0.01,(60,2))
    train_data_2=np.random.normal(0.24,0.01,(60,2))
    train_data_3=np.random.randn(60,2)
    train_label_1=np.full((60,1),1)
    train_label_2=np.full((60,1),2)
    train_label_3=np.full((60,1),3)
    plt.scatter(train_data_1[:,0].reshape(-1),train_data_1[:,1].reshape(-1))
    plt.scatter(train_data_2[:,0].reshape(-1),train_data_2[:,1].reshape(-1))
    plt.scatter(train_data_3[:,0].reshape(-1),train_data_3[:,1].reshape(-1))
    
    train_data=[train_data_1,train_data_2,train_data_3]
    train_label=[train_label_1,train_label_2,train_label_3]
    fisher=Fisher_Analysis(3,1)
    fisher.fit(train_data,train_label)
    fisher.draw()
    
    
    