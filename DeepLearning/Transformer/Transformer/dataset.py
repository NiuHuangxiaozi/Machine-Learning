class Data_prepare:
    def __init__(self,dir_path,train_batch,test_batch):
        
        self.dir_path=dir_path
        self.category=5
        self.train_batch=train_batch
        self.test_batch=test_batch
        
        self.train_data=None
        self.valid_data=None
        self.test_data=None
        self.train_loader =None
        self.valid_loader =None
        self.test_loader=None
        
        
        
        self.encoded_voc=None
        self.Prepare()
        
        
    def Prepare(self):
        
        train_data = pd.read_csv(self.dir_path+'/train.tsv', sep = '\t')
        test_data = pd.read_csv(self.dir_path+'/test.tsv', sep = '\t')
        
        test_data['Phrase'] = test_data['Phrase'].fillna(" ")
        
        #assert(test_data.isnull().sum()==0)
        
        train_data_pp = self.pre_process(train_data)
        test_data_pp = self.pre_process(test_data)
        
        print('Phrase before pre-processing: ', train_data['Phrase'][0])
        print('Phrase after pre-processing: ', train_data_pp[0])
        
        
        self.encoded_voc = self.encode_words(train_data_pp + test_data_pp)
        
        train_reviews_ints = self.encode_data(train_data_pp)
        test_reviews_ints = self.encode_data(test_data_pp)
        print('Example of encoded train data: ', train_reviews_ints[0])
        print('Example of encoded test data: ', test_reviews_ints[0])
        
        
        
        y_target = self.to_categorical(train_data['Sentiment'],self.category)
        print('Example of target: ', y_target[0])
        
        train_review_lens = Counter([len(x) for x in train_reviews_ints])
        test_review_lens = Counter([len(x) for x in test_reviews_ints])
        
        # remove reviews with 0 length
        non_zero_idx = [ii for ii, review in enumerate(train_reviews_ints) if len(review) != 0]

        train_reviews_ints = [train_reviews_ints[ii] for ii in non_zero_idx]
        y_target = np.array([y_target[ii] for ii in non_zero_idx])

        print('Number of reviews after removing outliers: ', len(train_reviews_ints))
        
        
        train_features = self.pad_features(train_reviews_ints, max(train_review_lens))
        X_test = self.pad_features(test_reviews_ints, max(train_review_lens))

        X_train,X_val,y_train,y_val = train_test_split(train_features,y_target,test_size = 0.2)
        print(X_train[0])
        print(y_train[0])

        print("X_train",X_train.shape)
        print("X_val",X_val.shape)
        print("X_test",X_test.shape)
        
        ids_test = np.array([t['PhraseId'] for ii, t in test_data.iterrows()])
        print(ids_test)
        
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
        
if __name__=='__main__':
    
    print("Test for Data preparation!")
    path=os.getcwd()
    train_batch=128
    test_batch=4
    pData=Data_prepare(path,train_batch,test_batch)
    train_loader=pData.Get_train_loader()
    for index,(data,label) in enumerate(train_loader):
        print("index: ", index)
        print("data's shape: ", data.shape)
        print("label's shape: ", label.shape)
        break