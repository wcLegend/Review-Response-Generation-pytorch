from os import path
from io import open
import torch
import torch.nn as nn
from torch import Tensor
import sys
sys.path.append('InferSent/encoder/')
from demo import Demo
import pickle
import numpy as np

#senti_emb_size = 4096
def save(screen_name):
    path1 = 'error_people_2.txt'
    file=open(path1,'a',encoding='utf-8')  # a 是追加模式
    file.write(screen_name)
    file.write('\n')
    pass


def init_s_embedding(senti_emb_size):

    h = torch.randn((30, senti_emb_size)).type(torch.FloatTensor)
    return h

class Sentence_embedding(object):
    def __init__(self,base_path):
        self.base_path = base_path
        self.miss_people = self.miss_people_get() #感情向量缺失的人物名单
        self.demo = Demo()
        #self.lin = nn.Linear(4096,512) #将结果降维  这里需要改为Lstm降维,将这步操作移到外面去做 

    def get_sen_twi(self,screen_name,senti): #获得其30条情感推特

        b = 'sen_twitter/'+screen_name+'/'+screen_name+'_t.txt'
        p_path = path.join(self.base_path, b)
        file =  open(p_path,'r',encoding='utf-8')
        lines = file.readlines()
        if senti == 'pos':
            sentences = [line.strip('\n').replace('.',' .').replace(',',' ,').replace('?',' ?').replace('!',' !').replace('、',' 、') for line in lines[:30]]
            pass

        if senti == 'neg':
            sentences = [line.strip('\n').replace('.',' .').replace(',',' ,').replace('?',' ?').replace('!',' !').replace('、',' 、') for line in lines[30:]]

        return sentences

    def miss_people_get(self):
        miss_people = []
        b = 'sen_twitter/error_people.txt'
        p_path = path.join(self.base_path,b)
        file =  open(p_path,'r',encoding='utf-8')
        lines = file.readlines()
        for line in lines:
            miss_people.append(line.strip('\n'))
        return miss_people
        pass

    def get_senti_embedding(self,sentences):
        #print(len(sentences))
        #print(sentences[:5])
        embeddings = self.demo.get_embedding(sentences)
        #embeddings = torch.from_numpy(embeddings)
        """ #合成一条
        embedding_add = embeddings[0]
        #所有元素相加取平均  这里似乎也可以采用 lin函数 但是参数或许过多 未采用
        for i in range(1,30):
            embedding_add  += embeddings[i]
            #print(embeddings[i])
            pass
        embedding_add = torch.div(embedding_add,30)
        #embedding_add = self.lin(embedding_add)  
        """
        return embeddings

    def get_senti_embedding_all(self,sentences):
        #print(sentences)
        #print(sentences[:5])
        embeddings = self.demo.get_embedding(sentences)
        #embeddings = torch.from_numpy(embeddings)

        return embeddings

    def get_emb_num(self,sentences):
        #print(sentences)
        """
        sens = []
        for sen in sentences:
            str_s = ''
            for w in sen:
                str_s = str_s+w+' '
                pass
            sens.append(str_s)
        print(sens)
        """
        #print(len(sentences))
        sens_emb = self.get_senti_embedding_all(sentences)
        #print(sens_emb)
        return sens_emb

    #""" #Infersent的embedding
    def get_sen_emb(self,screen_name,senti): #用以获取embedding


        path1 = 'data/sen_twitter/'+screen_name+'/'+senti+'_embedding.txt'
        try:
            file1 = open(path1,'rb') 
            num = pickle.load(file1) 
            s_embedding = torch.from_numpy(num)
            file1.close()  
            return s_embedding
            pass
        except Exception as e: #如果没得到情感向量 则初始化一个
            #print('error',screen_name)
            save(screen_name)
            h = init_s_embedding(4096)
            return h
            raise
    #""" 
    """ bert的embedding预处理
    def get_sen_emb(self,screen_name,senti): #用以获取embedding


        path1 = 'data/sen_twitter_bert/'+screen_name+'/'+senti+'.npy'
        try:
            file1 = open(path1,'rb') 
            num = np.load(file1) 
            s_embedding = torch.from_numpy(num)
            file1.close()  
            return s_embedding
            pass
        except Exception as e: #如果没得到情感向量 则初始化一个
            #print('error',screen_name)
            save(screen_name)
            h = init_s_embedding(1024)
            return h
            raise  
    """



        


