from os import path
from io import open
import torch
import torch.nn as nn
from torch import Tensor
import sys
from sentence_embedding import Sentence_embedding
import pickle


def get_people(miss_people):
    path = 'data/user_select.txt'
    file = open(path,'r',encoding = 'utf-8')
    lines = file.readlines()
    people = []
    for person in lines :
        if person.strip('\n') not in miss_people:
            people.append(person.strip('\n'))
            pass
        #print(people)
        #break

    return people



def save(person,s_embedding_pos,s_embedding_neg):

    path1 = 'data/sen_twitter/'+person+'/pos_embedding.txt'
    file1 = open(path1,'wb') 
    num_pos = s_embedding_pos.detach().numpy() #.detach()是干嘛的？分离出来?
    pickle.dump(num_pos, file1) 
    path2 = 'data/sen_twitter/'+person+'/neg_embedding.txt'
    file2 = open(path2,'wb') 
    num_neg = s_embedding_neg.detach().numpy() #.detach()是干嘛的？
    pickle.dump(num_neg, file2) 
    file1.close()  
    file2.close()
    pass

def load(person,senti):
    path1 = 'data/sen_twitter/'+person+'/'+senti+'_embedding.txt'
    file1 = open(path1,'rb') 
    num = pickle.load(file1) 
    s_embeddings = torch.from_numpy(num)
    file1.close()  
    return s_embeddings
    pass


def extract(sen_embedding,person):
    print(person)
    sens_pos = sen_embedding.get_sen_twi(person,'pos')
    sens_neg = sen_embedding.get_sen_twi(person,'neg')
    try:
        s_embedding_pos = sen_embedding.get_senti_embedding_all(sens_pos)
        s_embedding_neg = sen_embedding.get_senti_embedding_all(sens_neg)
        save(person,s_embedding_pos,s_embedding_neg)
    except Exception as e:
        print('error: ',person)
        path1  = 'data/sen_twitter/error_1.txt'
        file1  = open(path1,'a',encoding='utf-8') 
        file1.write(person)
        file1.write('\n')
        #raise


if __name__ == "__main__":

    sen_embedding = Sentence_embedding('data/')
    miss_people = sen_embedding.miss_people
    #print(len(miss_people))
    people = get_people(miss_people)
    print(len(people))
    i = 0
    for person in people:
        print(i)
        extract(sen_embedding,person)
        #print('load',load(person,'neg').size())
        i = i +1
        #break
        pass


