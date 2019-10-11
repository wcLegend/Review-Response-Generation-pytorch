from os import path
from io import open
import re
import os
import torch

import numpy as np

class Data_Preprocess(object):
    #这里设置了长度过滤 在5-20长度间的才被留下  max_length现为35
    def __init__(self, dir_path, min_length=5, max_length=20,img=False):
        self.dir_path = dir_path
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.min_length = min_length
        self.max_length = max_length
        self.vocab = set(["<PAD>", "<SOS>", "<EOS>"])
        self.word2index = {"<PAD>" : 0, "<SOS>" : 1, "<EOS>" : 2}
        self.index2word = ["<PAD>", "<SOS>", "<EOS>"]
        self.vocab_size = 3
        self.x_train = list()
        self.y_train = list()
        self.x_val = list()
        self.y_val = list()
        self.x_test = list()
        self.y_test = list()

        self.train_speakers = list()
        self.train_addressees = list()
        self.val_speakers = list()
        self.val_addressees = list()
        self.test_speakers = list()
        self.test_addressees = list()
        self.people = set()
        self.index2people = []

        self.train_senti = list()
        self.train_senti_num = list()
        self.val_senti = list()
        self.val_senti_num = list()
        self.test_senti = list()
        self.test_senti_num = list()

        self.train_lengths = []
        self.val_lengths = []
        self.test_lengths = []

        self.img = img
        self.p2imgvec = {}
        self.run()

    def load_vocabulary(self):
        with open(path.join(self.dir_path, 'vocabulary.txt'), encoding='utf-8') as f:
            for word in f:
                word = word.strip('\n')
                self.vocab.add(word)
                self.index2word.append(word)
                self.word2index[word] = self.vocab_size
                self.vocab_size += 1

    def load_dialogues(self):
        # Load train val and test set
        train_path = path.join(self.dir_path, 'train.txt')
        val_path = path.join(self.dir_path, 'val.txt')
        test_path = path.join(self.dir_path, 'test.txt')

        train_seq = [[], []] #训练用
        val_seq = [[], []]   #测试用
        test_seq = [[],[]]
        lengths = []

        #这里包含了三块信息，具体推特，回复者，原推人
        train_info = [train_seq, self.train_speakers, self.train_addressees,self.train_senti,self.train_senti_num]
        val_info = [val_seq, self.val_speakers, self.val_addressees,self.val_senti,self.val_senti_num]
        test_info = [test_seq,self.test_speakers,self.test_addressees,self.test_senti,self.test_senti_num]

        # Iterate over dialogues of both training and test datasets
        for datafile, datalist in zip([train_path, val_path,test_path], [train_info, val_info,test_info]):
            with open(datafile) as file:
                lines = file.readlines()

                for line in lines:
                    line = line.split('|')

                    input_1 = line[0].split()
                    speaker = input_1[0]
                    dialogue = input_1[1:]

                    input_2 = line[1].split()
                    addressee = input_2[0]
                    response = input_2[1:]

                    sen = line[2]
                    sen_val = line[3].strip('\n')

                    len_x = len(dialogue)
                    len_y = len(response)

                    if len_x <= self.min_length or len_x >= self.max_length or \
                       len_y <= self.min_length or len_y >= self.max_length:
                       continue

                    ''' No concept of Person ID during Pre-Training '''
                    #因为这里不考虑说话的人 所以应当将训练集的首位数字跳过，这里可能是因为未给出的原数据集是没有的所以未跳过
                    datalist[0][0].append([self.SOS_token] + [int(word)  for word in dialogue])
                    datalist[0][1].append([int(word)  for word in response] )
                    datalist[1].append(int(speaker))
                    datalist[2].append(int(addressee))
                    datalist[3].append(sen)
                    datalist[4].append(float(sen_val))

        return train_seq, val_seq,test_seq

    def get_people(self): #这里获取人物列表最好是直接从cast里获取而不是直接取 否则person_embedding会报错

        """
        people = self.train_speakers + self.val_speakers +self.test_speakers+\
                 self.train_addressees + self.val_addressees+self.test_addressees
        print(len(self.train_speakers))

        for person in people:
            #print(person)
            if person not in self.people:
                self.people.add(person)
        """
        people_path = path.join(self.dir_path, 'people.txt')
        file = open(people_path,'r',encoding = 'utf-8')
        lines = file.readlines()
        for people in lines:
            if people not in self.people:
                self.index2people.append(people.strip('\n'))
                self.people.add(people.strip('\n'))
                pass

        file.close()

    def convert_to_tensor(self, pairs):
        tensor_pairs = [[], []]
        lengths = []
        speakers = []
        addressees = []

        for i, tup in enumerate(pairs):
            tensor_pairs[0].append(torch.LongTensor(tup[0]))
            tensor_pairs[1].append(torch.LongTensor(tup[1]))
            lengths.append(len(tensor_pairs[0][-1]))


            speakers.append(tup[2])
            addressees.append(tup[3])

        speakers = torch.LongTensor(speakers)
        addressees = torch.LongTensor(addressees)

        return tensor_pairs[0], tensor_pairs[1], lengths, speakers, addressees

    def sort_and_tensor(self):
        #将对话作为整体以x的长度从大到小排序
        xysa_train = sorted(zip(self.x_train, self.y_train, self.train_speakers, self.train_addressees), key=lambda tup: len(tup[0]), reverse=True)
        xysa_val = sorted(zip(self.x_val, self.y_val, self.val_speakers, self.val_addressees), key=lambda tup: len(tup[0]), reverse=True)
        xysa_test = sorted(zip(self.x_test, self.y_test, self.test_speakers, self.test_addressees), key=lambda tup: len(tup[0]), reverse=True)


        self.x_train, self.y_train, self.train_lengths, self.train_speakers, self.train_addressees = self.convert_to_tensor(xysa_train)
        self.x_val, self.y_val, self.val_lengths, self.val_speakers, self.val_addressees = self.convert_to_tensor(xysa_val)
        self.x_test, self.y_test, self.test_lengths, self.test_speakers, self.test_addressees = self.convert_to_tensor(xysa_test)


    def get_imgvec(self):
        people = self.people  
        vec_size = 512
        features_dir = self.dir_path+'features/'
        miss_vec = np.zeros((vec_size))
        #print(miss_vec)
        k = 0
        for person in people:       
            x_path = os.path.join(features_dir, person + '.txt')
            if not os.path.exists(x_path):
                self.p2imgvec[person] = miss_vec
                #self.p2imgvec[person] = torch.from_numpy(miss_vec)
                continue    
            try:
                y = np.loadtxt(x_path, delimiter=',')
                self.p2imgvec[person] = y
                k = k+1
                #print(k)
            except Exception as e:  #一部分没有东西的............
                self.p2imgvec[person] = miss_vec
                continue   

        print(k)
            #self.p2imgvec[person] = torch.from_numpy(y)
            #print(person)  


    def run(self):
        print('Loading vocabulary.')
        self.load_vocabulary()

        #print('Loading data.')
        train_seq, val_seq,test_seq = self.load_dialogues()

        # Split to separate lists.
        self.x_train = train_seq[0]
        self.y_train = train_seq[1]
        self.x_val = val_seq[0]
        self.y_val = val_seq[1]
        self.x_test = test_seq[0]
        self.y_test = test_seq[1]

        self.get_people()
        self.sort_and_tensor()  #对所有的句子按张量长度排序
        if self.img:
            self.get_imgvec()
