# -*- coding: utf-8 -*-：

import argparse
import sys
import torch
from data_preprocess import Data_Preprocess
from sentence_embedding import Sentence_embedding
from Encoder import Encoder
from Generator import Generator
from left_model import Left_model
from eva import evaluate_all_val,save_result
from os import path

from torch import optim
import torch.nn as nn
import time

use_cuda = torch.cuda.is_available()



database = 'data_t/test2'
database = 'data_t/lr0.005'
database = 'data_t/wd0.8_lr0.005'
database = 'data_t/wd0.8_lr0.001'
database = 'data_t/test'
def to_cuda(tensors):
    for i, tensor in enumerate(tensors):
        tensors[i] = tensor.cuda()

    return tensors

def save_train_l(str,path):
    path = database+path
    file_l = open(path,'a',encoding='utf-8')
    file_l.write(str)
    file_l.write('\n')
    pass

def load_weights(model, state_dict): #加载已经训练过的模型参数
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state or own_state[name].size() != param.size():
             continue

        # Backwards compatibility for serialized parameters.
        if isinstance(param, torch.nn.Parameter): #判断二者是否是一种类型
            param = param.data

        own_state[name].copy_(param)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-pre_n", "--pre_num_iters", type=int, help="Number of iterations over the pre-training set.", default=1500)
    parser.add_argument("-n", "--num_iters", type=int, help="Number of iterations over the training set.", default=1500)
    parser.add_argument("-nl", "--num_layers", type=int, help="Number of layers in Encoder and Decoder", default=1)
    parser.add_argument("-z", "--hidden_size", type=int, help="GRU Hidden State Size", default=256)
    parser.add_argument("-pre_b", "--pre_batch_size", type=int, help="Pre_train Batch Size", default=128)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch Size", default=128)
    parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate of optimiser.", default=0.001) #
    #预训练时采用 lr = 1会有大问题
    parser.add_argument("-con_rate", "--con_rate", type=float, help="Ratio of the gan models .", default=0.2)    
    parser.add_argument("-dr", "--dropout", type=float, help="Dropout in decoder.", default=0)# 这里就只有一层所以计划不加入dropout
    parser.add_argument("-l0", "--min_length", type=int, help="Minimum Sentence Length.", default=3) #设置句子的长度阈值
    parser.add_argument("-l1", "--max_length", type=int, help="Maximum Sentence Length.", default=15) #句子的最长的长度 设为70先
    parser.add_argument("-img", "--img", type=bool, help="", default=True) 
    parser.add_argument("-f", "--fold_size", type=int, help="Size of chunks into which training data must be broken.", default=500000)
    parser.add_argument("-tm", "--track_minor", type=bool, help="Track change in loss per cent of Epoch.", default=True)
    parser.add_argument("-tp", "--tracking_pair", type=bool, help="Track change in outputs over a randomly chosen sample.", default=True)
    parser.add_argument("-d", "--dataset", type=str, help="Dataset directory.", default='/home/share/sunteng/code/data/')
    parser.add_argument("-w", "--weight_decay", type=int, help="weight_decay.", default=0.8)
    #parser.add_argument("-e", "--Elmo_file", type=str, help="File containing word embeddings.", default='../Embeddings/GoogleNews/GoogleNews-vectors-negative300.bin.gz')

    #parser.add_argument("-ep", "--encoder_parameters", type=str, help="Name of file containing encoder parameters.", default='encoder.pt')
    #parser.add_argument("-dp", "--decoder_parameters", type=str, help="Name of file containing decoder parameters.", default='decoder.pt')

 
    parser.add_argument("-ep", "--encoder_parameters", type=str, help="Name of file containing encoder parameters.", default='model/encoder_epoch.pt')
    parser.add_argument("-gp", "--generator_parameters", type=str, help="Name of file containing generator parameters.", default='model/generator_epoch.pt')
    parser.add_argument("-lp", "--lstm_parameters", type=str, help="Name of file containing lstm parameters.", default='model/lstm_epoch.pt')

    args = parser.parse_args()


    print('Model Parameters:')
    print('Hidden Size                   :', args.hidden_size)
    print('Batch Size                    :', args.batch_size)
    print('Number of Layers              :', args.num_layers)
    print('Max. input length             :', args.max_length)
    print('Learning rate                 :', args.learning_rate)
    print('Number of Epochs              :', args.num_iters)
    print('Number of pre_Epochs          :', args.pre_num_iters)
    print('Number of con_rate          :', args.con_rate)
    print('Number of weight_decay          :', args.weight_decay)
    print('dataset          :', database)

    print('--------------------------------------------\n')

    print('Loading data...')
    
    #数据加载部分
    data = Data_Preprocess(args.dataset, min_length=args.min_length, max_length=args.max_length,img = args.img)
    personas = len(data.people) + 1

    print("Number of training Samples    :", len(data.x_train))
    #print("sample:",data.x_train[0])
    #print("sample:",data.train_lengths[0])
    print("Number of validation Samples  :", len(data.x_val))
    print("Number of test Samples  :", len(data.x_test))
    print("Number of Personas            :", personas)
    print("Number of words            :", len(data.word2index)) 

    embedding = (len(data.word2index),128)
    encoder = Encoder(args.hidden_size, embedding, num_layers=args.num_layers, batch_size=args.batch_size, ELMo_embedding=False, train_embedding=True)

    generator = Generator(args.hidden_size, embedding, num_layers=args.num_layers,ELMo_embedding=False, train_embedding=True, dropout_p=args.dropout)

    sen_embedding = Sentence_embedding(args.dataset)

    if use_cuda:
        encoder = encoder.cuda()
        generator = generator.cuda()
        #sen_embedding = sen_embedding.cuda()
        pass
    """
    sens = sen_embedding.get_sen_twi('01USA18','pos')
    #print(sens) #打印句子
    #print(sen_embedding.miss_people[0]) #打印无情感向量名单
    s_embedding = sen_embedding.get_senti_embedding(sens)
    print(s_embedding.size())
    """

    left_model = Left_model(args.hidden_size,encoder,generator,sen_embedding,data.index2word,data.index2people,data.p2imgvec,num_layers=args.num_layers,max_length=args.max_length)

    encoder_trainable_parameters = list(filter(lambda p: p.requires_grad, left_model.encoder.parameters()))
    decoder_trainable_parameters = list(filter(lambda p: p.requires_grad, left_model.generator.parameters()))

    #encoder_optimizer = optim.Adam(encoder_trainable_parameters, lr=args.learning_rate,weight_decay=args.weight_decay)  #weight_decay=1e-5  加入正则化项 这里好像有问题，这里自己写的正则化
    #decoder_optimizer = optim.Adam(decoder_trainable_parameters, lr=args.learning_rate,weight_decay=args.weight_decay)  #weight_decay=1e-5  加入正则化项

    #encoder_optimizer = optim.SGD(encoder_trainable_parameters, lr=args.learning_rate,momentum=0.9)
    #decoder_optimizer = optim.SGD(decoder_trainable_parameters, lr=args.learning_rate,momentum=0.9)

    encoder_optimizer = optim.Adam(encoder_trainable_parameters, lr=args.learning_rate)  #weight_decay=1e-5  加入正则化项
    decoder_optimizer = optim.Adam(decoder_trainable_parameters, lr=args.learning_rate)  #weight_decay=1e-5  加入正则化项

    left_criterion = nn.NLLLoss(ignore_index=0)  




    encoder_parameters = path.join(database, args.encoder_parameters)
    generator_parameters = path.join(database, args.generator_parameters)
    lstm_parameters = path.join(database, args.lstm_parameters)

    print(encoder_parameters)
    if path.isfile(encoder_parameters) and path.isfile(generator_parameters) and path.isfile(lstm_parameters) :
        load_weights(encoder, torch.load(encoder_parameters))
        load_weights(generator, torch.load(generator_parameters))
        load_weights(left_model.lstm, torch.load(lstm_parameters))
        print('Successfully loaded model')
    else:
        print('One or more of the model parameter files are missing. Results will not be good.')


    train_in_seq = data.x_train#[:args.batch_size*3]
    train_out_seq = data.y_train#[:args.batch_size*3]
    train_lengths = data.train_lengths#[:args.batch_size*3]
    train_people = data.train_addressees#[:args.batch_size*3]
    train_senti = data.train_senti#[:args.batch_size*3]

    #"""
    val_in_seq = data.x_val#[:args.batch_size*3]
    val_out_seq = data.y_val#[:args.batch_size*3]
    val_lengths = data.val_lengths#[:args.batch_size*3]
    val_people = data.val_addressees#[:args.batch_size*3]
    val_senti = data.val_senti#[:args.batch_size*3]

    test_in_seq = data.x_test#[:args.batch_size*3]
    test_out_seq = data.y_test#[:args.batch_size*3]
    test_lengths = data.test_lengths#[:args.batch_size*3]
    test_people = data.test_addressees#[:args.batch_size*3]
    test_senti = data.test_senti#[:args.batch_size*3]
    #"""

    """ #替换val和test集合
    test_in_seq = data.x_val#[:args.batch_size*3]
    test_out_seq = data.y_val#[:args.batch_size*3]
    test_lengths = data.val_lengths#[:args.batch_size*3]
    test_people = data.val_addressees#[:args.batch_size*3]
    test_senti = data.val_senti#[:args.batch_size*3]

    val_in_seq = data.x_test#[:args.batch_size*3]
    val_out_seq = data.y_test#[:args.batch_size*3]
    val_lengths = data.test_lengths#[:args.batch_size*3]
    val_people = data.test_addressees#[:args.batch_size*3]
    val_senti = data.test_senti#[:args.batch_size*3]
    """


    """
    #这里我没取全部数据 实际操作记得修改
    train_in_seq = data.x_train[:args.batch_size*3]
    train_out_seq = data.y_train[:args.batch_size*3]
    train_lengths = data.train_lengths[:args.batch_size*3]
    train_people = data.train_addressees[:args.batch_size*3]
    train_senti = data.train_senti[:args.batch_size*3]

    val_in_seq = data.x_val[:args.batch_size*3]
    val_out_seq = data.y_val[:args.batch_size*3]
    val_lengths = data.val_lengths[:args.batch_size*3]
    val_people = data.val_addressees[:args.batch_size*3]
    val_senti = data.val_senti[:args.batch_size*3]

    test_in_seq = data.x_test[:args.batch_size*3]
    test_out_seq = data.y_test[:args.batch_size*3]
    test_lengths = data.test_lengths[:args.batch_size*3]
    test_people = data.test_addressees[:args.batch_size*3]
    test_senti = data.test_senti[:args.batch_size*3]
    """

    #path1 = 'data/sen_samples.txt'  #重置数据
    
    for epoch in range(args.pre_num_iters):
        path1 = database+'/sen_samples.txt'  #重置数据
        file1 = open(path1,'w') 
        file1.close()   

        #path2 = 'data/sen_samples_con.txt'
        path2 = database+'/sen_samples_con.txt'
        file2 = open(path2,'w') 
        file2.close() 
        if use_cuda:
            #print('sssssssssss')
            train_in_seq = to_cuda(train_in_seq)
            train_out_seq = to_cuda(train_out_seq)
            #lengths.cuda()
        

        fold_size = len(train_in_seq)
        fraction = fold_size // 10   #取整除
        loss_total = 0
        start1=time.time()
        for i in range(0, fold_size, args.pre_batch_size):
            start=time.time()
            print(i)
            input_variables = train_in_seq[i : i + args.pre_batch_size] # Batch Size x Sequence Length
            target_variables = train_out_seq[i : i + args.pre_batch_size]
            lengths = train_lengths[i : i + args.pre_batch_size]
            people = train_people[i : i + args.pre_batch_size]
            senti = train_senti[i : i + args.pre_batch_size]

            t1 = time.time()#记录运行时间
            #print('p1','t%s'%(t1-start))
            loss = left_model.train(input_variables,target_variables,lengths,left_criterion,encoder_optimizer,decoder_optimizer,people,senti,args.weight_decay)

            t2 = time.time()#记录运行时间
            #print('p2','t%s'%(t2-t1))

            loss_total += loss
            end=time.time()
            print('Running time: %s Seconds'%(end-start))

        end1=time.time()
        print('R time: %s Seconds'%(end1-start1))
        print('epoch======',epoch,'lr========',args.learning_rate,'train_loss:===========',loss_total)
        train_loss = str(epoch)+' loss: '+str(loss_total)
        save_train_l(train_loss,'/train_loss.txt')



        evaluate_all_val(left_model,val_in_seq,val_out_seq,val_people,val_senti,data.index2word,epoch)   


    print('test_result: ')
    evaluate_all_val(left_model,test_in_seq,test_out_seq,test_people,test_senti,data.index2word,-1)
