import argparse
import sys
from os import path
import torch
from data_preprocess import Data_Preprocess
from sentence_embedding import Sentence_embedding
from Encoder import Encoder
from Generator import Generator
from left_model import Left_model
from Discriminator import MLP,train_dis_epoch,MLP_con
from Rollout import Rollout
from eva import evaluate_all_val,save_result
from helper import GANLoss

from torch import optim
import torch.nn as nn
import time

use_cuda = torch.cuda.is_available()
use_cuda = False

def to_cuda(tensors):
    for i, tensor in enumerate(tensors):
        tensors[i] = tensor.cuda()

    return tensors

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-pre_n", "--pre_num_iters", type=int, help="Number of iterations over the pre-training set.", default=2)
    parser.add_argument("-n", "--num_iters", type=int, help="Number of iterations over the training set.", default=20)
    parser.add_argument("-nl", "--num_layers", type=int, help="Number of layers in Encoder and Decoder", default=1)
    parser.add_argument("-z", "--hidden_size", type=int, help="GRU Hidden State Size", default=512)
    parser.add_argument("-pre_b", "--pre_batch_size", type=int, help="Pre_train Batch Size", default=128)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch Size", default=1)
    parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate of optimiser.", default=0.01)
    #预训练时采用 lr = 1会有大问题
    parser.add_argument("-con_rate", "--con_rate", type=float, help="Ratio of the gan models .", default=1.0)    
    parser.add_argument("-dr", "--dropout", type=float, help="Dropout in decoder.", default=0)# 这里就只有一层所以计划不加入dropout
    parser.add_argument("-l0", "--min_length", type=int, help="Minimum Sentence Length.", default=5) #设置句子的长度阈值
    parser.add_argument("-l1", "--max_length", type=int, help="Maximum Sentence Length.", default=50) #句子的最长的长度 设为70先
    parser.add_argument("-f", "--fold_size", type=int, help="Size of chunks into which training data must be broken.", default=500000)
    parser.add_argument("-tm", "--track_minor", type=bool, help="Track change in loss per cent of Epoch.", default=True)
    parser.add_argument("-tp", "--tracking_pair", type=bool, help="Track change in outputs over a randomly chosen sample.", default=True)
    parser.add_argument("-d", "--dataset", type=str, help="Dataset directory.", default='data/')
    #parser.add_argument("-e", "--Elmo_file", type=str, help="File containing word embeddings.", default='../Embeddings/GoogleNews/GoogleNews-vectors-negative300.bin.gz')
    num = 0   #选择模型编号
    parser.add_argument("-ep", "--encoder_parameters", type=str, help="Name of file containing encoder parameters.", default='encoder_epoch_'+str(num)+'.pt')
    parser.add_argument("-gp", "--generator_parameters", type=str, help="Name of file containing generator parameters.", default='generator_epoch_'+str(num)+'.pt')
    parser.add_argument("-lp", "--lstm_parameters", type=str, help="Name of file containing lstm parameters.", default='lstm_epoch_'+str(num)+'.pt')
    parser.add_argument("-dp", "--discriminator_parameters", type=str, help="Name of file containing discriminator parameters.", default='discriminator_epoch_'+str(num)+'.pt')
    parser.add_argument("-dcp", "--discriminator_con_parameters", type=str, help="Name of file containing discriminator_con parameters.", default='discriminator_con_epoch_'+str(num)+'.pt')

    args = parser.parse_args()

    print('Model Parameters:')
    print('Hidden Size                   :', args.hidden_size)
    print('Batch Size                    :', args.batch_size)
    print('Number of Layers              :', args.num_layers)
    print('Max. input length             :', args.max_length)
    print('Learning rate                 :', args.learning_rate)
    print('Number of Epochs              :', args.num_iters)
    print('Number of pre_Epochs          :', args.pre_num_iters)
    print('--------------------------------------------\n')

    print('Loading data...')
    
    #数据加载部分
    data = Data_Preprocess(args.dataset, min_length=args.min_length, max_length=args.max_length,img = True)
    personas = len(data.people) + 1

    print("Number of training Samples    :", len(data.x_train))
    #print("sample:",data.x_train[0])
    #print("sample:",data.train_lengths[0])
    print("Number of validation Samples  :", len(data.x_val))
    print("Number of test Samples  :", len(data.x_test))
    print("Number of Personas            :", personas)
    print("Number of words            :", len(data.word2index))
    

    #embedding = Get_Eembedding(data.word2index) #使用elmo 还未实现
    embedding = (len(data.word2index),300)
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

    discriminator = MLP(args.batch_size,args.max_length,2,len(data.word2index),300) 
    dis_criterion = nn.NLLLoss()
    dis_optimizer = optim.Adam(discriminator.parameters())

    
    discriminator_con = MLP_con(4096+args.hidden_size,2,sen_embedding,data.index2word)
    #4608 这个值是4096+512
    dis_con_criterion = nn.NLLLoss()
    dis_con_optimizer = optim.Adam(discriminator_con.parameters())

    args.encoder_parameters = path.join('data/', args.encoder_parameters)
    args.generator_parameters = path.join('data/', args.generator_parameters)
    args.lstm_parameters = path.join('data/', args.lstm_parameters)
    args.discriminator_parameters = path.join('data/', args.discriminator_parameters)
    args.discriminator_con_parameters = path.join('data/', args.discriminator_con_parameters) 

    if path.isfile(args.encoder_parameters) and path.isfile(args.generator_parameters) and path.isfile(args.lstm_parameters) and path.isfile(args.discriminator_parameters) and path.isfile(args.discriminator_con_parameters):
        load_weights(encoder, torch.load(args.encoder_parameters))
        load_weights(generator_parameters, torch.load(args.generator_parameters))
        load_weights(lstm_parameters, torch.load(args.lstm_parameters))
        load_weights(discriminator_parameters, torch.load(args.discriminator_parameters))
        load_weights(discriminator_con_parameters, torch.load(args.discriminator_con_parameters))

    else:
        print('One or more of the model parameter files are missing. Results will not be good.')


    if use_cuda:
        discriminator = discriminator.cuda()
        discriminator_con = discriminator_con.cuda()


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


    evaluate_all_val(left_model,val_in_seq,val_out_seq,val_people,val_senti,data.index2word,-2)   
 