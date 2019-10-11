import os
import random
import math

import argparse
import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


database = 'data_t/test'


use_cuda = torch.cuda.is_available()
def save_train_l(str,path):
    path = database+path
    file_l = open(path,'a',encoding='utf-8')
    file_l.write(str)
    file_l.write('\n')
    pass



class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator""" #这里仅仅是根据target来计算生成结果的loss 但是，为什么一句话只有一种结果呢？完全可以有不同的回答吧？论文里的公式就考虑了所有的分支可能性吧
    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, target, reward):
        """
        Args:  N = batch_size * seq_len
            prob: (N, C), torch Variable 
            target : (N, ), torch Variable
            reward : (N, ), torch Variable
        """

        target = torch.nn.utils.rnn.pad_sequence(target)
        #print("kkkkkk")
        #print(prob.size())
        #print(target.size())
        #print(reward.size())
        seq_len = target.size(0)
        #print(seq_len)
        target = target.view(-1)
        prob = prob.view(-1,prob.size(2))
        #reward = reward[:,:seq_len].permute(1, 0).contiguous().view(-1,1)  #改为(batch_size*seq_len，1)    这里seqlen的位置有大问题，应当是在前面 因为是seqlen,batch_size
        reward = reward[:seq_len,:].permute(1, 0).contiguous().view(-1,1)  #改为(batch_size*seq_len，1)
        #print("kkkkkk")
        #print(prob.size())
        #print(target.size())
        #print(reward.size())
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1,1)), 1) #把原始的数字组代表的单词转换成one-hot模型也就是将数字转换成5000维的向量 其中只有对应位置有一个1
        one_hot = one_hot.type(torch.ByteTensor)
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot).view(1,-1) # 这一步就是取出target对应的词的生成概率log likelihood 结果是(N,)
        #根据掩码张量mask中的二元值，取输入张量中的指定项( mask为一个 ByteTensor)，将取值返回到一个新的1D张量，
        #print(loss.min(),'概率值： ',loss.size())
        #print(reward.min(),'奖励值： ',reward.size())


        """ 存下中间结果
        str_loss = 'loss: '+str(loss.min().item())
        save_train_l(str_loss,'/train_loss.txt')
        str_reward = 'loss: '+str(reward.min().item())
        save_train_l(str_reward,'/train_loss.txt')
        """


        loss = loss.mm(reward)  #将对应的产生概率也就是G(yi) * Q（yi）
        #print(loss.max(),'loss_t: ',loss.size())
        #str_loss = 'loss: '+str(loss.max().item())
        #save_train_l(str_loss,'/train_loss.txt')  

        #loss =  -torch.sum(loss)/seq_len  #训练发现loss上升，怀疑是这里有问题
        loss =  loss/seq_len     #这里之所以不取负号是因为 这里是要最大化奖励值*概率值 但是二者都是负数，相乘之后应当是最小化 ; 不用sum因为矩阵乘法会自动加
        return loss

"""
def l2_reg(model,weight_decay):
    l2_num =torch.tensor(0.)
    for param in model.parameters():
        l2_num += torch.norm(param)
        pass

    l2_num = l2_num*weight_decay

    if use_cuda:
        l2_num = l2_num.cuda()
    return l2_num
"""

def l2_reg(model,weight_decay):
    l2_loss = None
    for W in model.parameters():
        if l2_loss is None:
            l2_loss = W.norm(2)
        else:
            l2_loss = l2_loss + W.norm(2)

    l2_loss = l2_loss*weight_decay
    if use_cuda:
        l2_loss = l2_loss.cuda()
    return l2_loss
