import argparse
import torch
from torch import optim
import torch.nn as nn
from nltk import bleu_score
import math
from rouge import Rouge 

use_cuda = torch.cuda.is_available()

database = 'data_t/test2'
database = 'data_t/lr0.005'
database = 'data_t/wd0.8_lr0.005'
database = 'data_t/wd0.8_lr0.001'
database = 'data_t/test'


def save_result(str,path):
    path = database+path
    file1 = open(path,'a',encoding='utf-8')
    file1.write(str)
    file1.write('\n')
    pass

def to_cuda(tensors):
    for i, tensor in enumerate(tensors):
        tensors[i] = tensor.cuda()
    return tensors

def distinct1(output_words,types): #unigrams 一元的
    num_token = len(output_words)
    type_t = list(set(types+output_words))
    #print("tttttt",num_token)
    return num_token,type_t 
    

def distinct2(output_words,types_2):#bigrams 二元词组
    num_token = len(output_words) - 1
    types_t = []
    for j in range(len(output_words) -1):
        pair = tuple(output_words[j:j+2])
            #print(pair)
        types_t.append(pair)    

    types_2 = list(set(types_2+types_t))    

    return num_token,types_2

def evaluate_specific(model,in_seq,out_seq,in_len,person,senti,types,types_2,index2word):


    response = [index2word[j] for j in out_seq]
    
    criterion = nn.NLLLoss(ignore_index=0)

    loss_eva, output_words = model.evaluate([in_seq], [out_seq], [in_len], [person],[senti],criterion) #这里就跑一句话

    try:
        target_index = output_words[0].index('<EOS>') + 1
    except ValueError:
        target_index = len(output_words[0])

    # TODO: Remove this false target_index 所以隐去下面


    output_words = output_words[0][:target_index]

    output_sentence = ' '.join(output_words)
    response_sentence = ' '.join(response)
    #print('<', output_sentence)
    #print('>',response)

    #它这里计算是不是反了？应该是参考句在前，候选句在后吧--已经替换了
    """
    bleu1 = bleu_score.corpus_bleu([response],[output_sentence],weights=(1, 0, 0, 0))
    bleu2 = bleu_score.corpus_bleu([response],[output_sentence],weights=(0, 1, 0, 0))
    bleu3 = bleu_score.corpus_bleu([response],[output_sentence],weights=(0, 0, 1, 0))
    bleu4 = bleu_score.corpus_bleu([response],[output_sentence],weights=(0, 0, 0, 1))
    """


    bleu1 = bleu_score.sentence_bleu([response],output_words,weights=(1, 0, 0, 0))
    bleu2 = bleu_score.sentence_bleu([response],output_words,weights=(0, 1, 0, 0))
    bleu3 = bleu_score.sentence_bleu([response],output_words,weights=(0, 0, 1, 0))
    bleu4 = bleu_score.sentence_bleu([response],output_words,weights=(0, 0, 0, 1))

        
    #print('BLEU1 Score', bleu1,'BLEU4 Score', bleu4)



    num_token,types = distinct1(output_words,types)

    num_token_2,types_2 = distinct2(output_sentence,types_2)


    rouge = Rouge()
    scores = rouge.get_scores(output_sentence, response_sentence)
    rouge_1 = scores[0]['rouge-1']['f']
    rouge_2 = scores[0]['rouge-2']['f']
    rouge_l = scores[0]['rouge-l']['f']


    #help_fn.show_attention(dialogue, output_words, attentions, name=name)
    return loss_eva,bleu1,bleu2,bleu3,bleu4,num_token,types,num_token_2,types_2,rouge_1,rouge_2,rouge_l


def evaluate_all_val(model,val_in_seq,val_out_seq,val_people,val_senti,index2word,num):

    model.encoder.eval()
    model.generator.eval()
    model.lstm.eval()

    val_samples = len(val_in_seq)
    """
    if use_cuda:
        val_in_seq = to_cuda(val_in_seq)
        val_out_seq = to_cuda(val_out_seq)
        #val_speakers = val_speakers.cuda()
        val_speakers = to_cuda(val_speakers)
        #val_addressees = val_addressees.cuda()
        val_addressees = to_cuda(val_addressees)
            #计算全部val集的误差ppl以及belu和distinct
    """
    print('Evaluating Model on %d validation samples' % (val_samples))
    
    loss_eva_all = 0

    bleu1_score = 0
    bleu2_score = 0
    bleu3_score = 0
    bleu4_score = 0
    num_token_all = 0 
    num_token_2_all = 0
    types = []
    types_2 = []
    rouge_1_score = 0
    rouge_2_score = 0
    rouge_l_score = 0
    for i in range(val_samples):


        val_pair = [val_in_seq[i], val_out_seq[i], val_in_seq[i].size()[0],val_people[i],val_senti[i]]
        if use_cuda:
            val_pair[0] = val_pair[0].cuda()
            val_pair[1] = val_pair[1].cuda()

        print(i)
        loss_eva,bleu1,bleu2,bleu3,bleu4,num_token,types,num_token_2,types_2,rouge_1,rouge_2,rouge_l = evaluate_specific(model,*val_pair,types,types_2,index2word)

        #修改到这 加入了person
        bleu1_score += bleu1
        bleu2_score += bleu2
        bleu3_score += bleu3
        bleu4_score += bleu4 
        num_token_all += num_token
        num_token_2_all += num_token
        rouge_1_score += rouge_1
        rouge_2_score += rouge_2
        rouge_l_score += rouge_l
        loss_eva_all += loss_eva
    loss_eva_all = loss_eva_all/val_samples  #为了测试隐藏
    #loss_eva_all = loss_eva_all/i 
    try:
        ppl_val = math.exp(loss_eva_all)
        pass
    except Exception as e:
        ppl_val = -1
        #raise

    distinct_1 = len(types)/num_token_all
    distinct_2 = len(types_2)/num_token_2_all
    bleu1_score = bleu1_score / val_samples
    bleu2_score = bleu2_score / val_samples
    bleu3_score = bleu3_score / val_samples
    bleu4_score = bleu4_score / val_samples
    rouge_1_score = rouge_1_score / val_samples
    rouge_2_score = rouge_2_score / val_samples
    rouge_l_score = rouge_l_score / val_samples
    print('loss_val: ',loss_eva_all,'ppl_val: ',ppl_val)

    print('num: '+str(num)+'BLEU1 Score ', bleu1_score, 'BLEU2 Score ', bleu2_score, 'BLEU3 Score ', bleu3_score, 'BLEU4 Score ', bleu4_score)
    print('distinct-1 ',distinct_1,'distinct-2 ',distinct_2)
    print('rouge_1 ',rouge_1_score,'rouge_2 ',rouge_2_score,'rouge_l ',rouge_l_score)
    str_e ='num: '+str(num) +'  loss_val: '+str(loss_eva_all)+'  ppl_val: '+str(ppl_val)+'  BLEU1 Score '+ str(bleu1_score)+'  BLEU2 Score '+ str(bleu2_score)+'  BLEU3 Score '+ str(bleu3_score)+ '  BLEU4 Score '+str(bleu4_score)+'  distinct-1: '+str(distinct_1)+'  distinct-2: '+str(distinct_2)+'  rouge_1 Score: '+str(rouge_1_score)+'  rouge_2 Score: '+str(rouge_2_score)+'  rouge_l Score: '+str(rouge_l_score)
    save_result(str_e,'/train_result.txt')
    model.encoder.train()
    model.generator.train()
    model.lstm.train()
    return loss_eva_all
