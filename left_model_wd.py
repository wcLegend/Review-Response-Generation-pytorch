# -*- coding:utf-8 -*-
#无对抗训练过程

import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import time 
from helper import l2_reg

database = 'data_small'
#database = 'data'
database = 'data_img'
#database = 'data_t/256'
#database = 'data_t/128'
#database = 'data_t/64'
database = 'data_t/128SGD'
database = 'data_t/r1.5'
database = 'data_t/r0.5'
database = 'data_t/noadv'
database = 'data_t/r0.2'
database = 'data_t/r0.4'
database = 'data_t/r0.6'
database = 'data_t/max_len_35'
database = 'data_t/512'
database = 'data_t/35_512'
database = 'data_t/35_128_0.2'
database = 'data_t/test2'
database = 'data_t/35_256'
database = 'data_t/35_128_real'
database = 'data_t/35_128_0.6'
database = 'data_t/35_128_0.2'
database = 'data_t/35_128_0.4'
database = 'data_t/35_128_0.15'
database = 'data_t/35_128_0.2_len20'
database = 'data_t/test2'
database = 'data_t/35_128_0.2_len15_fix'
database = 'data_t/35_128_0.2_len15'
database = 'data_t/35_128_0.2_len15_transform'
database = 'data_t/35_128_0.2_len15_bert'
database = 'data_t/35_128_0.2_len15_wb1e0'
database = 'data_t/35_128_0.2_len15_wb1e1'
database = 'data_t/35_128_0.2_len15_wb1e2'
database = 'data_t/35_128_0.2_len15_wb1e2_lr0.01'
database = 'data_t/35_128_0.2_len15_wb1e2_lr0.0001'
database = 'data_t/35_128_0.2_len15_wb1e3_lr0.0001'
database = 'data_t/35_128_0.2_len15_wb1e3_lr0.001'
database = 'data_t/35_128_0.2_len15_wb1e2_lr0.001'
database = 'data_t/35_128_0.2_len15_wb1e3_lr0.01'
database = 'data_t/35_128_0.2_len15_wb1e3_lr0.0005'
database = 'data_t/35_128_0.2_len15_wb1e2_lr0.0005'
database = 'data_t/test'
database = 'data_t/test2'

class Left_model(object):
    """docstring for Left_model"""
    def __init__(self, hidden_size,encoder, generator,sentence_embedding, index2word,index2people,p2imgvec,num_layers=1, teacher_forcing_ratio=0.4,max_length = 70):

        self.hidden_size = hidden_size
        self.encoder = encoder
        self.generator = generator
        self.sentence_embedding = sentence_embedding
        self.index2word = index2word
        self.index2people = index2people
        self.p2imgvec = p2imgvec
        self.num_layers = num_layers
        self.SOS_token = 1
        self.EOS_token = 2
        self.use_cuda = torch.cuda.is_available()
        #self.use_cuda = False
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.senti_emb_size = 4096
        #self.senti_emb_size = 1024   #bert的size
        self.lstm = nn.LSTM(self.senti_emb_size, hidden_size) #可修改
        self.max_length = max_length

    def init_hidden(self, batch_size):
        h = Variable(torch.zeros((1, batch_size, self.hidden_size)))
        c = Variable(torch.zeros((1, batch_size, self.hidden_size)))
        if self.use_cuda:
            h, c = h.cuda(), c.cuda()
        return h, c



    def save_samples(self,decoded_wids,stype):  #type 0 : real  1: fake
        #print('save_samples')
        #path1 = 'data/sen_samples.txt'
        path1 = database+'/sen_samples.txt' #测试小数据集
        file1 = open(path1,'a') 

        for gen_sen in decoded_wids:
            gen_str = ''
            seq_len_fix = self.max_length - len(gen_sen)
            for word in gen_sen:
                gen_str += str(word.item())+' '
                pass
            for i in range(seq_len_fix):
                gen_str += '0 '
                pass
            gen_str += '|'+str(stype)  
            file1.write(gen_str)
            file1.write('\n')
            pass
        file1.close()
        pass

    def save_samples_con(self,decoded_words,h,stype):
        """该方法用于保存conditional gan 训练用到的数据
        decoded_words： 真实的句子/生成的句子 (batch_size,seq_len)
        h： 情感向量  (1,batch_size,hidden_size)
        """
        #path1 = 'data/sen_samples_con.txt'
        path1 = database+'/sen_samples_con.txt'
        file1 = open(path1,'a') 
        batch_size = len(decoded_words)

        #out_emb = self.sentence_embedding.get_emb_num(decoded_words).view(1,batch_size,-1)
        #x = torch.cat((out_emb,h),2).squeeze().detach().numpy()
        #print('save_samples_con')
        h = h.squeeze().detach()
        for i in range(batch_size):
            a = " ".join(str(z.item()) for z in h[i])
            data_c = decoded_words[i]
            if stype == 0:
                data_c = data_c.cpu().numpy()
                pass
            data_c = " ".join(str(z.item()) for z in data_c)
            #print('aaaaaaaaa',data_c)
            data_c += '|'+a+'|'+str(stype)
            file1.write(data_c)
            file1.write('\n')
            pass

        file1.close()
        pass


    def forward(self, input_variables, target_variables, lengths,people,senti):

        
        

        input_variables = torch.nn.utils.rnn.pad_sequence(input_variables)
        #print("size ",input_variables.size())
        target_variables = torch.nn.utils.rnn.pad_sequence(target_variables)

        real_tar = target_variables.permute(1, 0)

        input_length = input_variables.size()[0]
        target_length = target_variables.size()[0]
        batch_size = input_variables.size()[1]

        encoder_hidden = self.encoder.init_hidden(batch_size) #初始化隐层 h_0 (num_layers * num_directions, batch, hidden_size)
        encoder_outputs, encoder_hidden = self.encoder(input_variables, lengths, encoder_hidden)
        sel = [] # s_embedding_list   
        imgvecs = []
        for i in range(batch_size): 
            person = self.index2people[people[i].item()]
            s_embeddings = self.sentence_embedding.get_sen_emb(person,senti[i])[:30]
            sel.append(s_embeddings) 
            imgvec_temp = torch.from_numpy(self.p2imgvec[person]).type(torch.FloatTensor)
            imgvecs.append(imgvec_temp)
            pass

        s_embeddings = torch.stack([s_e for s_e in sel],0).view(30,batch_size,-1) #
        imgvecs = torch.stack([ivec for ivec in imgvecs],0).view(1,batch_size,-1)

        h0, c0 = self.init_hidden(batch_size)
        
        if self.use_cuda:
            s_embeddings=s_embeddings.cuda()
            self.lstm = self.lstm.cuda()
            imgvecs = imgvecs.cuda()
            pass

        s_embeddings, (h, c) = self.lstm(s_embeddings, (h0, c0)) 

        decoder_hidden = torch.cat((encoder_hidden,h),2) #这里不能直接拼接，需要先调整形状  现在 个人情感向量的影响并不明显 基本上一个问题的答案是一致的 考虑后续要不要处理
        decoder_hidden = torch.cat((decoder_hidden,imgvecs),2) #这里加入头像向量
        #print(encoder_outputs.size())
        #print(s_embeddings.size())
        encoder_semb_outputs = torch.cat((encoder_outputs,s_embeddings),0)#这里将sentence embedding的输出和encoder的输出级联作为attn层的输入 
        #这里或许还要加入一个linear层 后续去做
        decoder_inputs = torch.LongTensor([[self.SOS_token]*batch_size])
        #目前没用cuda
        decoder_inputs = decoder_inputs.cuda() if self.use_cuda else decoder_inputs  

        decoder_hidden = decoder_hidden.view(self.num_layers, batch_size, -1) 

    
        decoded_out = []   #存结果序列
        decoded_words = [[] for i in range(batch_size)] #存结果句子  
        decoded_wid = [[] for i in range(batch_size)]   #存结果序列

        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            """
            print('tar',di)
            print('1',decoder_inputs)
            print('2',decoder_hidden)
            print('3',encoder_semb_outputs)
            """
            #print(di)
            decoder_outputs, decoder_hidden = self.generator(decoder_inputs,decoder_hidden, encoder_semb_outputs)

            topv, topi = decoder_outputs.data.topk(1)

            for i, ind in enumerate(topi):
                decoded_words[i].append(self.index2word[ind])
                decoded_wid[i].append(ind)

            #print('ssss',decoder_outputs.size())
            decoded_out.append(decoder_outputs)

            decoder_inputs = topi.permute(1, 0).detach()  #permute 维度换位 seqlen,batch_size,xxx
        #print('save')
        


        g = random.uniform(0, 10)
        if g>6:

            self.save_samples(real_tar,0)
            self.save_samples(decoded_wid,1) #存放生成结果


            self.save_samples_con(real_tar,h,0)
            #t6 = time.time()#记录运行时间
            #print('p6','t%s'%(t6-t5))
            self.save_samples_con(decoded_wid,h,1)
            #end = time.time()#记录运行时间
            #print('all','t%s'%(start-end))
            #print(decoded_words)
            pass
        #print(len(decoded_wid))
        #print(len(decoded_wid[0]))

        #print(h.size())

        #这里将结果也通过sentence embedding的原因是想通过计算两个向量的余弦相似度来充当contional gan的部分  
        #这部分删去，采用直接级联计算con gan
        """
        out_emb = self.sentence_embedding.get_emb_num(decoded_words).view(1,batch_size,-1)
        o_s, (o_h, o_c) = self.lstm(out_emb, (h0, c0)) 
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        sim = cos(h,o_h)
        sim = torch.log(sim)
        print(sim)
        """
        prob = torch.stack([out for out in decoded_out],0).view(len(decoded_out),batch_size,-1)

        return prob,h #加入h是为了con gan的计算
        

    def train(self, input_variables, target_variables, lengths, criterion,
              encoder_optimizer, decoder_optimizer, people,senti,weight_decay):
        #people [id1,id2] 原推主与回复者id

        #start=time.time()
        
        #print(real_tar)

        input_variables = torch.nn.utils.rnn.pad_sequence(input_variables)
        #print(input_variables)
        #print("size ",input_variables.size())
        target_variables = torch.nn.utils.rnn.pad_sequence(target_variables)
        #print("target_variables ",target_variables)
        #这一步将真实的结果写入文件
        #real_data_1.append((target_variables.permute(1, 0),0))
        real_tar = target_variables.permute(1, 0)
        #self.save_samples(target_variables.permute(1, 0),0)

        #t1 = time.time()#记录运行时间
        #print('p1','t%s'%(t1-start))


        input_length = input_variables.size()[0]
        target_length = target_variables.size()[0]
        batch_size = target_variables.size()[1]

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss = 0


        encoder_hidden = self.encoder.init_hidden(batch_size) #初始化隐层 h_0 (num_layers * num_directions, batch, hidden_size)

        #print(input_variables.size())
        #print(self.encoder.batch_size)
        #print(encoder_hidden.type())
        encoder_outputs, encoder_hidden = self.encoder(input_variables, lengths, encoder_hidden)

        #print('people: ',people.size(),'  ',people[0].item())
        #print('senti: ',len(senti),'  ',senti[0])
        #print(self.index2people[people[0].item()])
        #t2 = time.time()#记录运行时间
        #print('p2','t%s'%(t2-t1))

        sel = [] # s_embedding_list   
        imgvecs = []
        for i in range(batch_size): #这里一个一个计算是不是太慢了？整合一起送进去是不是好一些
            #print('people: ',people[i])
            person = self.index2people[people[i].item()]
            #print(person)
            """ 
            sens = self.sentence_embedding.get_sen_twi(person,senti[i]) #这里pos neg需要 读取 需要重新预处理数据
            #print(sens) #打印句子
            #print(sen_embedding.miss_people[0]) #打印无情感向量名单

            s_embeddings = self.sentence_embedding.get_senti_embedding(sens)
            """

            #待全部预处理结束就采用这种方法 一次取出30条 本来应当就是30条，莫名有些会超过
            s_embeddings = self.sentence_embedding.get_sen_emb(person,senti[i])[:30]

            sel.append(s_embeddings) #这里会一次取30条

            imgvec_temp = torch.from_numpy(self.p2imgvec[person]).type(torch.FloatTensor)
            imgvecs.append(imgvec_temp)
            pass

        try:
            s_embeddings = torch.stack([s_e for s_e in sel],0).view(30,batch_size,-1)
            #print(torch.stack([s_e for s_e in sel],0).size())
        except Exception as e:
            print('error----------------')
            print(len(sel))
            print(sel[105].size())
            print(sel[104])
            print(sel[103].type())
            raise

         #这一步将batch_size的情感向量拼接    （30,batch_size,512）
        imgvecs = torch.stack([ivec for ivec in imgvecs],0).view(1,batch_size,-1)
        #print('sent   ',s_embeddings)
        #测试一下张量大小
        #print(encoder_hidden.size())
        #print(encoder_outputs.size())
        #print(s_embeddings.size())

        h0, c0 = self.init_hidden(batch_size)
        
        if self.use_cuda:
            s_embeddings=s_embeddings.cuda()
            imgvecs = imgvecs.cuda()
            self.lstm = self.lstm.cuda()

            pass
        
        #print(s_embeddings.type())
        #print(h0.type())

        #self.lstm.flatten_parameters()
        s_embeddings, (h, c) = self.lstm(s_embeddings, (h0, c0)) 
        #这里将embedding的结果通过gru 得到最终的embedding
        #print(s_embeddings.size())
        #print('h: ',h.size())
        #t3 = time.time()#记录运行时间
        #print('p3','t%s'%(t3-t2))

        


        decoder_hidden = torch.cat((encoder_hidden,h),2) #这里不能直接拼接，需要先调整形状  现在 个人情感向量的影响并不明显 基本上一个问题的答案是一致的 考虑后续要不要处理
        decoder_hidden = torch.cat((decoder_hidden,imgvecs),2) #这里加入头像向量

        encoder_semb_outputs = torch.cat((encoder_outputs,s_embeddings),0)#这里将sentence embedding的输出和encoder的输出级联作为attn层的输入 
        #这里或许还要加入一个linear层 后续去做

        #print(encoder_semb_outputs.size())


        decoder_inputs = torch.LongTensor([[self.SOS_token]*batch_size])
        #目前没用cuda
        decoder_inputs = decoder_inputs.cuda() if self.use_cuda else decoder_inputs  

        decoder_hidden = decoder_hidden.view(self.num_layers, batch_size, -1) 
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        decoded_words = [[] for i in range(batch_size)] #存结果句子  
        decoded_wid = [[] for i in range(batch_size)]   #存结果序列

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                """
                print('tar',di)
                print('1',decoder_inputs)
                print('2',decoder_hidden.size())
                print('3',encoder_semb_outputs.size())
                """
                decoder_outputs, decoder_hidden = self.generator(decoder_inputs,decoder_hidden, encoder_semb_outputs)

                # 测试结果
                topv, topi = decoder_outputs.data.topk(1)
                #print('tf',topi)
                for i, ind in enumerate(topi):
                    decoded_words[i].append(self.index2word[ind])
                    decoded_wid[i].append(ind)

                decoder_inputs = target_variables[di].view(1, -1)  # Teacher forcing
                loss += criterion(decoder_outputs, target_variables[di])

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                """
                print('tar',di)
                print('1',decoder_inputs)
                print('2',decoder_hidden)
                print('3',encoder_semb_outputs)
                """
                decoder_outputs, decoder_hidden = self.generator(decoder_inputs,decoder_hidden, encoder_semb_outputs)

                topv, topi = decoder_outputs.data.topk(1)

                #print(topi)
                for i, ind in enumerate(topi):
                    decoded_words[i].append(self.index2word[ind])
                    decoded_wid[i].append(ind)

                decoder_inputs = topi.permute(1, 0).detach()  #permute 维度换位 seqlen,batch_size,xxx

                loss += criterion(decoder_outputs, target_variables[di])

        loss.backward()
        #t4 = time.time()#记录运行时间
        #print('p4','t%s'%(t4-t3))
        encoder_optimizer.step()
        decoder_optimizer.step()
        #print(len(decoded_words))
        #self.save_samples(decoded_wid,1)
        #real_data_1.append((decoded_wid,1))
        #t5 = time.time()#记录运行时间
        #print('p5','t%s'%(t5-t4))


        g = random.uniform(0, 10)
        if g>6:

            self.save_samples(real_tar,0)
            self.save_samples(decoded_wid,1) #存放生成结果

            self.save_samples_con(real_tar,h,0)
            #t6 = time.time()#记录运行时间
            #print('p6','t%s'%(t6-t5))

            self.save_samples_con(decoded_wid,h,1)
            #end = time.time()#记录运行时间
            #print('all','t%s'%(start-end))
            #print(decoded_words)
            pass

        return loss.item() / target_length #这一步除啥 是个问题 /？


    def evaluate(self, input_variables, target_variables, lengths,people,senti,criterion):

        input_variables = torch.nn.utils.rnn.pad_sequence(input_variables)
        #print("size ",input_variables.size())
        target_variables = torch.nn.utils.rnn.pad_sequence(target_variables)

        #print("target_variables ",target_variables)

        input_length = input_variables.size()[0]
        target_length = target_variables.size()[0]
        batch_size = target_variables.size()[1]


        with torch.no_grad():
            encoder_hidden = self.encoder.init_hidden(batch_size) #初始化隐层 h_0 (num_layers * num_directions, batch, hidden_size)
            encoder_outputs, encoder_hidden = self.encoder(input_variables, lengths, encoder_hidden)    

            #print('people: ',people.size(),'  ',people[0].item())
            #print('senti: ',len(senti),'  ',senti[0])
            #print(self.index2people[people[0].item()])  
    
    
            sel = [] # s_embedding_list  
            imgvecs = [] 
            for i in range(batch_size): #这里一个一个计算是不是太慢了？整合一起送进去是不是好一些
                #print('people: ',people[i])
                person = self.index2people[people[i].item()]
                #print(person)
                """
                sens = self.sentence_embedding.get_sen_twi(person,senti[i]) #这里pos neg需要 读取 需要重新预处理数据
                #print(sens) #打印句子
                #print(sen_embedding.miss_people[0]) #打印无情感向量名单
                s_embeddings = self.sentence_embedding.get_senti_embedding(sens)    
                """
                #待全部预处理结束就采用这种方法 一次取出30条
                s_embeddings = self.sentence_embedding.get_sen_emb(person,senti[i])[:30]
                #print(len(s_embeddings))
                sel.append(s_embeddings) #这里会一次取30条 

                imgvec_temp = torch.from_numpy(self.p2imgvec[person]).type(torch.FloatTensor)
                imgvecs.append(imgvec_temp)
                pass  


            s_embeddings = torch.stack([s_e for s_e in sel],0).view(30,batch_size,-1) #这一步将batch_size的情感向量拼接    （30,batch_size,512）
            imgvecs = torch.stack([ivec for ivec in imgvecs],0).view(1,batch_size,-1)   
            #print('sent   ',s_embeddings)
            #测试一下张量大小
            #print(encoder_hidden.size())
            #print(encoder_outputs.size())
            #print(s_embeddings.size()) 

            h0, c0 = self.init_hidden(batch_size)   
    

            if self.use_cuda:
                s_embeddings=s_embeddings.cuda()
                self.lstm = self.lstm.cuda()
                imgvecs = imgvecs.cuda()
                pass
                
            s_embeddings, (h, c) = self.lstm(s_embeddings, (h0, c0)) 
            #这里将embedding的结果通过lstm 得到最终的embedding
            #print(s_embeddings.size())
            #print(h.size())    
    
    

            decoder_hidden = torch.cat((encoder_hidden,h),2) #这里不能直接拼接，需要先调整形状  现在 个人情感向量的影响并不明显 基本上一个问题的答案是一致的 考虑后续要不要处理 
            decoder_hidden = torch.cat((decoder_hidden,imgvecs),2) #这里加入头像向量  

            encoder_semb_outputs = torch.cat((encoder_outputs,s_embeddings),0)#这里将sentence embedding的输出和encoder的输出级联作为attn层的输入 
            #这里或许还要加入一个linear层 后续去做 

            #print(encoder_semb_outputs.size()) 

            decoder_inputs = torch.LongTensor([[self.SOS_token]*batch_size])
            #目前没用cuda
            decoder_inputs = decoder_inputs.cuda() if self.use_cuda else decoder_inputs    

            decoder_hidden = decoder_hidden.view(self.num_layers, batch_size, -1) 
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False   

            decoded_words = [[] for i in range(batch_size)]
            decoded_wid = [[] for i in range(batch_size)]   
            
            loss = 0
            for di in range(target_length):
                """
                print('tar',di)
                print('1',decoder_inputs)
                print('2',decoder_hidden)
                print('3',encoder_semb_outputs)
                """
                decoder_outputs, decoder_hidden = self.generator(decoder_inputs,decoder_hidden, encoder_semb_outputs)   

                topv, topi = decoder_outputs.data.topk(1)   

                #print(topi)
                for i, ind in enumerate(topi):
                    decoded_words[i].append(self.index2word[ind])
                    decoded_wid[i].append(ind)  

                decoder_inputs = topi.permute(1, 0).detach()  #permute 维度换位 seqlen,batch_size,xxx
                #print(decoder_outputs.size())
                loss += criterion(decoder_outputs, target_variables[di])    
    
            #print(decoded_words)
            return loss.item() / target_length ,decoded_words#这一步除啥 是个问题 /target_length？


    #该方法主要是为了在sample中 获得encoder层的一些输出作为Generator的输入
    def get_input_encoder(self,input_variables,lengths,people,senti):

        input_variables = torch.nn.utils.rnn.pad_sequence(input_variables)


        input_length = input_variables.size()[0]
        batch_size = input_variables.size()[1]

        encoder_hidden = self.encoder.init_hidden(batch_size) #初始化隐层 h_0 (num_layers * num_directions, batch, hidden_size)

        #print(input_variables.size())
        #print(self.encoder.batch_size)
        #print(encoder_hidden.type())
        encoder_outputs, encoder_hidden = self.encoder(input_variables, lengths, encoder_hidden)

        #print('people: ',people.size(),'  ',people[0].item())
        #print('senti: ',len(senti),'  ',senti[0])
        #print(self.index2people[people[0].item()])


        sel = [] # s_embedding_list   
        imgvecs = []
        for i in range(batch_size): #这里一个一个计算是不是太慢了？整合一起送进去是不是好一些
            #print('people: ',people[i])
            person = self.index2people[people[i].item()]
            #print(person)

            """ 
            sens = self.sentence_embedding.get_sen_twi(person,senti[i]) #这里pos neg需要 读取 需要重新预处理数据
            #print(sens) #打印句子
            #print(sen_embedding.miss_people[0]) #打印无情感向量名单

            s_embeddings = self.sentence_embedding.get_senti_embedding(sens)
            """

            #待全部预处理结束就采用这种方法 一次取出30条
            s_embeddings = self.sentence_embedding.get_sen_emb(person,senti[i])[:30]
            
            sel.append(s_embeddings) #这里会一次取30条
            imgvec_temp = torch.from_numpy(self.p2imgvec[person]).type(torch.FloatTensor)
            imgvecs.append(imgvec_temp)
            pass

        s_embeddings = torch.stack([s_e for s_e in sel],0).view(30,batch_size,-1) #这一步将batch_size的情感向量拼接    （30,batch_size,512）
        imgvecs = torch.stack([ivec for ivec in imgvecs],0).view(1,batch_size,-1)
        #print('sent   ',s_embeddings)
        #测试一下张量大小
        #print(encoder_hidden.size())
        #print(encoder_outputs.size())
        #print(s_embeddings.size())

        h0, c0 = self.init_hidden(batch_size)
        
        if self.use_cuda:
            s_embeddings=s_embeddings.cuda()
            self.lstm = self.lstm.cuda()
            imgvecs = imgvecs.cuda()
            pass
        
        #print(s_embeddings.type())
        #print(h0.type())

        s_embeddings, (h, c) = self.lstm(s_embeddings, (h0, c0)) 
        #这里将embedding的结果通过lstm 得到最终的embedding
        #print(s_embeddings.size())
        #print(h.size())

        decoder_hidden = torch.cat((encoder_hidden,h),2) #这里不能直接拼接，需要先调整形状  现在 个人情感向量的影响并不明显 基本上一个问题的答案是一致的 考虑后续要不要处理
        decoder_hidden = torch.cat((decoder_hidden,imgvecs),2) #这里加入头像向量

        encoder_semb_outputs = torch.cat((encoder_outputs,s_embeddings),0)#

        return decoder_hidden,encoder_semb_outputs

    def sample(self,batch_size,seq_len,data,h,e_s_o):
        #data : (seq_len,batch_size)  这和seqGAN的代码不同
        samples = []
        give_len = data.size(0)
        lis = data.chunk(data.size(0),dim = 0)

        for i in range(give_len):
            decoder_outputs, h = self.generator(lis[i],h, e_s_o,log = False)
            samples.append(lis[i])

        x = decoder_outputs.multinomial(1).permute(1, 0)
        #print(decoder_outputs.size())
        #print(x.size())
        #print(x)

        for i in range(give_len,seq_len):
            samples.append(x)
            decoder_outputs, h = self.generator(x,h,e_s_o,log = False)
            x = decoder_outputs.multinomial(1).permute(1, 0)

        output = torch.cat(samples, dim=0)

        return output
