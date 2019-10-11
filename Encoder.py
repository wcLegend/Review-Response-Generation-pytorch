import torch
import torch.nn as nn
from torch import Tensor
#from elmo import elmo_emb  #版本似乎有问题 待测试

#加入elmo_embedding 

class Encoder(nn.Module):
    """docstring for Encoder"""
    def __init__( self, hidden_size, embedding, num_layers=1, batch_size=1, ELMo_embedding=False, train_embedding=True):
        super(Encoder, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        #self.use_cuda = False
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        #self.idx2word = idx2word  elmo有Bug所以隐去
        
        if ELMo_embedding:
            #self.elmo_embedding = elmo_emb()
            #self.input_size = self.input_size +1024
            print("use elmo ")

        else: 
            self.embedding = nn.Embedding(embedding[0], embedding[1])
            self.input_size = embedding[1]
            self.ELMo_embedding = ELMo_embedding

        self.gru = nn.GRU(self.input_size, hidden_size, self.num_layers, bidirectional=False)  #gru的output的形状是output (seq_len, batch, hidden_size * num_directions)

        #这里加入jit 尝试进行加速
        #_input = torch.ones(15,128).long()
        #self.embedding = torch.jit.trace(self.embedding,_input)
        #jit不接受pack_padded_sequence之后的结果作为输入，或许gru不能使用

        self.embedding.weight.requires_grad = train_embedding #用来控制是否训练embedding
        #print('emb_weight',self.embedding.weight)
        nn.init.xavier_normal_(self.embedding.weight)
        #print('emb_weight_new',self.embedding.weight)


        """ #gru其实本身有初始化,不需要进行初始化
        print('gru_weight',self.gru.all_weights)
        #nn.init.xavier_normal_(self.gru.all_weights)
        print('gru_weight_new',self.gru.all_weights[0].size())
        """


        self.weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)  #elmo会得到两个向量  这里取其权重
        self.weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.weight_1.data.fill_(0.5)
        self.weight_2.data.fill_(0.5)


    def forward(self, input, input_lengths, hidden):
        """ 这里input_lengths 就是相应的长度
        input           -> (seq_len, batch, input_size)
        input_lengths   -> (Batch Size (Sorted in decreasing order of lengths))
        hidden          -> (num_layers * num_directions, batch, hidden_size)
        """

        #print('encoder embedding input ',input.size())
        embedded = self.embedding(input) # L, B, V
        seq_len = embedded.size()[0]
        #packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        #print(embedded.size())
        #print(seq_len)
        hidden_list = []
        for i in range(seq_len):
            #print('aa',embedded[i])
            #print('bb',embedded[i].size())
            outputs, hidden = self.gru(embedded[i].unsqueeze(0), hidden)#.unsqueeze(0)增加seq_len的维度
            hidden_list.append(hidden)
            pass
        #print(len(hidden_list))
        hiddens = torch.stack(hidden_list,0).squeeze(1)
        #print(hiddens.size())
        #print('encoder gru hidden',hidden.size())
        

        #outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        #print(outputs.size())
        # 这里也可以采用双向GRU

        #outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return hiddens

    def init_hidden(self, batch_size=0):
        if batch_size == 0: batch_size = self.batch_size
        #result = torch.zeros(2 * self.num_layers, batch_size, self.hidden_size) #双向
        result = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        if self.use_cuda:
            return result.cuda()
        else:
            return result