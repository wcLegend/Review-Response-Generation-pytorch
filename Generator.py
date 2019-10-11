import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from Attention import X_attn  
#本质是一个gru的decoder

class Generator(nn.Module):
    """docstring for Generator"""
    def __init__(self, hidden_size, embedding, num_layers=1, ELMo_embedding=False,
                 train_embedding=True, dropout_p=0.2):
        super(Generator, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        #self.use_cuda =False

        self.hidden_size = hidden_size # 这里暂不确定 先以二倍为准
        self.dropout_p = dropout_p
        self.num_layers = num_layers

        if ELMo_embedding:  #采用ELMO的embedding
            self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
            self.embedding.weight = nn.Parameter(embedding)
            self.input_size = embedding.shape[1] # Size of embedding vector
            self.output_size = embedding.shape[0] # Number of words in vocabulary

        else:
            self.embedding = nn.Embedding(embedding[0], embedding[1])
            self.input_size = embedding[1] # Size of embedding vector
            self.output_size = embedding[0] # Number of words in vocabulary
            #print('emb_weight',self.embedding.weight)
            nn.init.xavier_normal_(self.embedding.weight)
            #print('emb_weight_new',self.embedding.weight)
            #example_input = torch.ones(1,128).long()
            #self.embedding = torch.jit.trace(self.embedding,example_input)

        self.embedding.weight.requires_grad = train_embedding

        self.X_attn = X_attn(hidden_size)  
        #attn的结果的我维度是hidden_size  embedding结果是input_size
        self.gru = nn.GRU(hidden_size + self.input_size, self.hidden_size, self.num_layers, dropout=dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        nn.init.xavier_normal_(self.out.weight) #初始化

        #example_input = torch.randn(1, 128, 428)
        #example_hidden = torch.randn(1, 128, 768)
        #self.gru = torch.jit.trace(self.gru,(example_input,example_hidden))

        #example_input = torch.randn(128, 768)
        #self.out = torch.jit.trace(self.out,example_input)

    def forward(self, input, hidden,encoder_hiddens, h0_s,ax,log = True):
        '''
        input           -> (1 x Batch Size)
        hidden          -> (Num. Layers * Num. Directions x Batch Size x Hidden Size)
        encoder_hiddens ->(seq_len,batch_size,hidden_size) 编码器每一步的隐层
        h0_s            ->(30,batch_size,hidden_size)  个性化情感向量的所有隐层
        ax              ->历史权重和
        '''

        input_emb = self.embedding(input) #(1,b,e_v)

        #max_length = encoder_outputs.size(0)

        
        ci,ax = self.X_attn(hidden,encoder_hiddens, h0_s,ax) 
        #print('input: ',input_emb.size())
        #print('context: ',ci.size())
        ci = ci.unsqueeze(0) #（1，batch_size,hidden_size）
        gru_input = torch.cat((input_emb, ci), 2) #

        #print('gru_input: ',gru_input.size())
        #print('hidden: ',hidden.size())
        output, hidden = self.gru(gru_input, hidden)

        output = output.squeeze(0) # (1, B, V) -> (B, V)

        #print('out_input: ',output.size())
        if log:
            output = F.log_softmax(self.out(output), dim=1)
        else: #这里不取log 在rollout里会用到  用在拓展完整序列 sample
            output = F.softmax(self.out(output), dim=1)

        return output, hidden,ax#, attn_weights


    def init_hidden(self, batch_size):
        #if batch_size == 0: batch_size = self.batch_size
        #result = torch.zeros(2 * self.num_layers, batch_size, self.hidden_size) #双向

        result = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        result = nn.init.xavier_normal_(result)
        if self.use_cuda:
            return result.cuda()
        else:
            return result