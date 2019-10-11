import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        img_size = 512
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 3+img_size, hidden_size) #这里hidden的维度要大一倍因为有sentence embedding  加入头像向量 成四倍

    def forward(self, hidden, encoder_outputs):
        #print('hidden_size ',hidden.size())
        #print('encoder_outputs_size ',encoder_outputs.size())


        '''
        hidden : Previous hidden state of the Generator (1 x Batch Size x Hidden Size*2)
        encoder_outputs: Outputs from Encoder (Sequence Length x Batch Size x Hidden Size)

        return: Attention energies in shape (Batch Size x Sequence Length)
        '''
        max_len = encoder_outputs.size(0) # Encoder Outputs -> L, B, V
        batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        #print('H_size ',H.size())
        attn_energies = self.score(H, encoder_outputs.transpose(0, 1)) # compute attention score
        return F.softmax(attn_energies, dim=1).unsqueeze(1) # normalize 

    def score(self, hidden, encoder_outputs):

        #print('errrr1',hidden.size())
        #print('errrr2',encoder_outputs.size())

        #energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B, L, 3H]->[B, L, H]  #用sigmoid尝试替换tanh
        energy = torch.sigmoid(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(2, 1) # [B, H, L]
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1) #[B, 1, H]
        energy = torch.bmm(v, energy).squeeze(1) # [B, L]
        return energy



class X_attn(nn.Module):
    def __init__(self, hidden_size):
        super(X_attn, self).__init__()
        #img_size = 512

        self.hidden_size = hidden_size
        self.attn_size = int(hidden_size/2)  #attn的隐层维度
        # 这部分是为了处理关于encoder中每个隐层计算attn权重用到的权重。
        self.Linear1  = nn.Linear(self.hidden_size, self.attn_size,bias = False)
        self.Linear2  = nn.Linear(self.hidden_size, self.attn_size,bias = False)
        self.Linear3  = nn.Linear(1, self.attn_size,bias = False)
        self.Linear4  = nn.Linear(self.attn_size*3,1,bias = False)

        #这部分是为了处理关于产品信息的attn权重
        self.Linear5  = nn.Linear(self.hidden_size, self.attn_size,bias = False)
        self.Linear6  = nn.Linear(self.hidden_size, self.attn_size,bias = False)
        self.Linear7  = nn.Linear(self.hidden_size, self.attn_size,bias = False)
        self.Linear8  = nn.Linear(self.attn_size*3,1,bias = False)

        #这部分是GMU用到的参数
        self.Linear9 = nn.Linear(self.hidden_size, self.hidden_size,bias = False)
        self.Linear10 = nn.Linear(self.hidden_size, self.hidden_size,bias = False)
        self.Linear11 = nn.Linear(self.hidden_size*2, 1,bias = False)
        self.sigmoid = nn.Sigmoid()
        



    def forward(self, hidden,encoder_hiddens,h0_s,ax):

        #ax的维度是(batch_size,encoder_hidden_num,1)
        batch_size = hidden.size()[1]
        encoder_hidden_num = encoder_hiddens.size()[0]

        #print('hidden_size ',hidden.size())
        #print('encoder_hiddens ',encoder_hiddens.size())
        #print('h0_s ',h0_s.size())
        #print('ax  ',ax.size())
        exi = []
        for j in range(encoder_hidden_num):
            wx = self.Linear1(encoder_hiddens[j])
            wxy = self.Linear2(hidden.squeeze(0))
            wxr = self.Linear3(ax[:,j])
            #print(ax[:,j].size())

            hidden_tmp = torch.cat((wx,wxy,wxr),1)
            #print('hidden_tmp.size()',hidden_tmp.size())
            exij = nn.Tanh()(hidden_tmp)
            exij = self.Linear4(exij)  #(batch_size,1)
            exi.append(exij)
            pass
        exi = torch.stack(exi,1)  #(batch_size,encoder_hidden_num,1)
        #print('exi.size()',exi.size())
        axi = F.softmax(exi,dim=1)
        #print('axi.size()',axi.size())

        axi_t = axi.permute(0,2,1)
        encoder_hiddens_t = encoder_hiddens.permute(1,0,2)
        cxi = torch.bmm(axi_t,encoder_hiddens_t).squeeze(1)
        #print('cxi ',cxi.size())   #(batch_size,hidden_size)

        ax = ax+axi
        #print('ax.size',ax.size())

        #print('开始计算个性化attn============')
        h0_s_num = h0_s.size()[0]
        eti = []
        for j in range(h0_s_num):
            wt = self.Linear5(h0_s[j])
            wtx = self.Linear6(cxi)
            wty = self.Linear7(hidden.squeeze(0))

            #print('wt  ',wt.size())
            #print('wtx  ',wtx.size())
            #print('wty  ',wty.size())



            hidden_tmp = torch.cat((wt,wtx,wty),1)
            #print('hidden_tmp.size()',hidden_tmp.size())
            etij = nn.Tanh()(hidden_tmp)
            etij = self.Linear8(etij)
            eti.append(etij)
            pass

        eti = torch.stack(eti,1)  #(batch_size,encoder_hidden_num,1)
        #print('eti.size()',eti.size())
        ati = F.softmax(eti,dim=1)
        #print('ati.size()',ati.size())

        ati_t = ati.permute(0,2,1)
        h0_s_t = h0_s.permute(1,0,2)
        cti = torch.bmm(ati_t,h0_s_t).squeeze(1)
        #print('cti ',cti.size())   #(batch_size,hidden_size)


        #print('开始计算GMU')
        gxi = nn.Tanh()(self.Linear9(cxi))
        #print('gxi.size()',gxi.size())
        gti = nn.Tanh()(self.Linear10(cti))
        cxti = torch.cat((cxi,cti),1)
        #print('cxti.size()',cxti.size())
        zi = self.sigmoid(self.Linear11(cxti)) 
        #print('zi  ',zi.size())

        #print('1-zi',1-zi)
        #print('* size',(zi*gxi).size())  
        ci = (zi*gxi)+(1-zi)*gti   #这里直接用惩罚是没有问题的
        #print(ci.size())
        return ci,ax

    def score(self, hidden, encoder_outputs):

        #print('errrr1',hidden.size())
        #print('errrr2',encoder_outputs.size())

        #energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B, L, 3H]->[B, L, H]  #用sigmoid尝试替换tanh
        energy = torch.sigmoid(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(2, 1) # [B, H, L]
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1) #[B, 1, H]
        energy = torch.bmm(v, energy).squeeze(1) # [B, L]
        return energy


