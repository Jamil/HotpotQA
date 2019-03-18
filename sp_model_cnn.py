
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
from torch.nn import init
from torch.nn.utils import rnn

class SPModel(nn.Module):
    def __init__(self, config, word_mat, char_mat):
        super().__init__()
        self.config = config
        self.word_dim = config.glove_dim
        self.word_emb = nn.Embedding(len(word_mat), len(word_mat[0]), padding_idx=0)
        self.word_emb.weight.data.copy_(torch.from_numpy(word_mat))
        self.word_emb.weight.requires_grad = False
        self.char_emb = nn.Embedding(len(char_mat), len(char_mat[0]), padding_idx=0)
        self.char_emb.weight.data.copy_(torch.from_numpy(char_mat))

        self.char_cnn = nn.Conv1d(config.char_dim, config.char_hidden, 5)
        self.char_hidden = config.char_hidden
        self.hidden = config.hidden

        res_block_count = 5
        self.res_block_count = 5

        kernel1 = (5, config.char_hidden+len(word_mat[0]))
        self.cnn1 = GatedCNN(1, kernel1, 1, config.hidden, res_block_count)
        self.self1 = BiAttention(config.hidden, 1-config.keep_prob)
        self.cnn2 = GatedCNN(1, kernel1, 1, config.hidden, res_block_count)
        self.self2 = BiAttention(config.hidden, 1-config.keep_prob)
        self.linear0 = nn.Linear(config.hidden*4, config.hidden)
        nn.init.xavier_uniform_(self.linear0.weight, gain=nn.init.calculate_gain('relu'))
        self.qc_att1 = BiAttention(config.hidden, 1-config.keep_prob)

        self.linear1 = nn.Linear(config.hidden*4, config.hidden)
        nn.init.xavier_uniform_(self.linear1.weight, gain=nn.init.calculate_gain('relu'))
        self.relu = nn.LeakyReLU()

        kernel2 = (5, config.hidden)
        self.cnn3 = GatedCNN(1, kernel2, 1, config.hidden, res_block_count)
        self.qc_att2 = BiAttention(config.hidden, 1-config.keep_prob)

        self.linear2 = nn.Linear(config.hidden*4, config.hidden)
        nn.init.xavier_uniform_(self.linear2.weight, gain=nn.init.calculate_gain('relu'))

        self.cnn4 = GatedCNN(1, kernel2, 1, config.hidden, res_block_count)
        self.self_att = BiAttention(config.hidden, 1-config.keep_prob)


        self.linear3 = nn.Linear(config.hidden*4, config.hidden)
        nn.init.xavier_uniform_(self.linear3.weight, gain=nn.init.calculate_gain('relu'))

        self.cnn_sp    = GatedCNN(1, kernel2, 1, 2*config.hidden, res_block_count)
        self.lin_sp    = nn.Linear(config.hidden*2, 1)
        nn.init.xavier_uniform_(self.lin_sp.weight, gain=nn.init.calculate_gain('relu'))

        kernel3 = (5, 3*config.hidden)
        self.cnn_start = GatedCNN(1, kernel3, 1, 2*config.hidden, res_block_count)
        self.lin_start = nn.Linear(config.hidden*2, 1)
        nn.init.xavier_uniform_(self.lin_start.weight, gain=nn.init.calculate_gain('relu'))

        self.cnn_end   = GatedCNN(1, kernel3, 1, 2*config.hidden, res_block_count)
        self.lin_end   = nn.Linear(config.hidden*2, 1)
        nn.init.xavier_uniform_(self.lin_end.weight, gain=nn.init.calculate_gain('relu'))

        self.cnn_type  = GatedCNN(1, kernel3, 1, 2*config.hidden, res_block_count)
        self.lin_type  = nn.Linear(config.hidden*2, 3)
        nn.init.xavier_uniform_(self.lin_type.weight, gain=nn.init.calculate_gain('relu'))

        self.cache_S = 0


    def get_output_mask(self, outer):
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=False):
        para_size, ques_size, char_size, bsz = context_idxs.size(1), ques_idxs.size(1), context_char_idxs.size(2), context_idxs.size(0)

        context_mask = (context_idxs > 0).float()
        ques_mask = (ques_idxs > 0).float()

        context_ch = self.char_emb(context_char_idxs.contiguous().view(-1, char_size)).view(bsz * para_size, char_size, -1)
        ques_ch = self.char_emb(ques_char_idxs.contiguous().view(-1, char_size)).view(bsz * ques_size, char_size, -1)

        context_ch = self.char_cnn(context_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, para_size, -1)
        ques_ch = self.char_cnn(ques_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, ques_size, -1)

        context_word = self.word_emb(context_idxs)
        ques_word = self.word_emb(ques_idxs)

        context_output = torch.cat([context_word, context_ch], dim=2)
        ques_output = torch.cat([ques_word, ques_ch], dim=2)


        context_output = self.cnn1(context_output)
        context_output = self.self1(context_output, context_output, context_mask)
        ques_output    = self.cnn2(ques_output)
        ques_output    = self.self2(ques_output, ques_output, ques_mask)
        context_output = self.linear0(context_output)
        ques_output    = self.linear0(ques_output)
        output = self.qc_att1(context_output, ques_output, ques_mask)
        output = self.linear1(output)
        output = self.relu(output)

        output1 = self.cnn3(output)
        output1 = self.qc_att2(output1, ques_output, ques_mask)
        output1 = self.linear2(output1)
        output1 = self.relu(output1)

        output2 = self.cnn4(output)
        output2 = self.self_att(output2, output2, context_mask)
        output2 = self.linear3(output2)
        output2 = self.relu(output2)

        output = (output+output2+output1)

        sp_output = self.cnn_sp(output)

        start_output = torch.matmul(start_mapping.permute(0, 2, 1).contiguous(), sp_output[:,:,self.hidden:])
        end_output = torch.matmul(end_mapping.permute(0, 2, 1).contiguous(), sp_output[:,:,:self.hidden])
        sp_output = torch.cat([start_output, end_output], dim=-1)
        sp_output_t = self.lin_sp(sp_output)
        sp_output_t = self.relu(sp_output_t)
        sp_output_aux = Variable(sp_output_t.data.new(sp_output_t.size(0), sp_output_t.size(1), 1).zero_())
        predict_support = torch.cat([sp_output_aux, sp_output_t], dim=-1).contiguous()

        sp_output = torch.matmul(all_mapping, sp_output)
        output_start = torch.cat([output, sp_output], dim=-1)

        output_start = self.cnn_start(output_start)
        logit1 = self.relu(self.lin_start(output_start).squeeze(2)) - 1e30 * (1 - context_mask)
        # logit1 = F.normalize(logit1)
        output_end = torch.cat([output, output_start], dim=2)
        output_end = self.cnn_end(output_end)
        logit2 = self.relu(self.lin_end(output_end).squeeze(2)) - 1e30 * (1 - context_mask)
        # logit2 = F.normalize(logit2)

        output_type = torch.cat([output, output_end], dim=2)
        output_type = torch.max(self.cnn_type(output_type), 1)[0]
        predict_type = self.relu(self.lin_type(output_type))
        # predict_type = F.normalize(predict_type)

        logit1 = F.normalize(logit1)
        logit2 = F.normalize(logit2)
        predict_type = F.normalize(predict_type)
        predict_support = F.normalize(predict_support) 
        if not return_yp: return logit1, logit2, predict_type, predict_support

        outer = logit1[:,:,None] + logit2[:,None]
        outer_mask = self.get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        yp1 = outer.max(dim=2)[0].max(dim=1)[1]
        yp2 = outer.max(dim=1)[0].max(dim=1)[1]

        # yp1 = F.normalize(yp1)
        # yp2 = F.normalize(yp2)
        return logit1, logit2, predict_type, predict_support, yp1, yp2

class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x

class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        att = att - 1e30 * (1 - mask[:,None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)

class GateLayer(nn.Module):
    def __init__(self, d_input, d_output):
        super(GateLayer, self).__init__()
        self.linear = nn.Linear(d_input, d_output)
        self.gate = nn.Linear(d_input, d_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        return self.linear(input) * self.sigmoid(self.gate(input))

class GatedCNN(nn.Module):
    '''
        In : (N, sentence_len)
        Out: (N, sentence_len, embd_size)
    '''
    def __init__(self,
                 n_layers,
                 kernel,
                 in_chns,
                 out_chs,
                 res_block_count):
        super(GatedCNN, self).__init__()
        self.res_block_count = res_block_count
        self.conv_0 = nn.Conv2d(in_chns, 2*out_chs, kernel, padding=(2, 0))
        self.conv_bn = nn.BatchNorm2d(2*out_chs)
        self.b_0 = nn.Parameter(torch.randn(1, 2*out_chs, 1, 1))
        self.conv_gate_0 = nn.Conv2d(in_chns, 2*out_chs, kernel, padding=(2, 0))
        self.gate_bn = nn.BatchNorm2d(2*out_chs)
        self.c_0 = nn.Parameter(torch.randn(1, 2*out_chs, 1, 1))
        self.pool = nn.MaxPool1d(2, stride=2) 
        self.relu = nn.LeakyReLU()


    def forward(self, x):
        # x: (N, seq_len)

        # Embedding
        bs = x.size(0) # batch size
        seq_len = x.size(1)

        # CNN
        x = x.unsqueeze(1) # (bs, Cin, seq_len, embd_size), insert Channnel-In dim
        # Conv2d
        #    Input : (bs, Cin,  Hin,  Win )
        #    Output: (bs, Cout, Hout, Wout)
        A = self.conv_0(x)      # (bs, Cout, seq_len, 1)
        A += self.b_0.repeat(1, 1, seq_len, 1)
        A = self.conv_bn(A)
        A = self.relu(A)
        B = self.conv_gate_0(x) # (bs, Cout, seq_len, 1)
        B += self.c_0.repeat(1, 1, seq_len, 1)
        B = self.gate_bn(B)
        h = A * F.sigmoid(B)
        h = A.squeeze(dim=3)
        h = h.permute(0, 2, 1)
        h = self.pool(h)

        return h
