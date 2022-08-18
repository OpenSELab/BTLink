import logging

import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, config.num_labels)

    def forward(self, input_ids=None, labels=None):
        output = self.encoder(input_ids, attention_mask=input_ids.ne(1))
        # print(output)
        last_hidden_state, pooler_output = output[0], output[1]
        logits = self.fc(pooler_output)
        lg = self.fc2(logits)
        prob = torch.softmax(lg, -1)
        # print('logits:', logits)
        # print('prob:', prob)
        # logits = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        # print(self.encoder(input_ids, attention_mask=input_ids.ne(1)))
        # logging.getLogger().info(logits)
        # prob = torch.softmax(logits, -1)
        # print('logits:', logits)
        # print('prob:', prob)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            # print(lg)
            # print(labels)
            loss = loss_fct(lg, labels)
            return loss, prob
        else:
            return prob


class BTModel(nn.Module):
    def __init__(self, textEncoder, codeEncoder,
                 text_hidden_size, code_hidden_size, num_class):
        super(BTModel, self).__init__()
        self.textEncoder = textEncoder
        self.codeEncoder = codeEncoder
        self.text_hidden_size = text_hidden_size
        self.code_hidden_size = code_hidden_size
        self.num_class = num_class
        for param in self.textEncoder.parameters():
            param.requires_grad = True
        for param in self.codeEncoder.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(text_hidden_size + code_hidden_size, num_class)

    def forward(self, text_input_ids=None, code_input_ids=None, labels=None):
        text_output = self.textEncoder(text_input_ids, attention_mask=text_input_ids.ne(1))[1]  # [batch_size, hiddensize]
        code_output = self.codeEncoder(code_input_ids, attention_mask=code_input_ids.ne(1))[1]
        # print(text_output, text_output.shape)  # [batchsize, hiddensize]
        # print(code_output, code_output.shape)
        # 将text_output 与 code_output 合并，batch_size不动，只在hiddensize上合并
        combine_output = torch.cat([text_output, code_output], dim=-1)
        # print('combine_output.shape:',combine_output.shape)
        logits = self.fc(combine_output)
        prob = torch.softmax(logits, -1)
        # print('logits:',logits)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            # print(lg)
            # print(labels)
            # print('logits:', logits)
            # print('labels:', labels)
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob


class MyModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(MyModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.convs = nn.ModuleList(
            # 输入通道数,输出通道数（卷积核数），卷积核维度
            [nn.Conv2d(1, 256, (k, 1024)) for k in [3, 4, 5]]
            # (k,config.hidden_size)  n-gram,embedding维度
        )
        self.fc = nn.Linear(256 * 3,
                            2)  # 输入的最后一个维度，输出的最后一个维度 全连接层只改变数据的最后一个维度 由输入最后的一个维度转化为 类别数

        def conv_and_pool(self, x, conv):
            x = conv(x)  # [batch_size,channel_num,pad_size,embedding_size（1）]
            x = F.relu(x)
            x = x.squeeze(3)  # [batch_size,channel_num,pad_size]
            x = F.max_pool1d(x, x.size(2))  # 经过卷积之后，x
            x = x.squeeze(2)  # [batch_size,channel_num]
            return x

    def forward(self, input_ids=None, labels=None):
        logits = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]  # logits 包括cls在内的所有token的向量表示
        prob = torch.softmax(logits, -1)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob
