import torch.nn as nn
import torch


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

        self.fc = nn.Linear(text_hidden_size + code_hidden_size, int((text_hidden_size + code_hidden_size)/2))
        self.fc1 = nn.Linear(int((text_hidden_size + code_hidden_size)/2), int((text_hidden_size + code_hidden_size) / 4))
        self.fc2 = nn.Linear(int((text_hidden_size + code_hidden_size) / 4), num_class)

    def forward(self, text_input_ids=None, code_input_ids=None, labels=None):
        text_output = self.textEncoder(text_input_ids, attention_mask=text_input_ids.ne(1))[1]  # [batch_size, hiddensize]
        code_output = self.codeEncoder(code_input_ids, attention_mask=code_input_ids.ne(1))[1]
        combine_output = torch.cat([text_output, code_output], dim=-1)
        logits = self.fc(combine_output)
        logits = self.fc1(logits)
        logits = self.fc2(logits)
        prob = torch.softmax(logits, -1)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob

