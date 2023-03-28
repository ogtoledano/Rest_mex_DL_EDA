from builtins import super
from collections import OrderedDict

import torch.nn as nn
from transformers.models.roberta.modeling_roberta import RobertaEncoder
from transformers import RobertaModel
from transformers.modeling_outputs import TokenClassifierOutput
import torch

class CustomRoBERTaModel(nn.Module):
    def __init__(self, labels):
        super(CustomRoBERTaModel, self).__init__()
        self.labels = labels

        # Load Model with given checkpoint and extract its body
        self.model = RobertaModel.from_pretrained("roberta-base")
        self.config =self.model.config
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(768, 64)  # load and initialize weights
        self.fc = nn.Linear(64, labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None, labels_ids=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Add custom layers
        sequence_output = self.dropout(outputs.last_hidden_state)  # outputs[0]=last hidden state
        sequence_output = sequence_output[:, 0, :]
        logits = self.dense(torch.reshape(sequence_output,(-1, 768)))  # calculate losses torch.reshape(sequence_output,(-1, 4096))
        logits = torch.tanh(logits)
        logits = self.dropout(logits)
        logits = self.fc(logits)
        logits = torch.tanh(logits)
        logits = self.dropout(logits)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            logits=logits.view(-1, self.labels)
            loss = loss_fct(logits, labels_ids.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.last_hidden_state,
                                     attentions=outputs.attentions)