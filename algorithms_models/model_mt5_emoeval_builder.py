from builtins import super
from collections import OrderedDict

import torch.nn as nn
from transformers.models.mt5 import MT5ForConditionalGeneration
from transformers.modeling_outputs import TokenClassifierOutput


class CustomMT5Model(nn.Module):
    def __init__(self, labels):
        super(CustomMT5Model, self).__init__()
        self.labels = labels

        # Load Model with given checkpoint and extract its body
        self.model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, labels)  # load and initialize weights

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Add custom layers
        sequence_output = self.dropout(outputs[0])  # outputs[0]=last hidden state

        logits = self.fc(sequence_output[:, 0, :].view(-1, 768))  # calculate losses

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
                                     attentions=outputs.attentions)