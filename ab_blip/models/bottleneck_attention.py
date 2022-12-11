from typing import List
import torch
from torch import nn
import numpy as np

class AB(nn.module):
    def __init__(self):
        super().__init__()
        
        self.vb_att = BertAttention()
        self.tb_att = BertAttention()

    def forward(self, vit_output, text_output, bottleneck=None):  # per layer
        if bottleneck is None:
            bottleneck = guassian(batch, bt_size)

        # (batch, head, seq_len, hidden_size)
        vb = self.vb_att(hidden_states=bottleneck, encoder_hidden_states=vit_output)[0]
        tb = self.tb_att(hidden_states=bottleneck, encoder_hidden_states=text_output)[0]

        return ffn((vb + tb) / 2)
