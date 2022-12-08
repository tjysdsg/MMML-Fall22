class AB:
    def __init__(self):
        self.vb_att = BertAttention()
        self.tb_att = BertAttention()

    def forward(self, vit_output, text_output, bottleneck=None):  # per layer
        if bottleneck is None:
            bottleneck = guassian(batch, bt_size)

        vb = self.vb_att(hidden_states=vit_output, encoder_hidden_states=bottleneck)[0]
        tb = self.tb_att(hidden_states=text_output, encoder_hidden_states=bottleneck)[0]

        return ffn((vb + tb) / 2)
