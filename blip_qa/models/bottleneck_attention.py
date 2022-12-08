class AB:
    def __init__(self):
        self.vb_att = cross_att()
        self.tb_att = cross_att()

    def forward(self, vit_output, text_output, bottleneck=None):  # per layer
        if bottleneck is None:
            bottleneck = guassian(batch, bt_size)

        vb = self.vb_att(vit_output, bottleneck)
        tb = self.tb_att(text_output, bottleneck)

        return ffn((vb + tb) / 2)
