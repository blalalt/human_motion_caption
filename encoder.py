import torch


class RNNEncoder(torch.nn.Module):
    def __init__(self, in_channels=60, out_channels=120):
        super().__init__()
        self.rnn = torch.nn.LSTM(input_size=in_channels,
                                hidden_size=out_channels,
                                batch_first=True)
        pass

    def forward(self, x):
        batch = x.shape[0]
        output, (h_n, _) =  self.rnn(x)
        # h_n = _h_n.view(batch, -1)
        output = 1
        return h_n