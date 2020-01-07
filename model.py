import torch
from encoder import *
from decoder import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Encoder, self).__init__()
        self.hidden_size = output_dim
        self.bi = True
        self.rnn = torch.nn.GRU(input_dim, self.hidden_size, num_layers=1, bias=True,
                                batch_first=True, bidirectional=self.bi)
    
    def get_output_dim(self):
        return self.hidden_size * (2 if self.bi else 1)

    def forward(self, x):
        # print(f"x.shape={x.shape}")
        # x.size() == (batch_size, timestamp, feature_dim)
        output, _ = self.rnn(x)
        # print(f"output.shape={output.size()}")
        # output.size() == (seq_len, batch, num_directions * hidden_size)
        # hn.size() = (num_layers * num_directions, batch, hidden_size)
        # output = output.permute(1, 0, 2)
        # hn = hn.permute(1, 0, 2)
        # output.size() == (batch, seq_len, num_directions * hidden_size)
        # hn.size() = (batch, num_layers * num_directions, hidden_size)
        return output


class Attention(torch.nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attn_dim):
        super(Attention, self).__init__()
        self.enc_attn = torch.nn.Linear(encoder_dim, attn_dim)
        self.dec_attn = torch.nn.Linear(decoder_dim, attn_dim)
        self.ful_attn = torch.nn.Linear(attn_dim, 1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out.size() == (batch_size, timestamp, encoder_dim)
        # decoder_hidden.size() == (batch_size, decoder_dim)
        att1 = self.enc_attn(encoder_out) # (batch_size, timestamp, attn_dim)
        att2 = self.dec_attn(decoder_hidden)  # (batch_size, attn_dim)
        att = self.ful_attn(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        # attn.size() == (batch_size, timestamp, attn_dim)
        alpha = self.softmax(att) # (batch_size, timestamp)
        weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        # weighted_encoding.size() == (batch_size, numstamp)
        return weighted_encoding, alpha


class DecoderWithAttention(torch.nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attn_dim, embed_dim, vocab_size):
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attn_dim = attn_dim
        self.vocab_size = vocab_size
        
        self.attention = Attention(encoder_dim, decoder_dim, attn_dim)

        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)

        self.decode_step = torch.nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = torch.nn.Linear(encoder_dim, decoder_dim)
        self.init_c = torch.nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = torch.nn.Linear(decoder_dim, encoder_dim)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.fc = torch.nn.Linear(decoder_dim, vocab_size)
        # self.init_weight()

    def load_pretrained_embedding(self, embedding, fine_tune=False):
        self.embedding.weight = torch.nn.Parameter(embedding, False)
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.sum(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_length):
        # print(f'encoder_out.shape={encoder_out.shape}')
        batch_size = encoder_out.size(0)
        numstamps = encoder_out.size(1)

        # Sort input data by decreasing length
        caption_length, sort_index = caption_length.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_index]
        encoded_captions = encoded_captions[sort_index]

        # embedding
        embeddings = self.embedding(encoded_captions)
        # embeddings.size() == (batch_size, max_caption_length, embed_dim)

        h, c = self.init_hidden_state(encoder_out)
        # h.size() == (batch_size, decoder_dim)

        # not decode at the <end> position
        decode_lengths = (caption_length - 1).tolist()

        # create Tensor for hold word prediction scores and alpha
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), numstamps).to(device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l>t for l in decode_lengths])
            attention_weigthed_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            h, c = self.decode_step(torch.cat([embeddings[:batch_size_t, t, :], attention_weigthed_encoding], dim=1),
                   (h[:batch_size_t], c[:batch_size_t]))
            # (batch_size, decoder_dim)
            preds = self.fc(h) # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_index
