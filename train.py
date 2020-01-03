import torch
import random
from model import Encoder, DecoderWithAttention, Attention
from utils import *
from prepare_data import load, MyDataSet
from torch.utils.data.dataset import Subset
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
lr = 0.004
epoches = 100
alpha_c = 1
print_freq = 100


def _get_data_loader(dataset, ratio, batch_size):
    dataset_length = len(dataset)
    sample_indice = list(range(dataset_length)) 
    random.shuffle(sample_indice)
    train_indice = sample_indice[:int(dataset_length*ratio)]
    test_indice = sample_indice[int(dataset_length*ratio):]
    train_dataset = Subset(dataset, indices=train_indice)
    test_dataset = Subset(dataset, indices=test_indice)
    train_dataloader = DataLoader(train_dataset, num_workers=1, pin_memory=True,
                                  shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, num_workers=1, pin_memory=True,
                                  shuffle=True, batch_size=batch_size)
    return train_dataloader, test_dataloader

def main(data_name):
    dataset = MyDataSet(data_name=data_name)
    vocab_size = dataset.vocab_size
    corpus = dataset.corpus
    train_loader, val_loader = _get_data_loader(dataset, 0.9, batch_size)

    embedding, embed_dim = load_embedding(basic_settings['word2vec'], corpus)
    
    encoder = Encoder(dataset.feature_dim, output_dim=100)
    decoder = DecoderWithAttention(encoder.get_output_dim(), decoder_dim=100,
                                    attn_dim=100, embed_dim=embed_dim, vocab_size=vocab_size)
    decoder.load_pretrained_embedding(embedding)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                                 lr=lr)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for epoch in range(1, epoches+1):
        # One epoch's training
        train_epoch(train_loader=train_loader,
                    encoder=encoder,
                    decoder=decoder,
                    criterion=criterion,
                    optimizer=decoder_optimizer,
                    epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)



def train_epoch(train_loader, encoder, decoder, optimizer, criterion, epoch):
    encoder.train()
    decoder.train()

    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    for i, (data, cap, cap_length) in enumerate(train_loader):
        data = data.to(device)
        caps = cap.to(device)
        caps_length = cap_length.to(device)

        encoded_video, _ = encoder(data)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(encoded_video, caps, caps_length)

        targets = caps_sorted[:, 1:]

        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        loss = criterion(scores, targets)
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder, decoder, criterion):
    pass
