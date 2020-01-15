import torch
import random
from collections import Counter, defaultdict
from model import Encoder, DecoderWithAttention, Attention
from utils import *
from prepare_data import load, MyDataSet, save_result
from torch.utils.data.dataset import Subset
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(1)
batch_size = 64
lr = 0.004
epoches = 100
alpha_c = 1
print_freq = 1


def _get_train_test_indice(class_index, ratio):
    sample_indice = list(range(len(class_index)))
    train_indice = []
    test_indice = []
    cls_indexes_map = defaultdict(lambda: [])
    for i in sample_indice:
        cl_inx = class_index[i]
        cls_indexes_map[cl_inx].append(i)
    # map(lambda i: cls_indexes_map[class_index[i]].append(i), sample_indice)
    for k, v in cls_indexes_map.items():
        _c_len = len(v)
        _point = int(_c_len * ratio)
        train_indice += v[:_point]
        test_indice += v[_point:]
    return train_indice, test_indice


def _get_data_loader(dataset, ratio, batch_size):
    dataset_length = len(dataset)
    action_index = dataset.action_index
    train_indice, test_indice = _get_train_test_indice(action_index, ratio)

    assert dataset_length == (len(train_indice) + len(test_indice))
    assert len(set(train_indice + test_indice)) == dataset_length

    train_dataset = Subset(dataset, indices=train_indice)
    test_dataset = Subset(dataset, indices=test_indice)
    train_dataloader = DataLoader(train_dataset, num_workers=1, pin_memory=True,
                                  shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, num_workers=1, pin_memory=True,
                                 shuffle=True, batch_size=batch_size)
    return train_dataloader, test_dataloader


def main(data_name):
    dataset = MyDataSet(data_name=data_name, reset=False)
    vocab_size = dataset.vocab_size
    corpus = dataset.corpus
    id2word = {v: k for k, v in corpus.items()}
    train_loader, val_loader = _get_data_loader(dataset, 0.5, batch_size)

    embedding, embed_dim = load_embedding(basic_settings['word2vec'], corpus)

    encoder = Encoder(dataset.feature_dim, output_dim=100)
    decoder = DecoderWithAttention(encoder.get_output_dim(), decoder_dim=100,
                                   attn_dim=100, embed_dim=embed_dim, vocab_size=vocab_size)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=lr)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    best_bleu4 = 0
    best_hypos = []
    best_refs = []

    for epoch in range(1, epoches + 1):
        # One epoch's training
        train_epoch(train_loader=train_loader,
                    encoder=encoder,
                    decoder=decoder,
                    criterion=criterion,
                    optimizer=decoder_optimizer,
                    epoch=epoch)

        # One epoch's validation
        bleu4_score, refs, hypos = validate(val_loader=val_loader,
                                            encoder=encoder,
                                            decoder=decoder,
                                            criterion=criterion,
                                            word2id=corpus)
        if bleu4_score > best_bleu4:
            best_bleu4 = bleu4_score
            best_refs = refs
            best_hypos = hypos
    name = data_name + '_' + str(best_bleu4) + '.xlsx'
    save_result(name, best_refs, best_hypos, id2word)


def train_epoch(train_loader, encoder, decoder, optimizer, criterion, epoch):
    encoder.train()
    decoder.train()

    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    for i, (data, cap, cap_length) in enumerate(train_loader):
        data = data.to(device)
        caps = cap.to(device)
        caps_length = cap_length.to(device)

        encoded_video = encoder(data)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(encoded_video, caps, caps_length)

        targets = caps_sorted[:, 1:]
        # print(f'score.shape={scores.shape}, target.shape={targets.shape}')
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

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


def validate(val_loader, encoder, decoder, criterion, word2id):
    encoder.eval()
    decoder.eval()

    losses = AverageMeter()
    top5accs = AverageMeter()

    references = []
    hypotheses = []

    with torch.no_grad():
        for _, (data, caps, caps_length) in enumerate(val_loader):
            data = data.to(device)
            caps = caps.to(device)
            caps_length = caps_length.to(device)

            encoded_video = encoder(data)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(encoded_video, caps, caps_length)

            targets = caps_sorted[:, 1:]

            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))

            for j in range(caps_sorted.shape[0]):  # remove <start> and pads
                img_caps = caps_sorted[j].tolist()
                img_captions = list([w for w in img_caps
                                     if w not in {word2id['<start>'], word2id['<pad>']}])
                references.append([img_captions])

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4, references, hypotheses


if __name__ == "__main__":
    data_sets = ['combined_15', 'WorkoutUOW_18', 'MSRC_12', 'WorkoutSU_10', 'Combined_17']
    for data_set in data_sets:
        main(data_name=data_set)
