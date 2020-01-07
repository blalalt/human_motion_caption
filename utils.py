import os
import re
import json
import pickle
import torch
import numpy as np
from nltk.tokenize import WordPunctTokenizer 
from gensim.models import KeyedVectors

basic_settings = {
    'data_home': 'data',
    'action_description': 'description.txt',
    'skeleton': 'skeleton',
    'description': 'descriptionWB',
    'word2vec': './data/glove.6B.300d.txt'
}


def get_abs_path(path, file):
    return os.path.join(path, file)

def eng_tokenizer():
    token_pattern = r"(?u)\b\w\w+\b"
    pattern = re.compile(token_pattern)
    return lambda doc: pattern.findall(doc)

def save_to_pkl(file, obj):
    with open(file, 'wb') as f:
        pickle.dump(obj, f, )

def load_from_pkl(file):
    with open(file, 'rb') as f:
        obj = pickle.load(f)
    return obj

def save_to_json(file, obj):
    with open(file, 'w', encoding='utf8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_from_json(file):
    with open(file, 'r', encoding='utf8') as f:
        obj = json.load(f)
    return obj

def save_to_txt(file, texts):
    with open(file, 'w', encoding='utf8') as f:
        f.writelines(texts)

def split_text(strings):
    words = WordPunctTokenizer().tokenize(strings)
    return words

def map_to_id(words, word2id, max_len):
    ids = [word2id['<start>']] + [word2id.get(word, '<unk>') for word in words] + \
        [word2id['<end>']] + [word2id['<pad>']] * (max_len - len(words)) 
    return ids, len(words) + 2


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def load_word2vec_model(path):
    model = KeyedVectors.load_word2vec_format(path)
    return model


def init_embedding(embedding):
    bias = np.sqrt(3.0 / embedding.size(0))
    torch.nn.init.uniform_(embedding, -bias, bias)


def load_embedding(emd_file, word2id):
    wv_model = load_word2vec_model(emd_file)
    embed_dim = wv_model.vector_size
    vocab_size = len(word2id)
    embedding = torch.FloatTensor(vocab_size, embed_dim)
    init_embedding(embedding)

    for word in word2id.keys():
        if word not in wv_model:
            continue
        else:
            embedding[word2id[word]] = torch.from_numpy(wv_model.get_vector(word))
    return embedding, embed_dim


def pad(data, max_timestamp):
    time_stamp, dims = data.shape
    z = np.zeros(shape=(max_timestamp-time_stamp, dims))
    return np.vstack((data, z))


class Record:
    def __init__(self):
        self.batch_values = []
        pass

    def update(self, pred, target):
        pass

    def print_batch(self):
        pass

    def print_epoch(self):
        avg_values = self.batch_values / len(self.batch_values)
        self.batch_values = []
        # TODO: 记录训练过程
        pass

    def _format(self):
        pass
