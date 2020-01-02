import numpy
import pandas
from utils import *
import scipy.io as scio
import os
import shutil

bs = basic_settings

def _id_description(path):
    id_desc = {}
    action_id = 1
    with open(path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip().lower()
            action_name, descs = line.split(":")
            num_actions = len(descs.split('.'))
            id_desc[action_id] = {'name': action_name, 'num_actions': num_actions,
                                    'action_desc': descs}
            action_id += 1
    return id_desc

def _to_3d(matrix, nodes=20, dims=3):
    """
    @matrix: [[r1, g1, b1, r2, g2, b2, ....], [.....]]
    """
    new_matrix = np.array(matrix).reshape(shape=(1, nodes, dims))
    return new_matrix

def _get_data_path(data_name):
    data_home = bs['data_home']
    skeleton_dir = 'skeleton'
    desc_file = 'description.txt'
    this_data_home = get_abs_path(data_home, data_name)
    skeletion_data_path = get_abs_path(this_data_home, skeleton_dir)
    desc_path = get_abs_path(this_data_home, desc_file)
    return skeletion_data_path, desc_path

def _action_id_by_name(name):
    prefix = name.split('_')[0]
    action_id = int(prefix[1:])
    return action_id

def _load_data(data_name):
    dataset = []
    skeleton_path, desc_path =  _get_data_path(data_name)
    id_desc = _id_description(desc_path)
    skeleton_data = _load_skeleton(skeleton_path)
    for f_name in  skeleton_data.keys():
        data = {}
        a_id = _action_id_by_name(f_name)
        data['data'] = skeleton_data[f_name]
        data['action_id'] = a_id
        data['num_seg_actions'] = id_desc[a_id]['num_actions']
        data['desc'] = id_desc[a_id]['action_desc']
        data['action_name'] = id_desc[a_id]['name']
        dataset.append(data)
    return dataset

def _load_skeleton(path):
    _action_tag = 'one_sample'
    _map = {}
    for file in os.listdir(path):
        file_name = os.path.splitext(file)[0]
        file_path = get_abs_path(path, file)
        video_mat = scio.loadmat(file_path)[_action_tag]
        _map[file_name] = video_mat
    return _map


def _get_corpus(dataset):
    corpus = set()
    tokenizer = eng_tokenizer()
    for sample in dataset:
        desc = sample['desc']
        words = tokenizer(desc)
        corpus.update(words)
    return corpus

def _load_corpus(data_name):
    home = get_abs_path(bs['data_home'], data_name)
    processed_home = get_abs_path(home, 'processed')
    words = []
    with open(get_abs_path(processed_home, 'corpus.txt')) as f:
        for line in f.readlines:
            words.append(line.strip())
    word2id = {id: word for id, word in enumerate(words)}
    return word2id

def build(data_name, reset=False):
    home = get_abs_path(bs['data_home'], data_name)
    if not os.path.exists(home):
        raise Exception(f"Can't find origin data at {home}")
    processed_home = get_abs_path(home, 'processed')
    if not os.path.exists(processed_home) or reset:
        shutil.rmtree(processed_home, ignore_errors=True)
        os.makedirs(processed_home)
    dataset = _load_data(data_name)
    corpus = _get_corpus(dataset)
    corpus = ['<start>', '<end>'] + list(corpus)
    corpus = {word: _id for _id, word in enumerate(corpus)}
    save_to_pkl(get_abs_path(processed_home, 'dataset.pkl'), dataset)
    save_to_json(get_abs_path(processed_home, 'corpus.json'), corpus)


def load(data_name, reset):
    home = get_abs_path(bs['data_home'], data_name)
    processed_home = get_abs_path(home, 'processed')
    if reset or not os.path.exists(processed_home):
        build(data_name)
    dataset = load_from_pkl(get_abs_path(processed_home, 'dataset.pkl'))
    corpus = load_from_json(get_abs_path(processed_home, 'corpus.json'))
    return dataset, corpus

if __name__ == "__main__":
    data_names = ['combined_15']
    dataset, corpus = load(data_names[0], reset=True)
    print(dataset)
    print(corpus)

# def load_combined_15(reset=False):
#     data_name = 'combined_15'
#     home = get_abs_path(bs['data_home'], data_name)
#     processed_home = get_abs_path(home, 'processed')
#     if processed_home and reset:
#         shutil.rmtree(processed_home)
#     if not os.path.exists(processed_home):
#         os.makedirs(processed_home)
#     id2desc = _id_description(home)
#     video_sq = _load_skeleton(get_abs_path(home, bs['skeleton']))
#     descriptions = _load_description(get_abs_path(home, bs['description']), id2desc)
#     assert len(video_sq) == len(descriptions)
#     assert list(video_sq.keys()) == list(descriptions.keys())
#     corpus = _get_corpus(descriptions.values())

#     # save
#     data = [] # data [{data:, desc:}], corpus: word+\n+word...
#     for f_name, video in video_sq.items():
#         desc = descriptions.get(f_name)
#         data.append({'data': video, 'desc': desc})
#     corpus = '\n'.join(corpus)

#     save_to_pkl(get_abs_path(processed_home, 'data.pkl'), data)
#     save_to_txt(get_abs_path(processed_home, 'corpus.txt'), corpus)
#     pass


# def _load_description(path, id2desc):
#     _action_tag = 'action'
#     _map = {}
#     for file in os.listdir(path):
#         file_name = os.path.splitext(file)[0]
#         file_path = get_abs_path(path, file)
#         actions = scio.loadmat(file_path)[_action_tag].flatten()
#         description = ''.join(map(lambda action_id: id2desc[action_id].lower(), 
#                                 actions))
#         _map[file_name] = description
    
#     return _map