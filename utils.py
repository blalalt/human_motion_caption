import os
import re
import json
import pickle

basic_settings = {
    'data_home': 'data',
    'action_description': 'description.txt',
    'skeleton': 'skeleton',
    'description': 'descriptionWB'
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
    with open(file) as f:
        obj = pickle.load(f)
    return obj

def save_to_json(file, obj):
    with open(file, 'w', encoding='utf8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_to_txt(file, texts):
    with open(file, 'w', encoding='utf8') as f:
        f.writelines(texts)