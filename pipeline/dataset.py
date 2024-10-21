import json
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


def read_json(file_name):
    with open(file_name, 'r') as file:
        return [json.loads(line) for line in file]

def json_dataset_parser(jsons_list, labels_dict):
    data_dict = {"text": [], "labels": [], "domain": []}
    for obj in jsons_list:
        data_dict["text"].append(obj["text"])
        data_dict["labels"].append(labels_dict[obj["label"]])
        data_dict["domain"].append(obj["domain"])
    return pd.DataFrame(data_dict)

def prepare_dataset(file_path, labels_dict, test_size=0.15, val_size=0.15, sample_frac=1.0):
    jsons_list = read_json(file_path)
    df = json_dataset_parser(jsons_list, labels_dict)
    df = df.sample(frac=sample_frac).reset_index(drop=True)

    train_val, test = train_test_split(df, test_size=test_size, stratify=df['labels'])
    train, val = train_test_split(train_val, test_size=val_size/(1-test_size), stratify=train_val['labels'])

    dataset = DatasetDict({
        'train': Dataset.from_pandas(train),
        'val': Dataset.from_pandas(val),
        'test': Dataset.from_pandas(test)
    })
    return dataset


def save_tokenized_dataset(tokenized_datasets, save_path):
    tokenized_datasets.save_to_disk(save_path)

def load_tokenized_dataset(load_path):
    return DatasetDict.load_from_disk(load_path)