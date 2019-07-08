# _*_ coding: utf-8 _*_
import re
import torch
import spacy
import logging
import pandas as pd
import numpy as np
from torchtext import data
from torchtext import datasets

val_ratio = 0.2
test_ratio = 0.2
max_chars = 20000
nlp = spacy.load('en')  # load model with shortcut link "en_core_web_sm"
logger = logging.getLogger("contract_dataset")
# train_iter, valid_iter, test_iter = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 主要是为了将数据集分成训练，验证，测试三部分
def prepare_csv(seed=999):
    # load training kaggle_data
    df_train = pd.read_csv("SmartContractDataSet/contract_data_csv/contract2_csv.csv")
    df_train["sequence_text"] = df_train.sequence_text.str.replace("\n", " ")
    idx = np.arange(df_train.shape[0])
    # split training_data to training_data & validation_data
    np.random.seed(seed)
    np.random.shuffle(idx)
    val_size = int(len(idx) * val_ratio)
    test_size = int(len(idx) * test_ratio)
    df_train.iloc[idx[val_size + test_size:], :].to_csv("SmartContractDataSet/contract_data_csv/contract_train.csv",
                                                        index=False)
    df_train.iloc[idx[:val_size], :].to_csv("SmartContractDataSet/contract_data_csv/contract_valid.csv", index=False)
    df_train.iloc[idx[val_size:val_size + test_size], :].to_csv("SmartContractDataSet/contract_data_csv/contract_test.csv",
                                                                index=False)


# 标记化(Tokenization) && clean kaggle_data
def tokenizer(contract):
    contract = re.sub(r"[^A-Za-z0-9^,!;\/'+-=_{}()[]", " ", contract)
    contract = re.sub(r",", " , ", contract)
    contract = re.sub(r"\.", " ", contract)
    contract = re.sub(r"!", " ! ", contract)
    contract = re.sub(r"\/", " ", contract)
    contract = re.sub(r"\^", " ^ ", contract)
    contract = re.sub(r"\+", " + ", contract)
    contract = re.sub(r"\-", " - ", contract)
    contract = re.sub(r"\(", " ( ", contract)
    contract = re.sub(r"\)", " ) ", contract)
    contract = re.sub(r"\[", " [ ", contract)
    contract = re.sub(r"\]", " ] ", contract)
    contract = re.sub(r"\{", " { ", contract)
    contract = re.sub(r"\}", " } ", contract)
    contract = re.sub(r"\=", " = ", contract)
    contract = re.sub(r"'", " ", contract)
    contract = re.sub(r";", " ; ", contract)
    contract = re.sub(r":", " : ", contract)
    if (len(contract) > max_chars):
        contract = contract[:max_chars]
    return [x.text for x in nlp.tokenizer(contract) if x.text != " "]


# load kaggle_data
def load_dataset(fix_length=512, lower=False, vectors=None):
    if vectors is not None:
        lower = True
    logger.debug("preprocessing csv...")
    # prepare_csv()
    # 用 torchtext 的 kaggle_data 类加载数据
    contract = data.Field(sequential=True, tokenize=tokenizer, lower=lower, fix_length=fix_length, )

    logger.debug("reading train csv...")
    train_data = data.TabularDataset(
        path="SmartContractDataSet/contract_data_csv/contract_train.csv", format="csv", skip_header=True,
        fields=[
            ('id', None),
            ('sequence_text', contract),
            ('reentrancy', data.Field(use_vocab=False, sequential=False)),
            ('noreentrancy', data.Field(use_vocab=False, sequential=False)),
        ]
    )

    logger.debug("reading validation csv...")
    valid_data = data.TabularDataset(
        path="SmartContractDataSet/contract_data_csv/contract_valid.csv", format="csv", skip_header=True,
        fields=[
            ('id', None),
            ('sequence_text', contract),
            ('reentrancy', data.Field(use_vocab=False, sequential=False)),
            ('noreentrancy', data.Field(use_vocab=False, sequential=False)),
        ]
    )

    logger.debug("reading test csv...")
    test_data = data.TabularDataset(
        path="SmartContractDataSet/contract_data_csv/contract_test.csv", format="csv", skip_header=True,
        fields=[
            ('id', None),
            ('sequence_text', contract)
        ]
    )

    logger.debug("build vocab")
    contract.build_vocab(train_data, valid_data, test_data, vectors=vectors)

    logger.debug("preprocessing end!")

    word_embeddings = contract.vocab.vectors
    train_iter, valid_iter, test_iter = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=32,
        sort_key=lambda x: len(x.sequence_text),
        repeat=False,
        shuffle=True,
        device=device
    )
    vocab_size = len(contract.vocab)
    print("Length of contract Vocabulary: ", str(len(contract.vocab)))
    # print("Vector size of contract Vocabulary: ", word_embeddings.size())
    print("Length of contract Vocabulary: ", vocab_size)

    return contract, vocab_size, word_embeddings, train_iter, valid_iter, test_iter, len(train_data), len(valid_data)


# 分batch和迭代
def get_iterator(dataset, batch_size, train=True, shuffle=True, repeat=False):
    dataset_iter = data.Iterator(
        dataset, batch_size=batch_size, device=device,
        train=train, shuffle=shuffle, repeat=repeat,
        sort=False
    )

    return dataset_iter
