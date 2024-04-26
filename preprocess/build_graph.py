import os
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import sys

# from utils.utils import clea
# if len(sys.argv) != 2:
#     sys.exit('Use: python build_graph.py <dataset>')

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
dataset = datasets[-1]
random.seed(42)

word_embeddings_dim = 300
word_vector_map = {}

doc_name_list, doc_train_list, doc_test_list = [], [], []
with open('data/' + dataset + '_label.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        doc_name_list.append(line.strip())
        temp = line.split('\t')
        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip()) # remove '\n'
        # else: 
        #     print(temp[0])
         
doc_content_list = []
with open('data/corpus/' + dataset + '_clean.txt', 'r') as f:
    lines = f.readlines()
    doc_content_list = [line.strip() for line in lines]
# print(len(doc_train_list) + len(doc_test_list))
train_ids = [doc_name_list.index(line) for line in doc_train_list] 
# print(train_ids[:100])
random.shuffle(train_ids)

train_ids_str = '\n'.join(str(index) for index in train_ids)
with open('data/' + dataset + '.train.index', 'w') as f:
    f.write(train_ids_str)

test_ids = [doc_name_list.index(line) for line in doc_test_list]
random.shuffle(test_ids)
test_ids_str = '\n'.join(str(index) for index in test_ids)
with open('data/' + dataset + '.test.index', 'w') as f:
    f.write(test_ids_str)
ids = train_ids + test_ids

shuffle_doc_name_list = [doc_name_list[id] for id in ids]
shuffle_doc_content_list = [doc_content_list[id] for id in ids]

# print(shuffle_doc_name_list[0])
# print(shuffle_doc_content_list[0])

with open('data/' + dataset + '_shuffle.txt', 'w') as f:
    f.write('\n'.join(shuffle_doc_name_list))

with open('data/corpus/' + dataset + '_shuffle.txt', 'w') as f:
    f.write('\n'.join(shuffle_doc_content_list))

# print(shuffle_doc_content_list[0].split())
word_freq = {}
vocab = set()
for content in shuffle_doc_content_list:
    content = content.split()
    for word in content:
        vocab.add(word)
        word_freq[word] = word_freq[word] + 1 if word in word_freq else 1

print(len(vocab))