from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
import re
import itertools
import sys

# datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
# dataset = sys.args[1]
dataset = 'mr'

# if dataset in datasets:
#     sys.exit('wrong dataset')

nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
# print(stopwords)

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

# print(clean_str('sakjflkasjf9    21u4032432/1234124///124314/12123/4/1'))

doc_content_list = []
with open('data/' + dataset + '.txt', 'rb') as f:
    for line in f.readlines():
        doc_content_list.append(line.strip().decode('latin1'))

# print(doc_content_list[:5])
word_freq = {}
for doc in doc_content_list:
    texts = clean_str(doc).split()
    for text in texts:
        word_freq[text] = word_freq[text] + 1 if text in word_freq else 1

# print(dict(itertools.islice(word_freq.items(), 10))) # slice in dict

clean_docs = []
for doc_content in doc_content_list:
    words = clean_str(doc_content).split()
    doc_word = []
    # print(words)
    for word in words:
        if dataset == 'mr':
            doc_word.append(word)
        if word not in stopwords and word_freq[word] >= 5:
            doc_word.append(word)   
    #### ????
    doc_str = ' '.join(doc_word).strip()
    # print(doc_str)
    clean_docs.append(doc_str)

clean_docs_corpus = '\n'.join(clean_docs)
# print(clean_docs_corpus)
with open('data/corpus/' + dataset + '_clean.txt', 'w') as f:
    f.write(clean_docs_corpus)

min_len = 10000
average_len = 0
max_len = 0

with open('data/corpus/' + dataset + '_clean.txt', 'r') as f:
    lines = f.readlines()
    count_null_len = 0
    for index, line in enumerate(lines): 
        line = line.strip().split()
        average_len = average_len + len(line)
        if len(line) == 0:
            count_null_len += 1
        min_len = min(min_len, len(line))
        max_len = max(max_len, len(line))

# print("'s 's best work yet , girl woman believes world 's misery blind good".strip().split())
average_len = 1.0 * average_len / len(lines)
print(f'Max len: {max_len}')
print(f'Min len: {min_len}')
print(f'Zero-len: {count_null_len}')
print(f'Average len: {average_len}')