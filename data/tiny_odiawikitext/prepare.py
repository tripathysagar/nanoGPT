import os
import numpy as np
import subprocess
from tokenizers import Tokenizer


urls = [
    'https://media.githubusercontent.com/media/tripathysagar/odia_text_prep/refs/heads/main/wikipedia/odia_wiki_tinny/train.txt',
    'https://media.githubusercontent.com/media/tripathysagar/odia_text_prep/refs/heads/main/wikipedia/odia_wiki_tinny/valid.txt',
    'https://raw.githubusercontent.com/tripathysagar/odia_text_prep/refs/heads/main/od_tokenizer_hf.json'
]


for url in urls:
  subprocess.run(['wget', '-q', '-P', './data_src/', url])


tokenizer = Tokenizer.from_file('data/od_tokenizer_hf.json')


val_ids = []
with open("data_src/valid.txt", 'r', encoding='utf-8') as file:
    for i, line in enumerate(file):
        val_ids.extend(tokenizer.encode(line.strip()).ids)

val_ids = np.array(val_ids, dtype=np.uint16)
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))


train_ids = []
with open("data_src/train.txt", 'r', encoding='utf-8') as file:
    for i, line in enumerate(file):
        train_ids.extend(tokenizer.encode(line.strip()).ids)

train_ids = np.array(train_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))


print(f"train : {train_ids.shape[0]} tokens\nvalid : {val_ids.shape[0]} tokens")
