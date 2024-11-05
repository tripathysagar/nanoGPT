import os
import numpy as np
import subprocess
from tokenizers import Tokenizer
from fastcore.parallel import parallel


urls = [
    'https://media.githubusercontent.com/media/tripathysagar/odia_text_prep/refs/heads/main/wikipedia/odia_wiki_full/train.txt',
    'https://media.githubusercontent.com/media/tripathysagar/odia_text_prep/refs/heads/main/wikipedia/odia_wiki_full/valid.txt',
    'https://raw.githubusercontent.com/tripathysagar/odia_text_prep/refs/heads/main/od_tokenizer_hf.json'
]


for url in urls:
  subprocess.run(['wget', '-q', '-P', './data_src/', url])


tokenizer = Tokenizer.from_file('data_src/od_tokenizer_hf.json')

dir = [
    ['data_src/train.txt', os.path.join(os.path.dirname(__file__), 'train.bin'),],
    ['data_src/valid.txt', os.path.join(os.path.dirname(__file__), 'val.bin')]
]

def tokenize_text(lis):
    txt_file, bin_file = lis
    ids = []
    with open(txt_file, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
          ids.extend(tokenizer.encode(line.strip()).ids)
    ids = np.array(ids, dtype=np.uint16)
    ids.tofile(os.path.join(os.getcwd(), bin_file))
    return ids.shape[0]

results = parallel(tokenize_text, dir, n_workers=2, progress=True)

print(f"train : {results[0]} tokens\nvalid : {results[1]} tokens")
