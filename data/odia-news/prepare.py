# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

tok_url = 'https://raw.githubusercontent.com/tripathysagar/odia_text_prep/refs/heads/main/od_tokenizer_hf.json'
import os
from tqdm import tqdm
import numpy as np
from tokenizers import Tokenizer
import subprocess
from datasets import load_dataset

subprocess.run(['wget', '-q', '-P', './data_src/', tok_url])
tokenizer = Tokenizer.from_file('data_src/od_tokenizer_hf.json') # huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = os.cpu_count()

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

eot_token = tokenizer.token_to_id("<|endoftext|>")


if __name__ == '__main__':
    # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
    dataset = load_dataset("tripathysagar/odia-news", num_proc=num_proc_load_dataset).remove_columns('url')
    dataset['val'] = dataset.pop('test')
    """
    DatasetDict({
        train: Dataset({
            features: ['text'],
            num_rows: 233501
        })
        val: Dataset({
            features: ['text'],
            num_rows: 9730
        })
    })
    """

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        ids = tokenizer.encode(example['text']).ids # encode_ordinary ignores any special tokens
        ids.append(eot_token) # add the end of text token,
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # train.bin is ~377MB, val.bin ~8.5MB
    # train has ~197M tokens (197,386,239)
    # val has ~8M tokens (8,171,949)

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
