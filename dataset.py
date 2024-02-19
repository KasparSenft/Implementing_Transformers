import datasets
from datasets import load_dataset, load_from_disk
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import re

class TranslationDataset(Dataset):
    def __init__(self, dataset,):
        """
        args:
            dataset (datasets.Dataset)
        """
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


def load_dataset():
    ds = load_from_disk('/home/senft/ImpTransf/data/wmt17_de-en_cleaned.hf')
    train_ds = TranslationDataset(ds['train'])
    val_ds = TranslationDataset(ds['validation'])
    test_ds = TranslationDataset(ds['test'])

    return train_ds,val_ds,test_ds


def clean_string(text, whitelist):
    """
    args:
        s (str); string to be cleaned
        whitelist (set); permissable characters

    returns:
        (str); s with all non whitelist characters removed
    """

    # Remove non-UTF8 characters
    text = text.encode("utf-8", "ignore").decode("utf-8")

    # Remove URLs and HTML tags
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", text)
    text = re.sub(r"<.*?>", "", text)

    # Remove characters not in the whitelist
    text = ''.join(c for c in text if c in whitelist)
    
    return text


def clean_dataset(dataset, min_len = 5, max_len = 64, max_ratio = 1.5):

    WHITELIST = set('abcdefghijklmnopqrstuvwxyz ÄÖÜäöüß ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}:;-&$@#%£€/\\|_+*¥')
    clean_dataset = datasets.DatasetDict()

    #train, test, val splits
    splits = dataset.keys()

    for split in splits:
        data_split = {
            'en': [],
            'de': []
        }

        for entry in tqdm(dataset[split], desc = split):
            en_text = entry['translation']['en']
            de_text = entry['translation']['de']

            cleaned_en = clean_string(en_text, WHITELIST)
            cleaned_de = clean_string(de_text, WHITELIST)

            if min_len <= min(len(cleaned_en), len(cleaned_de)) and max(len(cleaned_en), len(cleaned_de)) <= max_len:
                #Check ratios of lengths
                ratio = len(cleaned_de)/len(cleaned_en)
                if 1/max_ratio <= ratio <= max_ratio:
                    data_split['en'].append(cleaned_en)
                    data_split['de'].append(cleaned_de)

                
        clean_dataset[split] = datasets.Dataset.from_dict(data_split)

    return clean_dataset



if __name__ == '__main__':

    outdir = 'data/'

    #clean data
    dirty_dataset = load_dataset("wmt17", "de-en")
    cleaned_dataset = clean_dataset(dirty_dataset)
    
    #save data
    cleaned_dataset.save_to_disk(outdir + 'wmt17_de-en_cleaned.hf')



    