from dataset import load_clean_dataset
from modelling.utils import *
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer
from argparse import ArgumentParser
import torch




def get_args():    
    # Not Implemented
    parser = ArgumentParser()
    parser.add_argument('--max_len', type=int, default='5000')
    parser.add_argument('--batch_size', type= int, default = 64)
    args = parser.parse_args()

    return args


def main(args):
    
    #Get Tokenizers
    en_tokenizer = AutoTokenizer.from_pretrained('models/en_tokenizer')
    de_tokenizer = AutoTokenizer.from_pretrained('models/de_tokenizer')

    #Load Dataset
    train_ds, val_ds, test_ds = load_clean_dataset(en_tokenizer=en_tokenizer, de_tokenizer=de_tokenizer, max_len=args.max_len)
    
    #Build DataLoader
    train_loader = DataLoader(train_ds, batch_size = args.batch_size, collate_fn = translation_collate_fn, shuffle = True)

    #Initialize Model

    




if __name__ == '__main__':
    args = get_args()
    main(args)


    
















if __name__ == '__main__':
    args = get_args()
    main(args)