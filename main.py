from dataset import load_clean_dataset
from modelling.utils import *
from modelling import Transformer
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer
from argparse import ArgumentParser
from torch import nn
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
    model = Transformer(
        vocab_size=args.vocab_size,
        d_model= args.d_model,
        num_heads = args.num_heads,
        num_decoder_layers = args.num_decoder_layers
        dim_feed_forward=args.dim_feed_forward,
        dropout = args.dropout,
        max_len = args.max_len
    )

    #Initialize LR-Scheduler
    optimizer = get_adam_optimizer(model, args.weight_decay)
    scheduler = LearningRateScheduler(optimizer, args.d_model, args.warmup_steps)
    criterion = torch.nn.CrossEntropyLoss()

    #Train Model
    for epoch in range(args.epochs):
        
        for batch in train_loader:
            
            trgt_lang = 'en' if args.src_lang == 'de' else 'en'

            #Get relevant tokens and masks
            src_tokens, src_masks = batch[args.src_lang]
            trgt_tokens, trgt_masks = batch[trgt_lang]

            #Forward pass
            outputs = model(src_tokens,src_masks, trgt_tokens, trgt_masks)

            #Calculate Loss
            loss = criterion(outputs, trgt_tokens)

            #Backward Pass
            optimizer.zero_grad()
            loss.backward()

            #Update Weights
            optimizer.step()

            #Update Schedule
            scheduler.step()







    




if __name__ == '__main__':
    args = get_args()
    main(args)


    
















if __name__ == '__main__':
    args = get_args()
    main(args)