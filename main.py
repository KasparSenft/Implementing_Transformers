from dataset import load_clean_dataset
from modelling.utils import *
from modelling import Transformer

from transformers import AutoTokenizer
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Subset
from torch import nn
from loguru import logger
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import List
import json


from rich.traceback import install

#Add some pretty tracebacks
# install(show_locals=True)




def get_args():    
    # Not Implemented
    parser = ArgumentParser()

    #Tokenizer Parameters
    parser.add_argument('--trgt', type=str, default ='de')
    parser.add_argument('--max_len', type=int, default='50')

    #Model Parameters
    parser.add_argument('--vocab_size', type=int, default=50001)
    parser.add_argument('--d_model', type=int, default =512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type = int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--dim_feed_forward', type=int, default=2048)


    #Training params
    parser.add_argument('--batch_size', type= int, default=64)
    parser.add_argument('--epochs', type = int, default=10)
    parser.add_argument('--learning_rate', type=float, default=10e-2)
    parser.add_argument('--weight_decay', type=float, default=10e-2)
    parser.add_argument('--warmup_steps', type=int, default=2)
    parser.add_argument('--subset', type=int, default = None)
    

    #logging/saving
    parser.add_argument('--outdir', type =str, default='experiments')
    parser.add_argument('--exp_name', type=str, default='en-de')


    args = parser.parse_args()

    return args


def main(args):
    
    #Get Tokenizers
    en_tokenizer = AutoTokenizer.from_pretrained('models/en_tokenizer')
    de_tokenizer = AutoTokenizer.from_pretrained('models/de_tokenizer')

    #Load Dataset
    train_ds, val_ds, test_ds = load_clean_dataset(en_tokenizer=en_tokenizer, de_tokenizer=de_tokenizer, trgt = args.trgt, max_len=args.max_len)
    

    if args.subset is not None:
        sub_indices = range(args.subset)
        train_ds = Subset(train_ds, sub_indices)
        args.batch_size = min(args.batch_size, args.subset)


    #Build DataLoader
    train_loader = DataLoader(train_ds, batch_size = args.batch_size, collate_fn = translation_collate_fn, shuffle = True)

 

    #Initialize Model

    logger.info('Building model...')

    model = Transformer(
        vocab_size=args.vocab_size,
        d_model= args.d_model,
        num_heads = args.num_heads,
        num_layers = args.num_layers,
        dim_feed_forward=args.dim_feed_forward,
        dropout = args.dropout,
        max_len = args.max_len
    )

    logger.info('Model built!')

    #Initialize LR-Scheduler
    optimizer = get_adam_optimizer(model, lr = args.learning_rate, weight_decay=args.weight_decay)
    scheduler = LearningRateScheduler(optimizer, args.d_model, args.warmup_steps)
    criterion = nn.CrossEntropyLoss()

    logger.info('Commencing Training')
    #Train Model
    for epoch in range(args.epochs):

        total_loss = 0
        
        for batch in train_loader:
            
            src_lang = 'en' if args.trgt == 'de' else 'en'

            #Get relevant tokens and masks
            src_tokens, src_masks = batch[src_lang]
            trgt_tokens, trgt_masks = batch[args.trgt]

            #get labels (undo right shift)
            labels = torch.cat([trgt_tokens[:,1:], trgt_tokens[:,-1].unsqueeze(dim=-1)], dim = -1)


            #Forward pass
            outputs = model(src_tokens,src_masks, trgt_tokens, trgt_masks)

            #Calculate Loss
            loss = criterion(outputs.permute(0,2,1), labels)

            #Keep track of loss
            total_loss += loss

            #Backward Pass
            optimizer.zero_grad()
            loss.backward()

            #Update Weights
            optimizer.step()

            #Update Schedule
            scheduler.step()


        total_loss /= len(train_loader)

        #Log the loss
        logger.info(f'Epoch:{epoch}/{args.epochs}: Average Loss: {total_loss}') 

    #Save final model
    logger.info(f'Saving model in {args.outdir}/model.pt')
    torch.save(model.state_dict(), f'{args.outdir}/model.pt')
    

if __name__ == '__main__':

    args = get_args()

    #Make a directory to save the experiment
    now = '({:02d}.{:02d}.{}|'.format(datetime.now().day, datetime.now().month, datetime.now().year) + \
              datetime.now().strftime("%H.%M.%S") + ')'
    args.outdir = f'{args.outdir}/{args.exp_name + now}'
    Path(args.outdir).mkdir(parents = True, exist_ok=True)

    #Tell the logger where to log
    logger.add(f'{args.outdir}/logging.log')
    logger.info(f'Beggining Experiment {args.exp_name}')

    #Save the hyperparameters
    with open(f'{args.outdir}/hp.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    main(args)