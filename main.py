import torch
from torch.utils.data.dataloader import DataLoader
from argparse import ArgumentParser
from dataset import load_dataset


def get_args():    
    # Not Implemented
    return None


def main(args):
    
    #Load Dataset
    train_ds, val_ds, test_ds = load_dataset()
    
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True)





    
















if __name__ == '__main__':
    args = get_args()
    main(args)