""" Libraries """
import os
import time
import shutil
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataloader import MyMapDataset, split_datasets



""" Functions """
def train_step(batch_data):
    pass


def valid_step(batch_data):
    pass


def train_epoch(dataloader):
    pbar = tqdm(enumerate(dataloader), ascii=True, desc="[TRAIN]")
    for batch_i, (batch_data) in pbar:
        train_step(batch_data)
    return


def valid_epoch(dataloader):
    pbar = tqdm(enumerate(dataloader), ascii=True, desc="[VALID]")
    for batch_i, (batch_data) in pbar:
        valid_step(batch_data)
    return


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    shutil.copy(__file__, args.save_dir)
    shutil.copy("models.py", args.save_dir)
    shutil.copy("dataloader.py", args.save_dir)
    my_data = list(range(100))
    my_dataset = MyMapDataset(my_data)
    my_train_dataset, my_valid_dataset = split_datasets(my_dataset)
    my_train_dataLoader = DataLoader(
        my_train_dataset, args.batch_size, shuffle=True, sampler=None,
        batch_sampler=None, num_workers=args.num_workers, collate_fn=None,
        pin_memory=True, drop_last=False, timeout=0, worker_init_fn=None,
        multiprocessing_context=None, generator=None,
        prefetch_factor=2, persistent_workers=False
    )
    my_valid_dataLoader = DataLoader(
        my_valid_dataset, args.batch_size, shuffle=True, sampler=None,
        batch_sampler=None, num_workers=args.num_workers, collate_fn=None,
        pin_memory=True, drop_last=False, timeout=0, worker_init_fn=None,
        multiprocessing_context=None, generator=None,
        prefetch_factor=2, persistent_workers=False
    )
    for epoch in range(1, args.epochs+1):
        print(f"{epoch}/{args.epochs}")
        train_epoch(my_train_dataLoader)
        valid_epoch(my_valid_dataLoader)
    return



""" Execution """
if __name__ == "__main__":

    DEFAULT_SAVE_DIR = f"logs/{time.strftime('%Y.%m.%d-%H.%M.%S', time.localtime())}"

    parser = argparse.ArgumentParser()
    parser.add_argument("-e",  "--epochs",      type=int, default=100)
    parser.add_argument("-bs", "--batch-size",  type=int, default=32)
    parser.add_argument("-nw", "--num-workers", type=int, default=8)
    parser.add_argument(       "--save-dir",    type=str, default=DEFAULT_SAVE_DIR)

    args = parser.parse_args()
    main(args)