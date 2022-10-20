""" Libraries """
import os
import time
import shutil
import argparse
from tqdm import tqdm



""" Functions """
def train_step():
    raise NotImplementedError


def valid_step():
    raise NotImplementedError


def train_epoch():
    raise NotImplementedError


def valid_epoch():
    raise NotImplementedError


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    shutil.copy(__file__, args.save_dir)
    return



""" Execution """
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir", type=str,
        default=f"logs/{time.strftime('%Y.%m.%d-%H.%M.%S', time.localtime())}",
        help="path of directory to save log files",
    )
    args = parser.parse_args()
    main(args)