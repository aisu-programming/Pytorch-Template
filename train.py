""" Libraries """
import os
import time
# import json
import torch
import shutil
# import random
import argparse
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Tuple
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from libs.models import MyModel
from libs.datasets import MyMapDataset, split_datasets



""" Classes """
class Metric():
    def __init__(self, length) -> None:
        self.length = length
        self.values = []

    def append(self, value) -> None:
        self.values.append(value)
        if len(self.values) > self.length: self.values.pop(0)
        return

    def avg(self) -> float:
        return np.average(self.values)


""" Functions """
def get_weighted_acc(pred, truth, weight) -> float:
    weighted_acc = 0.0
    # assert np.max(np.max(pred), np.max(truth)) <= len(weight)
    for wid, wt in enumerate(weight):
        weighted_acc += (np.logical_and(pred==wid, pred==truth).sum()/np.max((1, (truth==wid).sum()))) * wt
    return weighted_acc


def get_lr(optimizer) -> float:
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    

def plot_confusion_matrix(cm_length, cm, filename, title):
    cm_df = pd.DataFrame(cm, index=list(range(cm_length)), columns=list(range(cm_length)))
    plt.figure(figsize=(6, 5))
    cm_image:plt.Axes = sn.heatmap(cm_df, annot=True)
    cm_image.set_xlabel("prediction", fontsize=10)
    cm_image.set_ylabel("truth", fontsize=10)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return


def forward_step(model, batch_inputs, device) -> torch.Tensor:
    batch_inputs = batch_inputs.to(device)
    return model(batch_inputs)


def train_epoch(model:torch.nn.Module,
                dataloader:List[Tuple[torch.Tensor, torch.Tensor]],
                criterion,
                optimizer:torch.optim.Optimizer,
                lr_scheduler:torch.optim.lr_scheduler._LRScheduler,
                device,
                weight:list=None,
                cm_length:int=0):
    
    model = model.train()

    loss_metric = Metric(50)
    acc_metric  = Metric(50)
    if weight is not None: acc_weight = (np.array(weight)/sum(weight)).tolist()
    if cm_length != 0: confusion_matrixs = np.zeros((cm_length, cm_length))

    pbar = tqdm(dataloader, ascii=True, desc="[TRAIN]")
    for batch_inputs, batch_truth in pbar:

        model.zero_grad()
        batch_pred  = forward_step(model, batch_inputs, device)
        batch_truth = batch_truth.to(device)

        loss:torch.Tensor = criterion(batch_pred, batch_truth)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        loss_metric.append(loss.item())

        batch_truth = batch_truth.cpu().detach().numpy()
        batch_pred  = batch_pred.cpu().detach().numpy()
        acc = get_weighted_acc(batch_pred, batch_truth, acc_weight)
        acc_metric.append(acc)
        
        if cm_length != 0:
            confusion_matrixs += \
                confusion_matrix(batch_truth, batch_pred, labels=list(range(cm_length)))  # , sample_weight=weight)
        pbar.set_description(f"[TRAIN] loss: {loss_metric.avg():.5f}, " + \
                             f"Acc: {acc_metric.avg()*100:.3f}%, " + \
                             f"LR: {get_lr(optimizer):.10f}")
        
    # with np.printoptions(linewidth=150, formatter={'float': '{:5.03f}'.format, 'int': '{:2} '.format}):
    #     print("pred :", batch_pred)
    #     print("truth:", batch_truth)
    #     print("corct:", np.array(['X', 'O'])[np.uint8(batch_pred==batch_truth)])
    
    if cm_length != 0:
        return loss_metric.avg(), acc_metric.avg(), confusion_matrixs
    else:
        return loss_metric.avg(), acc_metric.avg()


def valid_epoch(model:torch.nn.Module,
                dataloader:List[Tuple[torch.Tensor, torch.Tensor]],
                criterion,
                device,
                weight:list=None,
                cm_length:int=0):
    
    model = model.eval()

    loss_metric = Metric(10000)
    acc_metric  = Metric(10000)
    if weight is not None: acc_weight = (np.array(weight)/sum(weight)).tolist()
    if cm_length != 0: confusion_matrixs = np.zeros((cm_length, cm_length))

    pbar = tqdm(dataloader, ascii=True, desc="[TRAIN]")
    for batch_inputs, batch_truth in pbar:

        model.zero_grad()
        batch_pred  = forward_step(model, batch_inputs, device)
        batch_truth = batch_truth.to(device)

        loss:torch.Tensor = criterion(batch_pred, batch_truth)
        loss_metric.append(loss.item())

        batch_truth = batch_truth.cpu().detach().numpy()
        batch_pred  = batch_pred.cpu().detach().numpy()
        acc = get_weighted_acc(batch_pred, batch_truth, acc_weight)
        acc_metric.append(acc)
        
        if cm_length != 0:
            confusion_matrixs += \
                confusion_matrix(batch_truth, batch_pred, labels=list(range(cm_length)))  # , sample_weight=weight)
        pbar.set_description(f"[VALID] loss: {loss_metric.avg():.5f}, " + \
                             f"Acc: {acc_metric.avg()*100:.3f}%, ")
        
    # with np.printoptions(linewidth=150, formatter={'float': '{:5.03f}'.format, 'int': '{:2} '.format}):
    #     print("pred :", batch_pred)
    #     print("truth:", batch_truth)
    #     print("corct:", np.array(['X', 'O'])[np.uint8(batch_pred==batch_truth)])
        
    if cm_length != 0:
        return loss_metric.avg(), acc_metric.avg(), confusion_matrixs
    else:
        return loss_metric.avg(), acc_metric.avg()


def main(args):

    os.makedirs(args.save_dir, exist_ok=True)
    shutil.copy(__file__, args.save_dir)
    shutil.copy("libs/models.py", args.save_dir)
    shutil.copy("libs/datasets.py", args.save_dir)
    tensorboard = SummaryWriter(args.save_dir)

    my_data = list(range(100))
    my_dataset = MyMapDataset(my_data)
    my_train_dataset, my_valid_dataset = split_datasets(my_dataset)
    my_train_dataLoader = DataLoader(
        my_train_dataset, args.batch_size, shuffle=True,
        pin_memory=True, drop_last=False, num_workers=args.num_workers, 
        # worker_init_fn=None, collate_fn=None,
        # sampler=None, batch_sampler=None, timeout=0,
        # multiprocessing_context=None, generator=None,
        # prefetch_factor=2, persistent_workers=False
    )
    my_valid_dataLoader = DataLoader(
        my_valid_dataset, args.batch_size, shuffle=True,
        pin_memory=True, drop_last=False, num_workers=args.num_workers, 
        # worker_init_fn=None, collate_fn=None,
        # sampler=None, batch_sampler=None, timeout=0,
        # multiprocessing_context=None, generator=None,
        # prefetch_factor=2, persistent_workers=False
    )

    # print("ground_truth_count:", my_train_dataset.ground_truth_count)
    # train_btc_avg      = sum(my_train_dataset.ground_truth_count) / len(my_train_dataset.ground_truth_count)
    # train_weight       = [ (train_btc_avg/btc) for btc in my_train_dataset.ground_truth_count ]
    # train_weight_torch = torch.from_numpy(np.array(train_weight)).float().to(args.device)
    # valid_btc_avg      = sum(my_valid_dataset.ground_truth_count) / len(my_valid_dataset.ground_truth_count)
    # valid_weight       = [ (valid_btc_avg/btc) for btc in my_valid_dataset.ground_truth_count ]

    model        = MyModel()
    criterion    = torch.nn.CrossEntropyLoss()  # weight=train_weight_torch)
    optimizer    = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    best_valid_loss, best_valid_acc = np.inf, 0
    for epoch in range(1, args.epochs+1):
        print(f"{epoch}/{args.epochs}")

        train_results = \
            train_epoch(model, my_train_dataLoader, criterion, optimizer, lr_scheduler, args.device)  # , train_weight)
        valid_results = \
            valid_epoch(model, my_valid_dataLoader, criterion, args.device)  # , valid_weight)
        if args.cm_length != 0:
            train_loss, train_acc, train_cm = train_results
            valid_loss, valid_acc, valid_cm = valid_results
        else:
            train_loss, train_acc = train_results
            valid_loss, valid_acc = valid_results
        
        tensorboard.add_scalar("0_Losses+LR/0_Train",  train_loss,        epoch)
        tensorboard.add_scalar("0_Losses+LR/1_Valid",  valid_loss,        epoch)
        tensorboard.add_scalar("0_Losses+LR/2_LR",     get_lr(optimizer), epoch)
        tensorboard.add_scalar("1_Accuracies/0_Train", train_acc,         epoch)
        tensorboard.add_scalar("1_Accuracies/1_Valid", valid_acc,         epoch)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if args.cm_length != 0:
                plot_confusion_matrix(args.cm_length, train_cm, f"{args.save_dir}/best_valid_loss_train_cm.png", "Train Confusion Matirx at Best Valid Loss")
                plot_confusion_matrix(args.cm_length, valid_cm, f"{args.save_dir}/best_valid_loss_valid_cm.png", "Valid Confusion Matirx at Best Valid Loss")
            torch.save(model, f"{args.save_dir}/best_valid_loss.pt")
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            if args.cm_length != 0:
                plot_confusion_matrix(args.cm_length, train_cm, f"{args.save_dir}/best_valid_acc_train_cm.png", "Train Confusion Matirx at Best Valid Acc")
                plot_confusion_matrix(args.cm_length, valid_cm, f"{args.save_dir}/best_valid_acc_valid_cm.png", "Valid Confusion Matirx at Best Valid Acc")
            torch.save(model, f"{args.save_dir}/best_valid_acc.pt")

    tensorboard.close()
    return



""" Execution """
if __name__ == "__main__":

    DEFAULT_EPOCHS      = 100
    DEFAULT_BATCH_SIZE  = 32
    DEFAULT_NUM_WORKERS = 8
    DEFAULT_DEVICE      = "cuda:0"
    DEFAULT_LR          = 5e-4
    DEFAULT_LR_DECAY    = 0.9992
    DEFAULT_CM_LENGTH   = 0  # Length of Confusion Matrix (0 to disable)
    # DEFAULT_SAVE_DIR    = f"logs/{time.strftime('%Y.%m.%d-%H.%M.%S', time.localtime())}"

    parser = argparse.ArgumentParser()
    parser.add_argument("-e",  "--epochs",        type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("-bs", "--batch-size",    type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("-nw", "--num-workers",   type=int,   default=DEFAULT_NUM_WORKERS)
    parser.add_argument("-d",  "--device",        type=str,   default=DEFAULT_DEVICE)
    parser.add_argument("-lr", "--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("-ld", "--lr-decay",      type=float, default=DEFAULT_LR_DECAY)
    parser.add_argument("-cl", "--cm-length",     type=int,   default=DEFAULT_CM_LENGTH)
    # parser.add_argument("-sd", "--save-dir",      type=str,   default=DEFAULT_SAVE_DIR)

    args = parser.parse_args()
    args.save_dir = f"logs/{time.strftime('%Y.%m.%d-%H.%M.%S', time.localtime())}"
    main(args)