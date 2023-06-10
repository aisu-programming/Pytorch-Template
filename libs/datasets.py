""" Libraries """
import math
import torch
import torch.utils.data
from typing import Any, Iterator, Tuple
from torch.utils.data import Dataset, IterableDataset, Subset



""" Classes """
class MyMapDataset(Dataset):
    def __init__(self) -> None:
        super(MyMapDataset).__init__()
        self.inputs = []
        self.truths = []

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.truths[index]

    def __len__(self) -> int:
        raise len(self.inputs)


class MyIterableDataset(IterableDataset):
    def __init__(self, start: int, end: int) -> None:
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
        
    def __iter__(self) -> Iterator[int]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = self.start
            iter_end = self.end
        else:
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(range(iter_start, iter_end))



""" Functions """
def split_datasets(dataset: Dataset) -> Tuple[Subset, Subset]:
    dataset_length = len(dataset)
    train_dataset_length = int(dataset_length*0.8)
    valid_dataset_length = dataset_length - train_dataset_length
    train_dataset, valid_dataset = \
        torch.utils.data.random_split(
            dataset, [train_dataset_length, valid_dataset_length],
            generator=torch.Generator().manual_seed(0))
    return train_dataset, valid_dataset