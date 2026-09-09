from src.data.intent_decouple_dataset import FocusDataset, GlobalDataset, IntentDecoupleDataset, LocalDataset
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import Dataset
import bisect
import torch
import random


import random
from torch.utils.data import BatchSampler

class MixTrainBucketBatchSampler(BatchSampler):
    """
    A batch sampler that supports batching with buckets, across multiple datasets,
    including CombinedDataset or single dataset.
    Moreover supports mix training sampling (global, local, and focus datasets).
    """

    def __init__(self, dataset, batch_size: int, drop_last: bool = False):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer, but got {batch_size}")
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last should be a boolean value, but got {drop_last}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle_within_bucket = True
        self.shuffle_batches = True
        self.seed = 628
        self.epoch = 0
        # 判断数据集是否为 CombinedDataset
        if isinstance(self.dataset, CombinedDataset):
            self.is_combined = True
        else:
            self.is_combined = False

        # Group indices by bucket
        if isinstance(self.dataset,CombinedDataset):
            # 处理 CombinedDataset
            self.bucket_indices = [[] for _ in range(len(self.dataset.buckets))]
            for idx in range(len(self.dataset)):
                ds_idx = bisect.bisect_right(self.dataset.cum_sizes, idx)
                if ds_idx == 0:
                    sample_idx = idx
                else:
                    sample_idx = idx - self.dataset.cum_sizes[ds_idx - 1]
                sample = self.dataset.datasets[ds_idx].samples[sample_idx]
                bucket_idx = sample['bucket_idx']
                self.bucket_indices[bucket_idx].append(idx)
        elif isinstance(self.dataset,MixTrainCombinedDataset):
            self.bucket_indices = [
                [[] for _ in range(len(self.dataset.buckets))] for _ in range(len(self.dataset.types))
            ]
            for idx in range(len(self.dataset)):
                ds_idx = bisect.bisect_right(self.dataset.cum_sizes, idx)
                if ds_idx == 0:
                    sample_idx = idx
                else:
                    sample_idx = idx - self.dataset.cum_sizes[ds_idx - 1]
                sample = self.dataset.datasets[ds_idx].samples[sample_idx]
                bucket_idx = sample['bucket_idx']

                type_idx = bisect.bisect_right(self.dataset.type_cum_sizes, idx)
                self.bucket_indices[type_idx][bucket_idx].append(idx)
        else:
            # 处理普通 Dataset
            self.bucket_indices = [[] for _ in range(len(self.dataset.buckets))]
            for idx, sample in enumerate(self.dataset.samples):
                bucket_idx = sample['bucket_idx']
                self.bucket_indices[bucket_idx].append(idx)

        self.batches = []
        self.sampler_len = 0
        for type_idx in range(len(self.dataset.types)):
            for indices_in_bucket in self.bucket_indices[type_idx]:
                random.shuffle(indices_in_bucket)
                for i in range(0, len(indices_in_bucket), self.batch_size):
                    batch = indices_in_bucket[i: i + self.batch_size]
                    if len(batch) < self.batch_size and self.drop_last:
                        continue
                    self.batches.append(batch)
                    self.sampler_len += 1


    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # Shuffle the order of the batches each epoch
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle_batches:
            rng.shuffle(self.batches)

        for batch in self.batches:
            yield batch

    def __len__(self):
        return self.sampler_len

#


class MixTrainCombinedDataset(Dataset):
    def __init__(self, opt):
        self.both_datasets = []
        self.global_datasets = []
        self.local_datasets = []
        self.buckets = None
        if 'buckets' in opt.keys():
            self.buckets = opt['buckets']
        for dataset in opt['datasets']['focus']:
            dataset_name, dataset_opt = next(iter(dataset.items()))   
            dataset_opt['buckets'] = self.buckets
            self.both_datasets.append(load_dataset(dataset_opt))
        for dataset in opt['datasets'].get('global',[]):
            dataset_name, dataset_opt = next(iter(dataset.items()))
            print(dataset_opt)   
            dataset_opt['buckets'] = self.buckets
            self.global_datasets.append(load_dataset(dataset_opt))
        for dataset in opt['datasets'].get('local',[]):
            dataset_name, dataset_opt = next(iter(dataset.items()))   
            dataset_opt['buckets'] = self.buckets
            self.local_datasets.append(load_dataset(dataset_opt))
        self.cum_sizes = []
        self.type_cum_sizes = []
        s = 0

        for ds in self.both_datasets:
            s += len(ds)
            self.cum_sizes.append(s)
        
        self.type_cum_sizes.append(s)
        
        for ds in self.global_datasets:
            s += len(ds)
            self.cum_sizes.append(s)

        self.type_cum_sizes.append(s)

        for ds in self.local_datasets:
            s += len(ds)
            self.cum_sizes.append(s)
        self.type_cum_sizes.append(s)

        self.datasets = self.both_datasets + self.global_datasets + self.local_datasets
        print("type-CUM-SIZE:",self.type_cum_sizes)
        print("type-CUM-SIZE:",self.type_cum_sizes)
        print("type-CUM-SIZE:",self.type_cum_sizes)
        self.types = ['focus','global','local']
    def __len__(self):
        return self.cum_sizes[-1]

    def __getitem__(self, index):
        # 定位到第几个子 dataset
        ds_idx = bisect.bisect_right(self.cum_sizes, index)
        type_idx = bisect.bisect_right(self.type_cum_sizes, index)
        if ds_idx == 0:
            sample_idx = index
        else:
            sample_idx = index - self.cum_sizes[ds_idx - 1]

        sample = self.datasets[ds_idx][sample_idx]

        # 可选：给你一个 dataset_idx 方便调试
        if isinstance(sample, dict):
            sample = dict(sample)
            sample["dataset_idx"] = ds_idx
            sample['type'] = self.types[type_idx]

        return sample
 
class CombinedDataset(Dataset):
    def __init__(self, opt):
        self.datasets = []
        self.buckets = None
        if 'buckets' in opt.keys():
            self.buckets = opt['buckets']
        for dataset in opt['datasets']:
            dataset_name, dataset_opt = next(iter(dataset.items()))   
            dataset_opt['buckets'] = self.buckets
            self.datasets.append(load_dataset(dataset_opt))

        self.cum_sizes = []
        s = 0
        for ds in self.datasets:
            s += len(ds)
            self.cum_sizes.append(s)

    def __len__(self):
        return self.cum_sizes[-1]

    def __getitem__(self, index):
        # 定位到第几个子 dataset
        ds_idx = bisect.bisect_right(self.cum_sizes, index)
        if ds_idx == 0:
            sample_idx = index
        else:
            sample_idx = index - self.cum_sizes[ds_idx - 1]

        sample = self.datasets[ds_idx][sample_idx]

        # 可选：给你一个 dataset_idx 方便调试
        if isinstance(sample, dict):
            sample = dict(sample)
            sample["dataset_idx"] = ds_idx

        return sample


DATASET_REGISTRY = {
    "IntentDecoupleDataset":IntentDecoupleDataset,
    "GlobalDataset": GlobalDataset,
    "FocusDataset": FocusDataset,
    "LocalDataset": LocalDataset,
    # Combined
    "CombinedDataset":CombinedDataset,
    "MixTrainCombinedDataset":MixTrainCombinedDataset,
}



def load_dataset(opt):
    dataset_name = opt['name']
    dataset_cls = DATASET_REGISTRY[dataset_name]
    dataset = dataset_cls(opt)
    print(f"Load Dataset: {dataset_name}")
    return dataset
