# Import required libraries
import torch
from torch.utils.data import Dataset, DataLoader
from nats_bench import create
import os
import pickle # Use for faster loading of benchmark file
import argparse
import logging
import numpy as np
from scipy.ndimage import convolve1d
from utils import get_lds_kernel_window # for LDS

print = logging.info

# Define new class for dataset
class NATSBenchDataset(Dataset):
    # Initialize dataset with NATSBench vars
    """
    Params:
    - benchmark_path: path to the benchmark file
    - search_space: 'tss' indicates the topology search space, while 'sss' represents the size search space
    - dataset_name: defaults to the 'Cifar 10' dataset, but inputs can also be 'cifar100' and 'ImageNet16-120' (see more info in README)
    - mode: which sample to test ('train', 'test', or 'valid')
    - fast_mode: boolean var – set to 'True' to use benchmark file and 'False' to use archive
    - verbose: boolean var for verbosity in output
    """
    def __init__(self, benchmark_path, search_space='tss', dataset_name='cifar10', mode='test', fast_mode=False, verbose=False, reweight='none', lds=False, lds_ks=5, lds_sigma=2):
        self.benchmark_path = benchmark_path
        self.dataset_name = dataset_name
        self.mode = mode # 'train' or 'test' based on your debug sample

        # Load from benchmark file's cache path if it exists
        cached_benchmark_path = benchmark_path.replace('.pickle.pbz2', '.raw_cache.pkl')
        if os.path.exists(cached_benchmark_path):
            print("Loading benchmark file from cache")
            with open(cached_benchmark_path, 'rb') as f:
                self.api = pickle.load(f)
        else:
            print("No cache found. Loading benchmark file directly")
            # Create benchmark instance 
            self.api = create(benchmark_path, search_space, fast_mode, verbose)
            # Save to cache for next time
            with open(cached_benchmark_path, 'wb') as w:
                pickle.dump(self.api, w)

        self.length = len(self.api)
        self.weights = self._prepare_weights(reweight=reweight, lds=lds, 
                                            lds_ks=lds_ks, lds_sigma=lds_sigma)


    # Return length of the dataset
    def __len__(self):
        return self.length
    
    # Return item in dataset
    def __getitem__(self, index):
        # Get architecture string of index
        architecture_str = self.api.arch(index)

        # Query the loss/accuracy/time for the index-th candidate architecture on the dataset
        # info is a dict, where you can easily figure out the meaning by key
        info = self.api.get_more_info(index, self.dataset_name, hp='200')

        # Optional: Query the flops, params, latency
        # cost_info = self.api.get_cost_info(index, self.dataset_name)

        if self.mode == 'test':
            accuracy = info['test-accuracy'] / 100
        else:
            accuracy = info['train-accuracy'] /100

        # Return Weight if it exists, else 1.0
        weight = np.float32(self.weights[index]) if self.weights is not None else np.float32(1.0)

        return {'x': architecture_str, 'y': torch.tensor(accuracy, dtype=torch.float32), 'w': weight}
    
    # Prepare weights (i.e. if we want to involve LDS in our model)
    def _prepare_weights(self, reweight, max_target=101, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert reweight != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"
        
        value_dict = {x: 0 for x in range(max_target)}
        
        labels = np.array([self.api.get_more_info(i, self.dataset_name, hp='200')['test-accuracy' if self.mode == 'test' else 'train-accuracy'] for i in range(self.length)])
 
        for label in labels:
            value_dict[min(max_target - 1, int(label))] += 1
        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
        num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
        if not len(num_per_label) or reweight == 'none':
            return None
        print(f"Using re-weighting: [{reweight.upper()}]")

        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        return weights
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark_file_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--mode', type=str, default='test', choices=['test','train'])
    parser.add_argument('--reweight', type=str, default='none', choices=['none', 'inverse', 'sqrt_inv'])
    parser.add_argument('--lds', action='store_true')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    ds = NATSBenchDataset(args.benchmark_file_path, dataset_name=args.dataset, mode=args.mode, reweight=args.reweight, lds=args.lds)
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    
    batch = next(iter(loader))
    print(f"\nArchitecture: {batch['x'][0]}")
    print(f"Accuracy:     {batch['y'][0]:.4f}")
    print(f"LDS Weight:   {batch['w'][0]:.4f}")

if __name__ == "__main__":
    main()