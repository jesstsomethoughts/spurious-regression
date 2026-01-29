## NATS-BENCH Dataset
This dataset is pulled from the [NATS-BENCH](https://github.com/D-X-Y/NATS-Bench?tab=readme-ov-file) open-source repository.

## Requirements
1. First install the NATS Bench Python library:
```
pip install nats_bench
```
or if you have pip3
```
pip3 install nats_bench
```

2. Then download the benchmark file:
[NATS-tss-v1_0-3ffb9.pickle.pbz2](https://drive.google.com/file/d/1vzyK0UVH2D3fTpa1_dSWnp1gvGpAxRul/view?usp=sharing)

## How to Use
Run the dataloader directly or call it via another module.

For example:
```
python3 nats-bench-dataset/dataloader.py --benchmark_file_path=[file path] --dataset_name='cifar10' --mode 'test'
```
The dataloader should work with the CIFAR-10 (`cifar10`), CIFAR-100 (`cifar100`), ImageNet16-120 (`ImageNet16-120`) datasets.

If you would like to enable LDS, pass in the arguments when calling the dataloader file.
For example: 
```
python3 nats-bench-dataset/dataloader.py --benchmark_file_path=[file path] --dataset_name='cifar10' --mode 'test' --reweight 'sqrt_inv' --lds True
```