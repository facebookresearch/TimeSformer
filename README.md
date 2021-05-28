# TimeSformer

This is an official pytorch implementation of our ICML 2021 paper [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/pdf/2102.05095.pdf). In this repository, we provide PyTorch code for training and testing our proposed TimeSformer model. TimeSformer provides an efficient video classification framework that achieves state-of-the-art results on several video action recognition benchmarks such as Kinetics-400.

If you find TimeSformer useful in your research, please use the following BibTeX entry for citation.

```BibTeX
@inproceedings{gberta_2021_ICML,
    author  = {Gedas Bertasius and Heng Wang and Lorenzo Torresani},
    title = {Is Space-Time Attention All You Need for Video Understanding?},
    booktitle   = {Proceedings of the International Conference on Machine Learning (ICML)}, 
    month = {July},
    year = {2021}
}
```

# Model Zoo

We provide TimeSformer models pretrained on Kinetics-400 (K400), Kinetics-600 (K600), Something-Something-V2 (SSv2), and HowTo100M datasets.

| name | dataset | # of frames | spatial crop | acc@1 | acc@5 | url |
| --- | --- | --- | --- | --- | --- | --- |
| TimeSformer | K400 | 8 | 224 | 77.9 | 93.2 | [model](https://www.dropbox.com/s/g5t24we9gl5yk88/TimeSformer_divST_8x32_224_K400.pyth?dl=0) |
| TimeSformer-HR | K400 | 16 | 448 | 79.6 | 94.0 | [model](https://www.dropbox.com/s/6f0x172lpqy3oxt/TimeSformer_divST_16x16_448_K400.pyth?dl=0) |
| TimeSformer-L | K400 | 96 | 224 | 80.6 | 94.7 | [model](https://www.dropbox.com/s/r1iuxahif3sgimo/TimeSformer_divST_96x4_224_K400.pyth?dl=0) |

| name | dataset | # of frames | spatial crop | acc@1 | acc@5 | url |
| --- | --- | --- | --- | --- | --- | --- |
| TimeSformer | K600 | 8 | 224 | 79.1 | 94.4 | [model](https://www.dropbox.com/s/4h2qt41m2z3aqrb/TimeSformer_divST_8x32_224_K600.pyth?dl=0) |
| TimeSformer-HR | K600 | 16 | 448 | 81.8 | 95.8 | [model](https://www.dropbox.com/s/ft1e92g2vhvxecv/TimeSformer_divST_16x16_448_K600.pyth?dl=0) |
| TimeSformer-L | K600 | 96 | 224 | 82.2 | 95.6 | [model](https://www.dropbox.com/s/857rx6xeclxfhdg/TimeSformer_divST_96x4_224_K600.pyth?dl=0) |

| name | dataset | # of frames | spatial crop | acc@1 | acc@5 | url |
| --- | --- | --- | --- | --- | --- | --- |
| TimeSformer | SSv2 | 8 | 224 | 59.1 | 85.6 | [model](https://www.dropbox.com/s/tybhuml57y24wpm/TimeSformer_divST_8_224_SSv2.pyth?dl=0) |
| TimeSformer-HR | SSv2 | 16 | 448 | 61.8 | 86.9 | [model](https://www.dropbox.com/s/9t68uzk8w2fpfnv/TimeSformer_divST_16_448_SSv2.pyth?dl=0) |
| TimeSformer-L | SSv2 | 64 | 224 | 62.0 | 87.5 | [model](https://www.dropbox.com/s/3f1rm2al8mhprwa/TimeSformer_divST_64_224_SSv2.pyth?dl=0) |

| name | dataset | # of frames | spatial crop | single clip coverage | acc@1 | url |
| --- | --- | --- | --- | --- | --- | --- |
| TimeSformer | HowTo100M | 8 | 224 | 8.5s | 56.8 | [model](https://www.dropbox.com/s/9v8hcm88b9tc6ff/TimeSformer_divST_8x32_224_HowTo100M.pyth?dl=0) |
| TimeSformer | HowTo100M | 32 | 224 | 34.1s | 61.2 | [model](https://www.dropbox.com/s/4roflx4q1gscu85/TimeSformer_divST_32x32_224_HowTo100M.pyth?dl=0) |
| TimeSformer | HowTo100M | 64 | 448 | 68.3s | 62.2 | [model](https://www.dropbox.com/s/15bvqltl1j5vyp3/TimeSformer_divST_64x32_224_HowTo100M.pyth?dl=0) |
| TimeSformer | HowTo100M | 96 | 224 | 102.4s | 62.6 | [model](https://www.dropbox.com/s/t2mzgahnfhgakma/TimeSformer_divST_96x32_224_HowTo100M.pyth?dl=0) |

We note that these models were re-trained using a slightly different implementation than the one used in the paper. Therefore, there might be a small difference in performance compared to the results reported in the paper.

You can load the pretrained models as follows:

```python
import torch
from timesformer.models.vit import TimeSformer

model = TimeSformer(img_size=224, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model='/path/to/pretrained/model.pyth')

dummy_video = torch.randn(2, 3, 8, 224, 224) # (batch x channels x frames x height x width)

pred = model(dummy_video,) # (2, 400)
```

# Installation

First, create a conda virtual environment and activate it:
```
conda create -n timesformer python=3.7 -y
source activate timesformer
```

Then, install the following packages:

- torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
- simplejson: `pip install simplejson`
- einops: `pip install einops`
- timm: `pip install timm`
- PyAV: `conda install av -c conda-forge`
- psutil: `pip install psutil`
- scikit-learn: `pip install scikit-learn`
- OpenCV: `pip install opencv-python`
- tensorboard: `pip install tensorboard`

Lastly, build the TimeSformer codebase by running:
```
git clone https://github.com/facebookresearch/TimeSformer
cd TimeSformer
python setup.py build develop
```

# Usage

## Dataset Preparation

Please use the dataset preparation instructions provided in [DATASET.md](timesformer/datasets/DATASET.md).

## Training the Default TimeSformer

Training the default TimeSformer that uses divided space-time attention, and operates on 8-frame clips cropped at 224x224 spatial resolution, can be done using the following command:

```
python tools/run_net.py \
  --cfg configs/Kinetics/TimeSformer_divST_8x32_224.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
```
You may need to pass location of your dataset in the command line by adding `DATA.PATH_TO_DATA_DIR path_to_your_dataset`, or you can simply add

```
DATA:
  PATH_TO_DATA_DIR: path_to_your_dataset
```

To the yaml configs file, then you do not need to pass it to the command line every time.

## Using a Different Number of GPUs

If you want to use a smaller number of GPUs, you need to modify .yaml configuration files in [`configs/`](configs/). Specifically, you need to modify the NUM_GPUS, TRAIN.BATCH_SIZE, TEST.BATCH_SIZE, DATA_LOADER.NUM_WORKERS entries in each configuration file. The BATCH_SIZE entry should be the same or higher as the NUM_GPUS entry. In [`configs/Kinetics/TimeSformer_divST_8x32_224_4gpus.yaml`](configs/Kinetics/TimeSformer_divST_8x32_224_4gpus.yaml), we provide a sample configuration file for a 4 GPU setup.


## Using Different Self-Attention Schemes

If you want to experiment with different space-time self-attention schemes, e.g., space-only or joint space-time attention, use the following commands:


```
python tools/run_net.py \
  --cfg configs/Kinetics/TimeSformer_spaceOnly_8x32_224.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
```

and

```
python tools/run_net.py \
  --cfg configs/Kinetics/TimeSformer_jointST_8x32_224.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
```

## Training Different TimeSformer Variants

If you want to train more powerful TimeSformer variants, e.g., TimeSformer-HR (operating on 16-frame clips sampled at 448x448 spatial resolution), and TimeSformer-L (operating on 96-frame clips sampled at 224x224 spatial resolution), use the following commands:

```
python tools/run_net.py \
  --cfg configs/Kinetics/TimeSformer_divST_16x16_448.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
```

and

```
python tools/run_net.py \
  --cfg configs/Kinetics/TimeSformer_divST_96x4_224.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
```

Note that for these models you will need a set of GPUs with ~32GB of memory.

## Inference

Use `TRAIN.ENABLE` and `TEST.ENABLE` to control whether training or testing is required for a given run. When testing, you also have to provide the path to the checkpoint model via TEST.CHECKPOINT_FILE_PATH.
```
python tools/run_net.py \
  --cfg configs/Kinetics/TimeSformer_divST_8x32_224_TEST.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  TEST.CHECKPOINT_FILE_PATH path_to_your_checkpoint \
  TRAIN.ENABLE False \
```

## Single-Node Training via Slurm

To train TimeSformer via Slurm, please check out our single node Slurm training script [`slurm_scripts/run_single_node_job.sh`](slurm_scripts/run_single_node_job.sh).


## Multi-Node Training via Submitit

Distributed training is available via Slurm and submitit

```
pip install submitit
```

To train TimeSformer model on Kinetics using 4 nodes with 8 gpus each use the following command:
```
python tools/submit.py --cfg configs/Kinetics/TimeSformer_divST_8x32_224.yaml --job_dir  /your/job/dir/${JOB_NAME}/ --num_shards 4 --name ${JOB_NAME} --use_volta32
```

We provide a script for launching slurm jobs in [`slurm_scripts/run_multi_node_job.sh`](slurm_scripts/run_multi_node_job.sh).

## Finetuning

To finetune from an existing PyTorch checkpoint add the following line in the command line, or you can also add it in the YAML config:

```
TRAIN.CHECKPOINT_FILE_PATH path_to_your_PyTorch_checkpoint
TRAIN.FINETUNE True
```

## HowTo100M Dataset Split

If you want to experiment with the long-term video modeling task on HowTo100M, please download the train/test split files from [here](https://www.dropbox.com/sh/ttvsxwqypijjuda/AACmJx1CnddW6cVBoc21eSuva?dl=0).


# Environment

The code was developed using python 3.7 on Ubuntu 20.04. For training, we used four GPU compute nodes each node containing 8 Tesla V100 GPUs (32 GPUs in total). Other platforms or GPU cards have not been fully tested.

# License

The majority of this work is licensed under [CC-NC 4.0 International license](LICENSE). However portions of the project are available under separate license terms: [SlowFast](https://github.com/facebookresearch/SlowFast) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) are licensed under the Apache 2.0 license.

# Contributing

We actively welcome your pull requests. Please see [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for more info.

# Acknowledgements

TimeSformer is built on top of [PySlowFast](https://github.com/facebookresearch/SlowFast) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) by [Ross Wightman](https://github.com/rwightman). We thank the authors for releasing their code. If you use our model, please consider citing these works as well:

```BibTeX
@misc{fan2020pyslowfast,
  author =       {Haoqi Fan and Yanghao Li and Bo Xiong and Wan-Yen Lo and
                  Christoph Feichtenhofer},
  title =        {PySlowFast},
  howpublished = {\url{https://github.com/facebookresearch/slowfast}},
  year =         {2020}
}
```

```BibTeX
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```
