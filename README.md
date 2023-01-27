# Multi-Spectral Self-Supervised Learning with Viewmaker Networks

## 0) Background

Multi-spectral satellite images can capture rich information in images by measuring light beyond the visible spectrum. However, self-supervised learning is challenging in this domain due to there being fewer pre-existing data augmentations. [Viewmaker networks](https://github.com/alextamkin/viewmaker) learn to produce appropriate augmentations for general data, enabling contrastive learning applications to many domains and modalities. In this project, we apply Viewmaker networks to four different multi-spectral imaging problems to demonstrate that these domain-agnostic learning methods can provide valuable performance gains over existing domain-specific deep learning methods for multi-spectral satellite images.

## 1) Install Dependencies

We used the following PyTorch libraries for CUDA 10.1; you may have to adapt for your own CUDA version:

```console
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

Install other dependencies:
```console
conda install scipy
pip install -r requirements.txt
```

## 2) Running experiments

Start by running
```console
source init_env.sh
```

Run experiments for the different datasets as follows:

```console
scripts/run.py config/eurosat/pretrain_eurosat_simclr_L1_forced.json --gpu-device 0
```

This command runs Viewmaker pretraining on the EuroSAT multi-spectral satellite dataset using GPU 0. The `config` directory holds configuration files for the different experiments, specifying the hyperparameters for each experiment. The first field in every config file is `exp_base`, which specifies the base directory to save experiment outputs. You should change this for your own setup and also update the dataset paths in `src/datasets/root_paths.py`. The experiments include standard Viewmaker pretraining, Divmaker pretraining, default views pretraining, and the associated linear protocol for transfer training.

Training curves and other metrics are logged using [wandb.ai](wandb.ai).