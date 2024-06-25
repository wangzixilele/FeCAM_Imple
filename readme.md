# FeCAM
## Introduction
This project is based on the offical implementation of the paper " FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning" (NeurIPS 2023). The official implementation is available at [here](https://github.com/dipamgoswami/FeCAM)
We are the project that replied the code. And try some slight modification to make a experiment.
## Installation
### Requirements
The requirements are listed in the `requirements.txt` file. You can install them by running the following command:
```bash
pip install -r requirements.txt
```
The list is as follows:
-continuum==1.2.7
-numpy==1.24.3
-Pillow==10.3.0
-scikit_learn==1.3.0
-scipy==1.14.0
-timm==1.0.7
-torch==2.2.1
-torchvision==0.17.1
-tqdm==4.65.0

### Data Preparation
#### The basic dataset
The basic dataset is cifar100, imagnet100 and mini-imagnet100.
#### append the dataset
if you want to append the dataset, here is what you should do:
1. Download the dataset and put it in your data folder.
2. You should modify the `data.py` file to add the dataset as a new class. There are two way to get the dataset, one is to use the `torchvision.datasets` and the other is to use the `torchvision.datasets.ImageFolder` in the `download_dataset` function. You can refer to the `download_dataset` function in the `data.py` file.
3. You can modify the `*_trsf` variable in the `data.py` file to add the transform matrix for the new dataset.
4. If you add the new dataset, you should modify the `_get_idata` function in the `DummyDataset` class in the `data_manager.py` file to add the new dataset. You also need to import the new dataset in the `data_manager.py` file.
5. You can apply the dataset by passing the dataset name to the `--dataset` argument in the `main.py` file.

## Usage
### Training
You can train the model by running the following command:
```bash
python main.py --config=exps/FeCAM_{dataset}.json
```
where `{dataset}` is the name of the dataset you want to train on. The configuration file is in the `exps` folder. You can modify the configuration file to change the hyperparameters of the model.
The {dataset} of our project is cifar100, imagnet100 and mini-imagnet100. You can also add the dataset by following the steps in the `Data Preparation` section. There is no need to make sure the name of the dataset is the same as the name of the folder of the dataset. You can pass the name of the dataset to the `--dataset` argument in the `main.py` file.