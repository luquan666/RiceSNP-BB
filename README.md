# RiceSNP-BB

## Introduction

```text
RiceSNP-ABST: A Deep Learning Approach to Identify Abiotic Stress-Associated Single Nucleotide Polymorphisms in Rice
```
In this paper, a model called RiceSNP-ABST is proposed for predicting ABST-SNPs in rice, utilizing a novel strat-egy for constructing negative samples. Firstly, six benchmark datasets were generated using three methods for con-structing negative samples across two lengths of DNA sequences. Secondly, four feature encoding methods were proposed based on DNA sequence fragments, followed by feature selection. Finally, Convolutional neural networks (CNNs) with residual connections were used to determine whether the sequences were ABST-SNPs in rice. 

## Data

#### Benchmark datasets
```shell
cd ./Feature/data/Benchmark datasets
```
“Random”, “Common”, and “Rearrange” represent three methods for constructing negative samples，“41_nt” and “101_nt” denote sequence lengths.
#### Independent dataset
```shell
cd ./Feature/data/Independent dataset
```
#### Rape and Sorghum datasets
```shell
cd ./Feature/data/Rape and Sorghum datasets
```
## Feature represention and selection

`Feature/Onehot.py`,`Feature/DNA2vec.py`,`Feature/DNABERT.py`,and `Feature/DNAshape.py` are four feature extraction methods. <br>
`Feature/Feature selection.py` is for feature selection.

## Model training and testing

`Model/train.py` is for model training.<br>
`Model/test.py` is for model testing.

## Setup environment
### create and activate virtual python environment
#### For Feature code:<br>
```shell
conda create -n Feature python=3.8
conda activate Feature
pip install Featurerequirements.txt
git clone https://github.com/JinsenLi/deepDNAshape
cd deepDNAshape
pip install .
```

#### For Model code:<br>
```shell
conda create -n Model python=3.7
conda activate Model
pip install Modelrequirements.txt
#if you don't have CUDA-enabled GPU, or on MacOS, replace tensorflow-gpu=1.15.0 with tensorflow=1.15.0
#Additionally, the model code needs to be run on a 1080 Ti graphics card.
```


## Contact

Please feel free to contact us if you need any help (E-mail: quanlu@stu.ahau.edu.cn).
