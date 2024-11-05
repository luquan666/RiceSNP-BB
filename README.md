# RiceSNP-BB

## Introduction

```text
RiceSNP-BB: an automated modeling CNN framework for interpretable predicting Bacterial Blight resistance SNPs in rice via integrated sequence and structural representation
```
RiceSNP-BB is a deep learning framework designed for genome-wide interpretable prediction of rice Bacterial Blight (BB) resistance SNPs. It effectively captures informative multimodal DNA sequence features using four distinct encoding methods, leveraging only BB-resistant SNP sites as input. Additionally, it incorporates an encoder layer and automated CNN-based modeling to learn nucleotide distributions based on the context of SNP loci.

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

Please feel free to contact us if you need any help (E-mail: gaoyujia@ahau.edu.cn).
