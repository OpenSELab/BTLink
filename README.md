# BTLink

This repo provides the code for reproducing the experiments in BTLink : Automatic link recovery between issues and commits based on pre-trained BERT model. Specially, we propose a novel BERT-based method (i.e., BTLink), which applies the pre-trained models to automatically recover the issue-commits links. 

### Dataset

We build labeled link datasets from open-source software projects to verify the effectiveness of BLink in the within-project and cross-project contexts. You can get the data through the following link (Google drive)：https://drive.google.com/drive/folders/1mjJrscTS63dXt0fwYlqeifg7P1GzmiPU?usp=sharing

### Dependency

- pip install torch
- pip install transformers
- pip install sklearn 
- pip install nltk


### Relevant Pretrained-Models

BTLink mainly relies on the following pre-trained models as the backbone to obtain the embedding representation of NL-NL (issue text-commit text) pairs and NL-PL (issue text-commit code) pairs and obtain feature vectors to complete subsequent link recovery.

- NL-NL Encoder: [RoBERTa-large](https://huggingface.co/roberta-large)

- NL-PL Encoder: [CodeBERT](https://huggingface.co/microsoft/codebert-base)

Besides, you can get our trained model and reproduce the experimental results from the link: [1](https://huggingface.co/microsoft/codebert-base).
### Quick Start

```python

```
