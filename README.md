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

Besides, you can get our trained model and reproduce the experimental results from the link: [BTLink_saved_models](https://huggingface.co/microsoft/codebert-base).

### Start

You can reproduce the results of within-project link recovery by running the [file](https://github.com/glnmzx888/BTLink/blob/main/WithinCode/allRUN.sh) or reproduce the results of cross-project link recovery by running the [file](https://github.com/glnmzx888/BTLink/blob/main/codeCross/allRUN.sh).

### Result

We present the average performance of BTLink on the within-project link recovery task and the cross-project link recovery task. You can find more detailed results under the results folder.

#### Within-project(%)

| Model       |   Precision    | Recall |    F1     |  MCC   |   AUC    |    ACC    |  PF  |
| ----------- | :-------: | :--------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| BTLink     |   9.64    |   10.21    |   13.98   |   15.93   |   15.09   |   21.08   |   14.32   |
| FRLink |   11.18   |   11.59    |   16.38   |   15.81   |   16.26   |   22.12   |   15.56   |
| DeepLink     |   11.17   |   11.90    |   17.72   |   18.14   |   16.47   |   24.02   |   16.57   |
| hybrid-linker    | **12.16** | **14.90**  | **18.07** | **19.06** | **17.65** | **25.16** | **17.83** |

#### Cross-project(%)

| Model       |   Precision    | Recall |    F1     |  MCC   |   AUC    |    ACC    |  PF  |
| ----------- | :-------: | :--------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| BTLink     |   9.64    |   10.21    |   13.98   |   15.93   |   15.09   |   21.08   |   14.32   |
| FRLink |   11.18   |   11.59    |   16.38   |   15.81   |   16.26   |   22.12   |   15.56   |
| DeepLink     |   11.17   |   11.90    |   17.72   |   18.14   |   16.47   |   24.02   |   16.57   |


# Reference
If you use this code or BTLink, please consider citing us.
<pre><code></code></pre>
