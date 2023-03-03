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

Besides, you can get our trained model and reproduce the experimental results from the link: [BTLink_saved_models](https://zenodo.org/record/7015037#.ZAHeCnZByUkhttps:%2F%2Fhome.firefoxchina.cn).

### Start

First, you shold download the dataset from our [link](https://drive.google.com/drive/folders/1mjJrscTS63dXt0fwYlqeifg7P1GzmiPU?usp=sharing). 

You can reproduce the results of within-project link recovery by running the [file](https://github.com/glnmzx888/BTLink/blob/main/WithinCode/allRUN.sh) or reproduce the results of cross-project link recovery by running the [file](https://github.com/glnmzx888/BTLink/blob/main/CrossCode/allRUN.sh). 

Please follow the instructions to complete the reproduction：[within-project link recovery](https://github.com/glnmzx888/BTLink/tree/main/WithinCode), [Cross-project link recovery](https://github.com/glnmzx888/BTLink/tree/main/CrossCode).

### Result

We present the average performance of BTLink on the within-project link recovery task and the cross-project link recovery task. You can find more detailed results under the results folder.

#### Within-project(%)

| Model       |    F1   | Recall |    Precision     |  MCC   |   AUC    |    ACC    |  PF  |
| ----------- | :-------: | :--------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| BTLink     |   **80.03**    |   **84.34**    |   **78.81**   |   **77.53**   |   **90.22**   |   **94.76**   |   **3.91**   |
| FRLink |   55.89   |   49.39    |   66.58   |   50.48   |   72.28   |   89.43   |   4.82   |
| DeepLink     |   38.50   |   66.47    |   29.43   |   27.95   |   70.21   |   73.22   |   26.05   |
| hybrid-linker    | 38.04 | 30.63  | 72.03 | 36.46 | 62.75 | 88.02 | 5.12 |

#### Cross-project(%)

| Model       |   F1    | Recall |     Precision    |  MCC   |   AUC    |    ACC    |  PF  |
| ----------- | :-------: | :--------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| BTLink     |   **62.19**    |   **71.02**    |   62.72   |   **59.06**   |  **81.65**   |   **89.75**   |   **7.73**   |
| FRLink |   49.32   |   48.80    |   **62.73**   |   45.83   |   71.49   |   87.92   |   5.81   |
| DeepLink     |   13.92   |   53.12    |   17.95   |   6.58   |   54.89   |   56.30   |   43.35   |


# Reference
If you use this code or BTLink, please consider citing us.
<pre><code></code></pre>
