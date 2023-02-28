# Quick start

This repo provides the code for reproducing the experiments of BTLink in within-project link recovery. 

## Files

**run_pre.py**: The script where the main program is located, including the main function and functions related to training, verification and testing.

**utils.py**: Provide necessary auxiliary functions for the operation of the main program, including *set_seed(seed)*, *MySubSampler(df, x)*, *getargs()*, *convert_examples_to_features*. You can directly assign values to hyperparameters in the getargs() function, or specify the dataset folder, model path, and prediction result save path, etc., or pass in parameters at runtime.

**model.py**: This script describes the model network architecture, namely BTLink.

**preprocessor.py**: This script provides the functions for data cleaning and text preprocessing. 

**pro.csv**: In our data preprocessing process, operations such as *Removing issue identifiers* are involved. Here we give each project and the corresponding JIRA key. It is worth mentioning that for projects on Github, **we use # as the key**, and for other projects that use JIRA as the issue tracking system, we recommend that you refer to the [JIRA official website](https://issues.apache.org/jira/secure/BrowseProjects.jspa?selectedCategory=all&selectedProjectType=all).

**allRun.sh**: We have saved some fine-tuning samples, you can run the script directly in Idea after setting parameters such as the folder path in the getargs() function in *utils.py*.

## Fine-tuning and Evaluation

Our experiments were all conducted on the same machine, which includes two GPU (NVIDIA 3090) and 64GB of memory.

```shell
python run_pre.py --do_train --do_test --pro $pro_name --key $issue_identifier --model_name_or_path $pretrained_model --text_model_path $pretrained_model_for_roberta --code_model_path $pretrained_model_for_codeBERT --data_dir $data_dir --output_dir $output_dir --result_dir $result_dir --max_seq_length 512 --train_batch_size 8 --eval_batch_size 16 --learning_rate 1r-5 --weight_decay 0.0 --seed 42
```

Finally, you can get the experimental results in the result_dir folder.
