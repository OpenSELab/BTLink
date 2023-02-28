# Quick start

This repo provides the code for reproducing the experiments of BTLink in cross-project link recovery. You must have the corresponding trained model in the within-project link recovery task to complete cross-project link recovery

## Files

**run_pre.py**: The script where the main program is located, including the main function and functions related to testing.

**utils.py**: Provide necessary auxiliary functions for the operation of the main program, including *set_seed(seed)*, *MySubSampler(df, x)*, *getargs()*, *convert_examples_to_features*. You can directly assign values to hyperparameters in the getargs() function, or specify the dataset folder, model path, and prediction result save path, etc., or pass in parameters at runtime.

**preprocessor.py**: This script provides the functions for data cleaning and text preprocessing. 

**allRun.sh**: We have saved some fine-tuning samples, you can run the script directly in Idea after setting parameters such as the folder path in the getargs() function in *utils.py*.

## Fine-tuning and Evaluation

Our experiments were all conducted on the same machine, which includes two GPU (NVIDIA 3090) and 64GB of memory.

```shell
python run_pre.py --do_test --trained_pro $pro_name_of_train_project --pro $pro_name_of_test_project --key $issue_identifier_of_test_project --model_name_or_path $pretrained_model --data_dir $data_dir --output_dir $model_saved_path_by_within_link_recovery --result_dir $result_dir --max_seq_length 512 --eval_batch_size 16 --learning_rate 1r-5 --weight_decay 0.0 --seed 42
```

Finally, you can get the experimental results in the result_dir folder.
