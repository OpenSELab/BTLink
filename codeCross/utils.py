from __future__ import absolute_import, division, print_function
from preprocessor import textProcess
import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from imblearn.under_sampling import RandomUnderSampler


epoch = 10
eval_batch_size = 16
train_batch_size = 8
flag_train = False
flag_test = True
seed_num = 42
key = ''
pro = ''
train_pro = pro


def set_seed(seed=42):
    # random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label


def MySubSampler(df, x):
    X, y = df[['Issue_Text', 'Commit_Text', 'Commit_Code']], df['label']
    rus = RandomUnderSampler(random_state=x)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    df = pd.concat([X_resampled, y_resampled], axis=1)
    return df.sample(frac=1)


def getargs():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default='../dataset', type=str,
                        help="data_dir")

    # NOTE: cross-project link recovery must rely on the models saved by in-project BTLink.
    parser.add_argument("--output_dir", default='../code/saved_models_f1', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--result_dir", default='../results/crossResult', type=str,
                        help="The output directory where the result files will be written.")

    # Other parameters
    parser.add_argument("--text_model_path", default='../roberta-large', type=str,
                        help="The NL-NL model checkpoint for weights initialization.")
    parser.add_argument("--code_model_path", default='../CodeBERT', type=str,
                        help="The NL-PL model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_name", default="../roberta-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", default=flag_train, type=bool,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, type=bool,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=flag_test, type=bool,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=train_batch_size, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=eval_batch_size, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=seed_num,
                        help="random seed for initialization")
    parser.add_argument('--num_train_epochs', type=int, default=epoch,
                        help="num_train_epochs")
    parser.add_argument("--key", default=key, type=str,
                        help="Key of the project.")
    parser.add_argument("--pro", default=pro, type=str,
                        help="The used project.")
    parser.add_argument("--trained_pro", default=train_pro, type=str,
                        help="The trained project model.")
    parser.add_argument("--num_class", default=2, type=int,
                        help="The number of classes.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    return args


def convert_examples_to_features(row, tokenizer, commitType, args):
    issue_text = textProcess(row['Issue_Text'], args.key)
    commit_text = textProcess(row[commitType], args.key)
    issue_token = tokenizer.tokenize(issue_text)
    commit_token = tokenizer.tokenize(commit_text)
    if len(issue_token) + len(commit_token) > args.max_seq_length - 3:
        if len(issue_token) > (args.max_seq_length - 3) / 2 and len(commit_token) > (args.max_seq_length - 3) / 2:
            issue_token = issue_token[:int((args.max_seq_length - 3) / 2)]
            commit_token = commit_token[:args.max_seq_length - 3 - len(issue_token)]
        elif len(issue_token) > (args.max_seq_length - 3) / 2:
            issue_token = issue_token[:args.max_seq_length - 3 - len(commit_token)]
        elif len(commit_token) > (args.max_seq_length - 3) / 2:
            commit_token = commit_token[:args.max_seq_length - 3 - len(issue_token)]
    combined_token = [tokenizer.cls_token] + issue_token + [tokenizer.sep_token] + commit_token + [tokenizer.sep_token]
    combined_ids = tokenizer.convert_tokens_to_ids(combined_token)
    if len(combined_ids) < args.max_seq_length:
        padding_length = args.max_seq_length - len(combined_ids)
        combined_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(combined_token, combined_ids, row['label'])


def convert_examples_to_features1(row, tokenizer, args):
    issue_text = textProcess(row['Issue_Text'], args.key)
    commit_text = textProcess(row['Commit_Code'], args.key)
    issue_token = tokenizer.tokenize(issue_text)
    commit_token = tokenizer.tokenize(commit_text)
    if len(issue_token) + len(commit_token) > args.max_seq_length - 3:
        if len(issue_token) > (args.max_seq_length - 3) / 2 and len(commit_token) > (args.max_seq_length - 3) / 2:
            issue_token = \
                issue_token[:int((args.max_seq_length - 3) / 2)]
            commit_length = (args.max_seq_length - 3) - len(issue_token)
            commit_token = \
                commit_token[:int(commit_length/2)] + \
                commit_token[
                len(commit_token) - (commit_length - int(commit_length / 2)):]
            # assert len(issue_token) + len(commit_token) + 3 == 512, 'Error1'
        elif len(issue_token) > (args.max_seq_length - 3) / 2:
            issue_token = issue_token[:args.max_seq_length - 3 - len(commit_token)]
            # print(len(issue_token),len(commit_token))
            # print(args.max_seq_length - 3 - len(commit_token))
            # assert len(issue_token) + len(commit_token)+3 == 512, 'Error2'
        elif len(commit_token) > (args.max_seq_length - 3) / 2:
            commit_length = (args.max_seq_length - 3) - len(issue_token)
            commit_token = \
                commit_token[:int(commit_length / 2)] + \
                commit_token[
                len(commit_token) - (commit_length - int(commit_length / 2)):]
            # print(len(issue_token), len(commit_token))
            # assert len(issue_token) + len(commit_token)+3 == 512, 'Error3'
    combined_token = [tokenizer.cls_token] + issue_token + [tokenizer.sep_token] + commit_token + [tokenizer.sep_token]
    combined_ids = tokenizer.convert_tokens_to_ids(combined_token)
    if len(combined_ids) < args.max_seq_length:
        padding_length = args.max_seq_length - len(combined_ids)
        combined_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(combined_token, combined_ids, row['label'])


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.text_examples = []
        self.code_examples = []
        # df_link = MySubSampler(pd.read_csv(file_path), random.randint(100, 5000))
        if 'TRAIN' in file_path:
            df_link = MySubSampler(pd.read_csv(file_path), args.seed)
        else:
            df_link = pd.read_csv(file_path)
        # 将每一行数据得到相应的token+id+label
        for i_row, row in df_link.iterrows():
            self.text_examples.append(convert_examples_to_features(row, tokenizer, 'Commit_Text', args))
            self.code_examples.append(convert_examples_to_features(row, tokenizer, 'Commit_Code', args))
        assert len(self.text_examples) == len(self.code_examples), 'ErrorLength'
        # if 'TRAIN' in file_path:
        #     for text_example, code_example in zip(self.text_examples[:3], self.code_examples[:3]):
        #         print("*** Example ***")
        #         print("label: {}".format(code_example.label))
        #         print("input_tokens: {}".format([x.replace('\u0120', '_') for x in code_example.input_tokens]))
        #         print("input_tokens: {}".format([x.replace('\u0120', '_') for x in text_example.input_tokens]))
        #         print("input_ids: {}".format(' '.join(map(str, code_example.input_ids))))
        #         print("input_ids: {}".format(' '.join(map(str, text_example.input_ids))))

    def __len__(self):
        return len(self.text_examples)

    def __getitem__(self, i):
        return (torch.tensor(self.text_examples[i].input_ids),
                torch.tensor(self.code_examples[i].input_ids), torch.tensor(self.text_examples[i].label))
