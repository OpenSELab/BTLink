from __future__ import absolute_import, division, print_function

import datetime

from sklearn.metrics import precision_score, recall_score, f1_score, \
    roc_auc_score, matthews_corrcoef, brier_score_loss, confusion_matrix
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, \
    SequentialSampler
from tqdm import tqdm
from transformers import RobertaTokenizer
from utils import getargs, set_seed, TextDataset
import warnings

warnings.filterwarnings(action='ignore')


def test(args, model, tokenizer):
    # Note that DistributedSampler samples randomly
    stime = datetime.datetime.now()
    eval_dataset = TextDataset(tokenizer, args, os.path.join(args.data_dir, args.pro + '_TEST.csv'))
    args.seed += 3
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    print("********** Running Test **********")
    print("  Num examples = {}".format(len(eval_dataset)))
    print("  Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        text_inputs = batch[0].to(args.device)
        code_inputs = batch[1].to(args.device)
        label = batch[2].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(text_inputs, code_inputs, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits.argmax(-1)
    etime = datetime.datetime.now()

    eval_time = (etime - stime).seconds
    eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
    eval_precision = precision_score(labels, preds)
    eval_recall = recall_score(labels, preds)
    eval_f1 = f1_score(labels, preds)
    eval_auc = roc_auc_score(labels, preds)
    eval_mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=preds).ravel()
    eval_pf = fp / (fp + tn)
    eval_brier = brier_score_loss(labels, preds)

    result = {
        "eval_loss": float(perplexity),
        "eval_time": float(eval_time),
        "eval_acc": round(float(eval_acc), 4),
        "eval_precision": round(eval_precision, 4),
        "eval_recall": round(eval_recall, 4),
        "eval_f1": round(eval_f1, 4),
        "eval_auc": round(eval_auc, 4),
        "eval_mcc": round(eval_mcc, 4),
        "eval_brier": round(eval_brier, 4),
        "eval_pf": round(eval_pf, 4),
    }
    print(preds[:25], labels[:25])
    print("********** Test results **********")
    dfScores = pd.DataFrame(columns=['Metrics', 'Score'])
    for key in sorted(result.keys()):
        print('-' * 10 + "  {} = {}".format(key, str(round(result[key], 4))))
        dfScores.loc[len(dfScores)] = [key, str(round(result[key], 4))]

    filepath = os.path.join(args.result_dir, args.trained_pro)
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    dfScores.to_csv(os.path.join(filepath, args.trained_pro + '_' + args.pro + "_Metrics.csv"), index=False)
    assert len(logits) == len(preds) and len(logits) == len(labels), 'error'
    logits4class0, logits4class1 = \
        [logits[iclass][0] for iclass in range(len(logits))], \
        [logits[iclass][1] for iclass in range(len(logits))]
    df = pd.DataFrame(np.transpose([logits4class0, logits4class1, preds, labels]),
                      columns=['0_logit', '1_logit', 'preds', 'labels'])
    df.to_csv(os.path.join(filepath, args.trained_pro + '_' + args.pro + "__predictions.csv"), index=False)


def main(args):
    print("device: {}, n_gpu: {}".format(args.device, args.n_gpu))
    # Set seed
    set_seed(args.seed)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    print("Training/evaluation parameters {}".format(args))

    if args.do_test:
        checkpoint_prefix = args.trained_pro + '_checkpoint-best/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model = torch.load(output_dir)
        model.to(args.device)
        test(args, model, tokenizer)


if __name__ == "__main__":
    ls = ['Avro', 'Buildr', 'Giraph', 'ant-ivy', 'logging-log4net', 'Nutch', 'OODT', 'Tez', 'Tika', 'Isis',
          'keras']
    key = ['AVRO', 'BUILDR', 'GIRAPH', 'IVY', 'LOG4NET', 'NUTCH', 'OODT', 'TEZ', 'TIKA', 'ISIS', '#']
    args = getargs()
    for p, k in zip(ls, key):
        if args.trained_pro == p:
            continue
        args.pro = p
        args.key = k

        filepath = os.path.join(args.result_dir, args.trained_pro)
        if os.path.exists(os.path.join(filepath, args.trained_pro + '_' + args.pro + "__predictions.csv")):
            continue

        print("Test: " + args.pro + "\nTrained model:" + args.trained_pro)
        print("======BEGIN======" * 20)
        main(args)
