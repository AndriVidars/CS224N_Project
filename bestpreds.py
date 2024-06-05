import os
from datasets import load_multitask_data
import csv
import numpy as np
import fnmatch

sst_path =   "data/ids-sst-dev.csv"
para_path ="data/quora-dev.csv"
sts_path="data/sts-dev.csv"

def get_label_dict(data, idx_label, idx_key):
    # get id -> label dict for dataset
    return {x[idx_key]: x[idx_label] for x in data}

def read_preds(filepath, dtype):
    pred_dict = {}
    with open(filepath, 'r') as fp:
        lines = fp.readlines()
        for record in lines[1:]:
            record_ = record.split()
            if 'para' in filepath:
                id = f'{record_[0]} {record_[1]}'
            else:
                id = record_[0]
            label = int(record_[-1]) if dtype == 'int' else float(record_[-1])
            pred_dict[id] = label    
    return pred_dict

def accuracy(label_dict, pred_dict):
    ids = [k for k in label_dict.keys()]
    return sum([1 if label_dict[id] == pred_dict[id] else 0 for id in ids]) / len(ids)

def corr(label_dict, pred_dict):
    labels = [v for _, v in label_dict.items()]
    preds = [pred_dict[k] for k in label_dict.keys()]
    pearson_mat = np.corrcoef(preds, labels)
    corr = pearson_mat[1][0]
    return corr
    
def findbest(task, label_dict, metric_fn, dtype):
    score_dict = {} # filename -> score on metric
    for root, dirs, files in os.walk('predictions'):
        for filename in fnmatch.filter(files, f"*{task}*dev*.csv"):
            filepath = os.path.join(root, filename)
            pred_dict = read_preds(filepath, dtype)
            score = metric_fn(label_dict, pred_dict)
            score_dict[filepath] = score
    
    best_file = max(score_dict, key=score_dict.get)
    best_score = score_dict[best_file]

    print(f'Task, {task}')
    print(f'Best file, {best_file}\nScore, {best_score}')
    print(f"Corresponding test file, {best_file.replace('dev', 'test')}\n")
    
    
if __name__ == '__main__':
    sst_dev_data, _, para_dev_data, sts_dev_data = load_multitask_data(sst_path, para_path, sts_path, split='dev')

    sst_dict = get_label_dict(sst_dev_data, 1, 2)
    para_dict = get_label_dict(para_dev_data, 2, 3)
    sts_dict = get_label_dict(sts_dev_data, 2, 3)
    
    print('\n')
    findbest('sst', sst_dict, accuracy, 'int')
    findbest('para', para_dict, accuracy, 'float')
    findbest('sts', sts_dict, corr, 'float')
