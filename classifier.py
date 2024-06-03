import random, numpy as np, argparse
from types import SimpleNamespace
import csv
import pandas as pd

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

from tokenizer import BertTokenizer
from bert import BertModel
#from optimizer import AdamW
from transformers import AdamW
from tqdm import tqdm
import os
from utils import most_common_item, add_relative_weight_noise

TQDM_DISABLE=True
BAGG_SAVE_MODELS = False
BAGG_ADD_NOISE = False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class BertSentimentClassifier(torch.nn.Module):
    '''
    This module performs sentiment classification using BERT embeddings on the SST dataset.

    In the SST dataset, there are 5 sentiment categories (from 0 - "negative" to 4 - "positive").
    Thus, your forward() should return one logit for each of the 5 classes.
    '''
    def __init__(self, config):
        super(BertSentimentClassifier, self).__init__()
        self.num_labels = config.num_labels
        if config.fine_tune_mode == 'lora':
            self.bert = BertModel.from_pretrained('bert-base-uncased',  
                                                use_lora=True,
                                                lora_rank = config.lora_rank,
                                                lora_svd_init=config.lora_svd_init)
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Pretrain mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model", "lora"]
        if config.fine_tune_mode == 'lora':
            for name, param in self.bert.named_parameters():
                if "lora" in name or ("bert" in name and 'bias' in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False   
        else:
            for param in self.bert.parameters():
                if config.fine_tune_mode == 'last-linear-layer':
                    param.requires_grad = False
                elif config.fine_tune_mode == 'full-model':
                    param.requires_grad = True
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)     
        self.linear_proj = nn.Linear(config.hidden_size, self.num_labels)

        # Create any instance variables you need to classify the sentiment of BERT embeddings.

    def forward(self, input_ids, attention_mask):
        '''Takes a batch of sentences and returns logits for sentiment classes'''
        # The final BERT contextualized embedding is the hidden state of [CLS] token (the first token).
        # HINT: You should consider what is an appropriate return value given that
        # the training loop currently uses F.cross_entropy as the loss function.
        ### TODO

        last_hidden_state, pooler_output = {v for k,v in self.bert(input_ids, attention_mask).items()}
        embedding = self.dropout(pooler_output)

        proj = self.linear_proj(embedding)
        logits = F.softmax(proj, dim=1)
        return logits


class SentimentDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids= self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'sents': sents,
                'sent_ids': sent_ids
            }

        return batched_data


class SentimentTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        sent_ids = [x[1] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        return token_ids, attention_mask, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, sents, sent_ids= self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'sents': sents,
                'sent_ids': sent_ids
            }

        return batched_data


# Load the data: a list of (sentence, label).
def load_data(filename, flag='train'):
    num_labels = {}
    data = []
    if flag == 'test':
        with open(filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                data.append((sent,sent_id))
    else:
        with open(filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                label = int(record['sentiment'].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                data.append((sent, label,sent_id))
        print(f"load {len(data)} data from {filename}")

    if flag == 'train':
        return data, len(num_labels)
    else:
        return data


# Evaluate the model on dev examples.
def model_eval(dataloader, model, device):
    model.eval()
    y_true = []
    y_pred = []
    sents = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_mask, b_labels, b_sents, b_sent_ids = batch['token_ids'],batch['attention_mask'],  \
                                                        batch['labels'], batch['sents'], batch['sent_ids']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model(b_ids, b_mask)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        b_labels = b_labels.flatten()
        y_true.extend(b_labels)
        y_pred.extend(preds)
        sents.extend(b_sents)
        sent_ids.extend(b_sent_ids)

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, y_pred, y_true, sents, sent_ids
    
# Evaluate the model on test examples.
def model_test_eval(dataloader, model, device):
    model.eval() # Switch to eval model, will turn off randomness like dropout.
    y_pred = []
    sents = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_mask, b_sents, b_sent_ids = batch['token_ids'],batch['attention_mask'],  \
                                                         batch['sents'], batch['sent_ids']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model(b_ids, b_mask)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        y_pred.extend(preds)
        sents.extend(b_sents)
        sent_ids.extend(b_sent_ids)

    return y_pred, sents, sent_ids


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    train_data, num_labels = load_data(args.train, 'train')
    dev_data = load_data(args.dev, 'valid')
    
    train_dataset = SentimentDataset(train_data, args)
    dev_dataset = SentimentDataset(dev_data, args)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode,
              'lora_rank': args.lora_rank,
              'lora_svd_init': args.lora_svd_init
              }

    config = SimpleNamespace(**config)

    model = BertSentimentClassifier(config)
    model = model.to(device)
    print(f'training device, {device}\n')

    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters, {format(n_trainable_params, ',')}")
    
    # set different learning rate for 
    optimizer = AdamW([
    {'params': [param for name, param in model.named_parameters() if 'bert' in name], 'lr': args.lr_bert},
    {'params': [param for name, param in model.named_parameters() if 'bert' not in name], 'lr': args.lr_class},
    ], weight_decay=args.weight_decay)


    best_dev_acc = 0
    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, train_f1, *_  = model_eval(train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
    return best_dev_acc

def train_bagg(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    print(f'training device, {device}\n')

    train_data, num_labels = load_data(args.train, 'train')
    dev_data = load_data(args.dev, 'valid')
    test_data = load_data(args.test, 'test')
    dev_dataset = SentimentDataset(dev_data, args)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    test_dataset = SentimentTestDataset(test_data, args)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=test_dataset.collate_fn)
    
    dev_labels = {x[2]: x[1] for x in dev_data}
    
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode,
              'lora_rank': args.lora_rank,
              'lora_svd_init': args.lora_svd_init
              }

    config = SimpleNamespace(**config)
    
    eval_preds = [] # predictions of each model in ensamble on dev set
    test_preds = []
    filepath = args.filepath
    model_filepaths = []
    for i in tqdm(range(args.n_models)):
        train_data_model = random.choices(train_data, k=len(train_data)) # bootstrap
        train_dataset = SentimentDataset(train_data_model, args)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)

        model = BertSentimentClassifier(config) ## todo, maybe add some perturbation into model here
        if BAGG_ADD_NOISE:
            model = add_relative_weight_noise(model, noise_std=0.1) # the std value is arbitrary, not tuned
        
        model = model.to(device)
        optimizer = AdamW([
            {'params': [param for name, param in model.named_parameters() if 'bert' in name], 'lr': args.lr_bert},
            {'params': [param for name, param in model.named_parameters() if 'bert' not in name], 'lr': args.lr_class},
            ], weight_decay=args.weight_decay)
        
        
        if i == 0:
            n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Number of trainable parameters, {format(n_trainable_params, ',')}")
        
        filepath_model = f'model_{i}_{filepath}'
        model_filepaths.append(filepath_model)
        best_dev_acc = 0
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            num_batches = 0
            for batch in train_dataloader:
                b_ids, b_mask, b_labels = (batch['token_ids'],
                                           batch['attention_mask'], batch['labels'])

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model(b_ids, b_mask)
                loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            train_loss = train_loss / (num_batches)

            #train_acc, train_f1, *_  = model_eval(train_dataloader, model, device)
            dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                save_model(model, optimizer, args, config, filepath_model)
        
        # load best model and predict
        eval_pred = predict(dev_dataloader, filepath_model, device)
        eval_preds.append(eval_pred)
        test_pred = predict(test_dataloader, filepath_model, device)
        test_preds.append(test_pred)
        if not BAGG_SAVE_MODELS:
            os.remove(filepath_model)

    print('... Writing prediction files ...')
    eval_preds_out = pred_bagg(eval_preds, dev_labels, f"predictions/acc_{filepath_model.replace('pt', 'txt')}")
    test_preds_out = pred_bagg(test_preds)
    
    write_pred_file(eval_preds_out, args.dev_out)
    write_pred_file(test_preds_out, args.test_out)
    return best_dev_acc

def write_pred_file(preds, filepath):
    with open(filepath, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in preds.items():
                f.write(f"{p} , {s} \n")


def pred_bagg(preds, labels=None, acc_file=None):
    # majority vote prediction
    preds_out = {}
    sent_ids = [k for k in preds[0].keys()]
    for sent_id in sent_ids:
        preds_ = [pred[sent_id] for pred in preds]
        preds_out[sent_id] = most_common_item(preds_) # majority vote, if tied most freq, then random among most freq
    
    if labels:
        accuracy = sum([1 if preds_out[sent_id] == labels[sent_id] else 0 for sent_id in sent_ids]) / len(sent_ids)
        print(f'Accuracy on dev set, {accuracy}')
        with open(acc_file, 'w') as file:
            file.write(f'{accuracy}\n')

    return preds_out
        
def test(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']
        model = BertSentimentClassifier(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"load model from {args.filepath}")
        
        dev_data = load_data(args.dev, 'valid')
        dev_dataset = SentimentDataset(dev_data, args)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

        test_data = load_data(args.test, 'test')
        test_dataset = SentimentTestDataset(test_data, args)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)
        
        dev_acc, dev_f1, dev_pred, dev_true, dev_sents, dev_sent_ids = model_eval(dev_dataloader, model, device)
        print('DONE DEV')
        test_pred, test_sents, test_sent_ids = model_test_eval(test_dataloader, model, device)
        print('DONE Test')
        with open(args.dev_out, "w+") as f:
            print(f"dev acc :: {dev_acc :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sent_ids,dev_pred ):
                f.write(f"{p} , {s} \n")

        with open(args.test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s  in zip(test_sent_ids,test_pred ):
                f.write(f"{p} , {s} \n")

def predict(dataloader, filepath_model, device):
    with torch.no_grad():
        saved = torch.load(filepath_model)
        config = saved['model_config']
        model = BertSentimentClassifier(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        pred_dict = model_pred(dataloader, model, device)
    
    return pred_dict

def model_pred(dataloader, model, device):
    # collect predictions for one model in bagging ensamble
    model.eval()
    sent_ids = []
    y_pred = []
    for _, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_mask, b_sent_ids = batch['token_ids'],batch['attention_mask'], batch['sent_ids']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model(b_ids, b_mask)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()
        y_pred.extend(preds)
        sent_ids.extend(b_sent_ids)
    
    # return dict sent_id -> pred
    return {sent_id: y_pred[i] for i, sent_id in enumerate(sent_ids)}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well, lora:lora on bert layers',
                        choices=('last-linear-layer', 'full-model', 'lora'), default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr_bert", type=float, help="learning rate for bert layers, full or lora",
                        default=1e-3)
    parser.add_argument("--lr_class", type=float, help="learning rate for classification layers",
                        default=1e-3)
    parser.add_argument("--lora_rank", type=int, help="rank of lora adapters",
                        default=8)
    parser.add_argument("--lora_svd_init", action='store_true')
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--use_bagging", action='store_true')
    parser.add_argument('--n_models_bagging', type=int, help='Number of models to use in bagging ensamble', default=10)

    parser2 = argparse.ArgumentParser()
    parser2.add_argument("--lr_class_min", type=float, help="Minimum learning rate for the learning rate range of classifier", default=5e-4)
    parser2.add_argument("--lr_class_max", type=float, help="Maximum learning rate for the learning rate range of classifier", default=5e-2)
    parser2.add_argument("--lr_bert_min", type=float, help="Minimum learning rate for the learning rate range of the bert model", default=5e-5)
    parser2.add_argument("--lr_bert_max", type=float, help="Maximum learning rate for the learning rate range of the bert model", default=5e-3)
    parser2.add_argument("--n_lr_steps", type = int, help="Number of learning rate steps in both the classifier and bert model", default = 10)
    
    args, _ = parser.parse_known_args()
    args2, _ = parser2.parse_known_args()
    return args, args2


if __name__ == "__main__":
    args, lr_args = get_args()
    #seed_everything(args.seed)

    base_filepath_sst = f'classifier_{args.fine_tune_mode}'
    base_dev_out = f'predictions/{args.fine_tune_mode}'
    base_test_out = f'predictions/{args.fine_tune_mode}'

    if args.fine_tune_mode == 'lora':
        lora_details = f'_{args.lora_rank}_{args.lora_svd_init}'
    else:
        lora_details = ''
    
    if args.use_bagging:
        bagg_details = f'_bagg_{args.n_models_bagging}'
    else:
        bagg_details = ''

    filepath_sst = f'sst-{base_filepath_sst}{lora_details}_{args.lr_bert}_{args.lr_class}{bagg_details}.pt'
    dev_out_sst = f'{base_dev_out}{lora_details}_{args.lr_bert}_{args.lr_class}{bagg_details}-sst-dev-out.csv'
    test_out_sst = f'{base_test_out}{lora_details}_{args.lr_bert}_{args.lr_class}{bagg_details}-sst-test-out.csv'

    filepath_cfimdb = f'cfimdb-{base_filepath_sst}{lora_details}_{args.lr_bert}_{args.lr_class}{bagg_details}.pt'
    dev_out_cdimdb = f'{base_dev_out}{lora_details}_{args.lr_bert}_{args.lr_class}{bagg_details}-cfimdb-dev-out.csv'
    test_out_cfimdb = f'{base_test_out}{lora_details}_{args.lr_bert}_{args.lr_class}{bagg_details}-cdimdb-test-out.csv'

    n = lr_args.n_lr_steps
    bert_lrs = np.linspace(lr_args.lr_bert_min, lr_args.lr_bert_max, n)
    class_lrs = np.linspace(lr_args.lr_class_min, lr_args.lr_class_max, n)
    
    results_sst = pd.DataFrame(np.zeros([n, n]), index = bert_lrs, columns = class_lrs)
    results_cfimdb = pd.DataFrame(np.zeros([n, n]), index = bert_lrs, columns = class_lrs)
    
    lr_search_file_sst = f'lr_searches/sst_class_{class_min}-{class_max}__bert_{bert_min}-{bert_max}_{args.fine_tune_mode}_{lora_details}{bagg_details}.csv'
    lr_search_file_cfimdb = f'lr_searches/cfimdb_class_{class_min}-{class_max}__bert_{bert_min}-{bert_max}_{args.fine_tune_mode}_{lora_details}{bagg_details}.csv'
    
    for i in range(n):
        for j in range(n):
            lr_bert = bert_lrs[i]
            lr_class = class_lrs[j]
            
            config_sst = SimpleNamespace(
                filepath=filepath_sst,
                lr_bert=lr_bert,
                lr_class=lr_class,
                use_gpu=args.use_gpu,
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden_dropout_prob=args.hidden_dropout_prob,
                train='data/ids-sst-train.csv', # modify to use augmented data
                dev='data/ids-sst-dev.csv',
                test='data/ids-sst-test-student.csv',
                fine_tune_mode=args.fine_tune_mode,
                dev_out = dev_out_sst,
                test_out = test_out_sst,
                lora_rank=args.lora_rank,
                lora_svd_init=args.lora_svd_init,
                weight_decay=args.weight_decay,
                n_models = args.n_models_bagging
            )

            if args.use_bagging:
                print('Training Bagging Ensamble Classifier on SST...')
                min_sst = train_bagg(config_sst)
                
            else:
                print('Training Sentiment Classifier on SST...')
                min_sst = train(config_sst)

            print('Training Sentiment Classifier on cfimdb...')
            config_cfimbd = SimpleNamespace(
                filepath=filepath_cfimdb,
                lr_bert=lr_bert,
                lr_class=lr_class,
                use_gpu=args.use_gpu,
                epochs=args.epochs,
                batch_size=8,
                hidden_dropout_prob=args.hidden_dropout_prob,
                train='data/ids-cfimdb-train.csv',
                dev='data/ids-cfimdb-dev.csv',
                test='data/ids-cfimdb-test-student.csv',
                fine_tune_mode=args.fine_tune_mode,
                dev_out = dev_out_cdimdb,
                test_out = test_out_cfimdb,
                lora_rank=args.lora_rank,
                lora_svd_init=args.lora_svd_init,
                weight_decay=args.weight_decay,
                n_models = args.n_models_bagging
            )

            if args.use_bagging:
                print('Training Bagging Ensamble Classifier on cfimbd...')
                min_cfimdb = train_bagg(config_cfimbd)
            else:
                print('Training Sentiment Classifier on cfimdb...')
                min_cfimdb = train(config_cfimbd)
                
            results_sst.iloc[i, j] = min_sst
            results_cfimdb.iloc[i, j] = min_cfimdb
            
            #Write to csv each time just so results aren't lost if job is cut short or another issue is encountered.
            results_sst.to_csv(lr_search_file_sst)
            results_cfimdb.to_csv(lr_search_file_cfimdb)
            
