import random, numpy as np, argparse
from types import SimpleNamespace
import csv

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

from tokenizer import BertTokenizer
from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm


TQDM_DISABLE=False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class BertSimilarityClassifier(torch.nn.Module):
    '''
    This module performs sentiment classification using BERT embeddings on the SemEval dataset.
    '''
    def __init__(self, config):
        super(BertSimilarityClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased') # use the pre-trained BERT model
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = True # was initially False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear_proj = nn.Linear(config.hidden_size, 1)
    
    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        outputs1 = self.bert(input_ids=input_ids1, attention_mask=attention_mask1)
        outputs2 = self.bert(input_ids=input_ids2, attention_mask=attention_mask2)
        embedding1 = self.dropout(outputs1['pooler_output'])
        embedding2 = self.dropout(outputs2['pooler_output'])
        return embedding1, embedding2
    
# cosine similarity loss function
def cosine_similarity_loss(embeddings1, embeddings2, scores):
    cos_sim = F.cosine_similarity(embeddings1, embeddings2)
    loss = F.mse_loss(cos_sim, scores)
    return loss

class SemEvalDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def pad_data(self, data):
        sents1 = [x[0] for x in data]
        sents2 = [x[1] for x in data]
        labels = [x[2] for x in data]
        sent_ids = [x[3] for x in data]

        encoding1 = self.tokenizer(sents1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(sents2, return_tensors='pt', padding=True, truncation=True)

        token_ids1 = torch.LongTensor(encoding1['input_ids'])
        attention_mask1 = torch.LongTensor(encoding1['attention_mask'])
        token_ids2 = torch.LongTensor(encoding2['input_ids'])
        attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
        
        labels = torch.FloatTensor(labels)

        return token_ids1, attention_mask1, token_ids2, attention_mask2, labels, sents1, sents2, sent_ids

    
    def collate_fn(self, all_data):
        token_ids1, attention_mask1, token_ids2, attention_mask2, labels, sents1, sents2, sent_ids = self.pad_data(all_data)
        batched_data = {
            'token_ids1': token_ids1,
            'attention_mask1': attention_mask1,
            'token_ids2': token_ids2,
            'attention_mask2': attention_mask2,
            'labels': labels,
            'sents1': sents1,
            'sents2': sents2,
            'sent_ids': sent_ids
        }
        return batched_data

class SemEvalTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents1 = [x[0] for x in data]
        sents2 = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding1 = self.tokenizer(sents1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(sents2, return_tensors='pt', padding=True, truncation=True)
        
        token_ids1 = encoding1['input_ids']
        attention_mask1 = encoding1['attention_mask']
        token_ids2 = encoding2['input_ids']
        attention_mask2 = encoding2['attention_mask']

        return token_ids1, attention_mask1, token_ids2, attention_mask2, sents1, sents2, sent_ids
    
    def collate_fn(self, all_data):
        
        token_ids1, attention_mask1, token_ids2, attention_mask2, sents1, sents2, sent_ids = self.pad_data(all_data)
        batched_data = {
            'token_ids1': token_ids1,
            'attention_mask1': attention_mask1,
            'token_ids2': token_ids2,
            'attention_mask2': attention_mask2,
            'sents1': sents1,
            'sents2': sents2,
            'sent_ids': sent_ids
        }
        return batched_data

# Load the data: a list of (sentence, sentence, label) tuples.
def load_data(filename, flag='train'):
    num_labels = {}
    data = []
    if flag == 'test':
        with open(filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent1 = record['sentence1'].lower().strip()
                sent2 = record['sentence2'].lower().strip()
                sent_id = record['id'].lower().strip()
                data.append((sent1, sent2 ,sent_id))
    else:
        with open(filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent1 = record['sentence1'].lower().strip()
                sent2 = record['sentence2'].lower().strip()
                sent_id = record['id'].lower().strip()
                label = float(record['similarity'].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                data.append((sent1, sent2, label, sent_id))
        print(f"load {len(data)} data from {filename}")

    if flag == 'train':
        return data, len(num_labels)
    else:
        return data


def model_eval(dataloader, model, device):
    model.eval()  # Switch to eval model, will turn off randomness like dropout
    y_true = []
    y_pred = []
    sents1 = []
    sents2 = []
    sent_ids = []

    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids1, b_mask1, b_ids2, b_mask2, b_similarities, b_sents1, b_sents2, b_sent_ids = \
            batch['token_ids1'], batch['attention_mask1'], batch['token_ids2'], batch['attention_mask2'], \
            batch['similarities'], batch['sents1'], batch['sents2'], batch['sent_ids']

        b_ids1 = b_ids1.to(device)
        b_mask1 = b_mask1.to(device)
        b_ids2 = b_ids2.to(device)
        b_mask2 = b_mask2.to(device)
        b_similarities = b_similarities.to(device)

        embeddings1, embeddings2 = model(b_ids1, b_mask1, b_ids2, b_mask2)
        cos_sim = F.cosine_similarity(embeddings1, embeddings2).detach().cpu().numpy()

        y_true.extend(b_similarities.cpu().numpy())
        y_pred.extend(cos_sim)
        sents1.extend(b_sents1)
        sents2.extend(b_sents2)
        sent_ids.extend(b_sent_ids)

        mse = np.mean((np.array(y_true) - np.array(y_pred)) ** 2)
    return mse, y_true, y_pred, sents1, sents2, sent_ids


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


# adjust the linear_proj layer to match the expected shape from the loaded state dictionary
# mismatch due to there being 5 classes in the first fine-tuned model and 2 classes in the current model
def load_model_with_mismatch(model, state_dict):
    model_state_dict = model.state_dict()
    pretrained_state_dict = state_dict
    # Filter out keys that have size mismatches
    filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
    # Update the existing model state dict with the pre-trained weights
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)


def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    
    train_data, num_labels = load_data(args.train)
    dev_data = load_data(args.dev)
    
    train_dataset = SemEvalDataset(train_data, args)
    dev_dataset = SemEvalDataset(dev_data, args)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode}
    
    config = SimpleNamespace(**config)

    # Sequential fine-tuning: load the already fine-tuned sentiment model
    if args.pretrained_model_path:
        print(f"Loading pretrained model from {args.pretrained_model_path}")
        saved = torch.load(args.pretrained_model_path, map_location=torch.device('cpu'))
        model = BertSimilarityClassifier(config)
        load_model_with_mismatch(model, saved['model'])
    else:
        model = BertSimilarityClassifier(config)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids1, b_mask1, b_ids2, b_mask2, b_scores = batch['token_ids1'], batch['attention_mask1'], \
                                                         batch['token_ids2'], batch['attention_mask2'], batch['labels']
            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)
            b_scores = b_scores.to(device)

            optimizer.zero_grad()
            embeddings1, embeddings2 = model(b_ids1, b_mask1, b_ids2, b_mask2)
            loss = cosine_similarity_loss(embeddings1, embeddings2, b_scores)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / len(train_dataloader)
        train_loss = train_loss / (num_batches)

        train_acc, train_f1, *_  = model_eval(train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")

def model_eval(dataloader, model, device):
    model.eval()
    y_true = []
    y_pred = []
    for step, batch in enumerate(tqdm(dataloader, desc='eval', disable=TQDM_DISABLE)):
        b_ids1, b_mask1, b_ids2, b_mask2, b_scores = batch['token_ids1'], batch['attention_mask1'], \
                                                     batch['token_ids2'], batch['attention_mask2'], batch['scores']
        b_ids1 = b_ids1.to(device)
        b_mask1 = b_mask1.to(device)
        b_ids2 = b_ids2.to(device)
        b_mask2 = b_mask2.to(device)

        embeddings1, embeddings2 = model(b_ids1, b_mask1, b_ids2, b_mask2)
        cos_sim = F.cosine_similarity(embeddings1, embeddings2).cpu().numpy()
        y_true.extend(b_scores.cpu().numpy())
        y_pred.extend(cos_sim)
    
    mse = np.mean((np.array(y_true) - np.array(y_pred)) ** 2)
    return mse, y_true, y_pred


def test(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']
        model = BertSimilarityClassifier(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"load model from {args.filepath}")
        
        dev_data = load_data(args.dev, 'valid')
        dev_dataset = SemEvalDataset(dev_data, args)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

        test_data = load_data(args.test, 'test')
        test_dataset = SemEvalTestDataset(test_data, args)
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-3)
    parser.add_argument("--pretrained_model_path", type=str, default=None, 
                        help="Path to the already fine-tuned model")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)

    print('Training Similarity Classifier on STS...')
    config = SimpleNamespace(
        filepath='sts-classifier.pt',
        lr=args.lr,
        use_gpu=args.use_gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        train='data/sts-train.csv',
        dev='data/sts-dev.csv',
        test='data/sts-test-student.csv',
        fine_tune_mode=args.fine_tune_mode,
        pretrained_model_path=args.pretrained_model_path,
        dev_out='predictions/cosine_similarity/' + args.fine_tune_mode + '-sts-dev-out.csv',
        test_out='predictions/cosine_similarity/' + args.fine_tune_mode + '-sts-test-out.csv'
    )

    train(config)

    print('Evaluating on STS...')
    test(config)