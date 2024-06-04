import argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random

from bertmodel import BertModel
from transformers import AdamW
from tqdm import tqdm
from utils import most_common_item, add_relative_weight_noise

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_test_sst

TQDM_DISABLE = False
BAGG_ADD_NOISE = False

BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

class SentimentBERT(nn.Module):
    def __init__(self, config, bert_state_dict, classifier_state_dict):
        super(SentimentBERT, self).__init__()
        self.bert = BertModel(config, bert_state_dict)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sentiment_classifier = nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        self.sentiment_classifier.load_state_dict(classifier_state_dict)

        self.siar_lamb = config.siar_lamb
        self.siar_eps = config.siar_eps

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
    
    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs['last_hidden_state'][:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        return cls_embedding
    
    

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        cls_embedding = self.forward(input_ids, attention_mask)
        logits = self.sentiment_classifier(cls_embedding)
        return logits
    
    # SIAR =========================
    def forward_with_noise(self, input_ids, attention_mask):
        'Produces noisy embeddings for the sentences, used for SIAR.'
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs['last_hidden_state'][:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        return cls_embedding + self.siar_eps * torch.randn_like(cls_embedding)
    
    def predict_sentiment_siar(self, input_ids, attention_mask):
        cls_embedding = self.forward_with_noise(input_ids, attention_mask)
        logits = self.sentiment_classifier(cls_embedding)
        return logits
    
    def siar_reg_loss_sentiment(self, input_ids, attention_mask):
        '''SIAR regularization loss for sentiment classification task.'''
        logits = self.predict_sentiment(input_ids, attention_mask)
        perturbed_logits = self.predict_sentiment_siar(input_ids, attention_mask) # uses the noisy embeddings
        loss = F.kl_div(F.log_softmax(logits, dim=1), F.softmax(perturbed_logits, dim=1), reduction='batchmean')
        return loss
    # ==============================

def train(args, sst_train_data, sst_dev_data, sst_test_data):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sst_dev_data.collate_fn)
    sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
    sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size, collate_fn=sst_test_data.collate_fn)

    config = gen_config(args.fine_tune_mode, args.lora_rank, args.lora_svd_init, args.lora_pretrained,
                        args.use_siar, args.siar_lamb, args.siar_eps)
    saved_checkpoint = torch.load(args.load_checkpoint_path)

    model = SentimentBERT(config, saved_checkpoint['bert'] , saved_checkpoint['classifier'])
    
    if args.use_bagging and BAGG_ADD_NOISE:
        model = add_relative_weight_noise(model, noise_std=0.1) # the std value is arbitrary, not tuned

    model.to(device)
    print(f"training device, {device}\n")

    optimizer = AdamW([
    {'params': [param for name, param in model.named_parameters() if 'bert' in name], 'lr': args.lr_bert},
    {'params': [param for name, param in model.named_parameters() if 'bert' not in name], 'lr': args.lr_class},
    ], weight_decay=args.weight_decay)

    best_dev_acc = 0.0
    best_dev_preds = None
    best_test_preds = None # test preds for corresponding to best eval epoch

    for epoch in range(args.epochs):
        model.train()
        model.to(device)
        train_loss = 0
        num_batches = 0
        for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            ids, mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels']
            batch_size = ids.size(0) # effective batch size
            ids = ids.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model.predict_sentiment(ids, mask)
            loss = F.cross_entropy(logits, labels.view(-1), reduction='sum') / batch_size
            if args.use_siar:
                loss += args.siar_lamb * model.siar_reg_loss_sentiment(ids, mask)/ batch_size

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1
        
        train_loss = train_loss / (num_batches)

        train_acc, train_f1, *_  = model_eval_sst(sst_train_dataloader, model, device)
        dev_acc, dev_f1, dev_preds, _, _, dev_sst_sent_ids = model_eval_sst(sst_dev_dataloader, model, device)
        test_preds, test_sst_sent_ids = model_eval_test_sst(sst_test_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_dev_preds = dev_preds
            best_test_preds = test_preds
            if args.save_checkpoints:
                save_checkpoint(model, config, args.dump_checkpoint_path)

        if not args.use_bagging:
            print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")

    print(f'Best dev accuracy, {best_dev_acc}')
    dev_pred_dict = {sent_id: best_dev_preds[i] for i, sent_id in enumerate(dev_sst_sent_ids)}
    test_pred_dict = {sent_id: best_test_preds[i] for i, sent_id in enumerate(test_sst_sent_ids)}
    
    return dev_pred_dict, test_pred_dict

def train_bagg(args, sst_train_data, sst_dev_data, sst_test_data, n_models):
    dev_labels = {x[2]: x[1] for x in sst_dev_data}
    dev_pred_dicts, test_pred_dicts = [], []
    assert args.save_checkpoints == False, "Cannot save checkpoint bagging models, too much storage needed"

    for _ in tqdm(range(n_models)):
        train_data_model = random.choices(sst_train_data, k=len(sst_train_data)) # bootstrap
        dev_preds, test_preds = train(args, train_data_model, sst_dev_data, sst_test_data)
        dev_pred_dicts.append(dev_preds)
        test_pred_dicts.append(test_preds)

    dev_preds_out = pred_bagg(dev_pred_dicts, labels=dev_labels, acc_filepath=f"predictions/acc_{args.dump_checkpoint_path.replace('pt', 'txt')}")
    test_preds_out = pred_bagg(test_pred_dicts)
    return dev_preds_out, test_preds_out    

def pred_bagg(preds, labels=None, acc_filepath=None):
    # majority vote prediction
    preds_out = {}
    sent_ids = [k for k in preds[0].keys()]
    for sent_id in sent_ids:
        preds_ = [pred[sent_id] for pred in preds]
        preds_out[sent_id] = most_common_item(preds_) # majority vote, if tied most freq, then random among most freq
    
    if labels:
        accuracy = sum([1 if preds_out[sent_id] == labels[sent_id] else 0 for sent_id in sent_ids]) / len(sent_ids)
        print(f'\n Bagging accuracy on dev set, {accuracy}')
        with open(acc_filepath, 'w') as file:
            file.write(f'{accuracy}\n')
    
    return preds_out
    
def save_checkpoint(model, config, dump_checkpoint_path):
    save_info = {
        'bert': model.bert.cpu().state_dict(),
        'classifier': model.sentiment_classifier.cpu().state_dict(),
        'config': config
    }
    torch.save(save_info, dump_checkpoint_path)
    print(f"save the model checkpoint to {dump_checkpoint_path}")

def write_pred_file(pred_dict, filepath):
    with open(filepath, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in pred_dict.items():
                f.write(f"{p} , {s} \n")      
        
def gen_config(fine_tune_mode, lora_rank=None, lora_svd_init=None, lora_pretrained=False,
               use_siar=False, siar_lamb=None, siar_eps=None):
    config = {
        "num_labels": N_SENTIMENT_CLASSES,
        "hidden_size": 768,
        "data_dir": '.',
        "fine_tune_mode": fine_tune_mode,
        "lora_rank": lora_rank,
        "lora_svd_init": lora_svd_init,
        "vocab_size": 30522,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "pad_token_id": 0,
        "gradient_checkpointing": False,
        "position_embedding_type": "absolute",
        "use_cache": True,
        "lora_post": lora_pretrained,
        "use_siar": use_siar,
        "siar_lamb": siar_lamb,
        "siar_eps": siar_eps
    }

    return SimpleNamespace(**config)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    # included for load_multitask_data, not used
    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")
    ##############################################################################

    parser.add_argument("--load_checkpoint_path", type=str, default="sst_full-model-siar-0.1-0.1-1-1e-05-0.0001-multitask.pt")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--fine-tune-mode", type=str,
                        choices=('last-linear-layer', 'full-model', 'lora'), default="full-model")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output-ft")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output-ft")
    
    parser.add_argument("--lr_bert", type=float, help="learning rate for bert layers, full or lora",
                        default=1e-5)
    parser.add_argument("--lr_class", type=float, help="learning rate for classification layers",
                        default=1e-4)
    parser.add_argument("--lora_rank", type=int, help="rank of lora adapters",
                        default=32)
    parser.add_argument("--lora_svd_init", action='store_true')
    parser.add_argument("--lora_pretrained", action='store_true') # multitask model was trained with lora
    parser.add_argument("--batch_size", type=int, default=64) 
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--use_bagging", action='store_true')
    parser.add_argument('--n_models_bagging', type=int, help='Number of models to use in bagging ensamble', default=10)
    parser.add_argument("--save_checkpoints", action='store_true')

    parser.add_argument("--use_siar", action='store_true')
    parser.add_argument("--siar_lamb", type=float, default=0.1)
    parser.add_argument("--siar_eps", type=float, default=0.1)

    args = parser.parse_args()
    assert not (args.lora_pretrained and args.fine_tune_mode == 'lora'), "Cannot do lora on model that was pretrained with lora"
    return args

if __name__ == '__main__':
    args = get_args()

    sst_train_data, *_ = load_multitask_data(args.sst_train, args.para_train, args.sts_train, split='train')
    sst_dev_data, *_ = load_multitask_data(args.sst_dev, args.para_dev, args.sts_dev, split='dev')
    sst_test_data, *_= load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

    if args.fine_tune_mode == 'lora':
        lora_details = f'-lora-{args.lora_rank}-{args.lora_svd_init}'
    else:
        lora_details = ''
    
    if args.use_siar:
        siar_details = f'-siar-{args.siar_lamb}-{args.siar_eps}'
    else:
        siar_details = ''

    if args.use_bagging:
        bagg_details = f'-bagg-{args.n_models_bagging}'
    else:
        bagg_details = ''
    
    args.dump_checkpoint_path = f'ft{bagg_details}{siar_details}{lora_details}-{args.epochs}{args.load_checkpoint_path}' # Save path.
    args.sst_dev_out = f"{args.sst_dev_out}{siar_details}{lora_details}-{args.epochs}-{args.lr_bert}-{args.lr_class}{bagg_details}-ft.csv"
    args.sst_test_out = f"{args.sst_test_out}{siar_details}{lora_details}-{args.epochs}-{args.lr_bert}-{args.lr_class}{bagg_details}-ft.csv"
    
    print("=== Finetuning SST BERT ===")
    
    if args.use_bagging:
        TQDM_DISABLE = True
        dev_pred_dict, test_pred_dict = train_bagg(args, sst_train_data, sst_dev_data, sst_test_data, args.n_models_bagging)
    else:
        dev_pred_dict, test_pred_dict = train(args, sst_train_data, sst_dev_data, sst_test_data)
    
    write_pred_file(dev_pred_dict, args.sst_dev_out)
    write_pred_file(test_pred_dict, args.sst_test_out)

