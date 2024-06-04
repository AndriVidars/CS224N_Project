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
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)
import numpy as np

from evaluation import model_eval_sts, model_eval_test_sts

TQDM_DISABLE = False
BAGG_ADD_NOISE = False

BERT_HIDDEN_SIZE = 768

class SimilarityBERT(nn.Module):
    def __init__(self, config, bert_state_dict, projector_state_dict):
        super(SimilarityBERT, self).__init__()
        self.bert = BertModel(config, bert_state_dict)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.similarity_proj = nn.Sequential(
            nn.Linear(BERT_HIDDEN_SIZE, int(BERT_HIDDEN_SIZE / 2)),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(int(BERT_HIDDEN_SIZE / 2), int(BERT_HIDDEN_SIZE / 8)),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.similarity_proj.load_state_dict(projector_state_dict)

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

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).'''
        cls_embedding_1 = self.similarity_proj(self.forward(input_ids_1, attention_mask_1))
        cls_embedding_2 = self.similarity_proj(self.forward(input_ids_2, attention_mask_2))
        logits = 5 * F.cosine_similarity(cls_embedding_1, cls_embedding_2, dim=1).unsqueeze(1)
        return logits

    # SIAR =========================
    def forward_with_noise(self, input_ids, attention_mask):
        'Produces noisy embeddings for the sentences, used for SIAR.'
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs['last_hidden_state'][:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        return cls_embedding + self.siar_eps * torch.randn_like(cls_embedding)

    def predict_similarity_siar(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        cls_embedding_1 = self.similarity_proj(self.forward_with_noise(input_ids_1, attention_mask_1))
        cls_embedding_2 = self.similarity_proj(self.forward_with_noise(input_ids_2, attention_mask_2))
        logits = 5 * F.cosine_similarity(cls_embedding_1, cls_embedding_2, dim=1).unsqueeze(1)

        return logits

    def siar_reg_loss_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        '''SIAR regularization loss for similarity task.'''
        logits = self.predict_similarity(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        perturbed_logits = self.predict_similarity_siar(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        loss = F.kl_div(F.log_softmax(logits, dim=1), F.softmax(perturbed_logits, dim=1), reduction='batchmean')

        return loss
     # ==============================

def train(args, train_data, dev_data, test_data):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    train_data = SentencePairDataset(train_data, args)
    dev_data = SentencePairDataset(dev_data, args)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size, collate_fn=train_data.collate_fn)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=dev_data.collate_fn)
    test_data = SentencePairTestDataset(test_data, args)
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=args.batch_size, collate_fn=test_data.collate_fn)

    config = gen_config(args.fine_tune_mode, args.lora_rank, args.lora_svd_init, args.lora_pretrained,
                        args.use_siar, args.siar_lamb, args.siar_eps)
    
    saved_checkpoint = torch.load(args.load_checkpoint_path)
    model = SimilarityBERT(config, saved_checkpoint['bert'] , saved_checkpoint['projector'])
    
    if args.use_bagging and BAGG_ADD_NOISE:
        model = add_relative_weight_noise(model, noise_std=0.1) # the std value is arbitrary, not tuned

    model.to(device)
    print(f"training device, {device}\n")

    optimizer = AdamW([
    {'params': [param for name, param in model.named_parameters() if 'bert' in name], 'lr': args.lr_bert},
    {'params': [param for name, param in model.named_parameters() if 'bert' not in name], 'lr': args.lr_class},
    ], weight_decay=args.weight_decay)

    best_dev_corr = 0.0
    best_dev_preds = None
    best_test_preds = None # test preds for corresponding to best eval epoch

    for epoch in range(args.epochs):
        model.train()
        model.to(device)
        train_loss = 0
        num_batches = 0
        for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            sts_ids1, sts_mask1, sts_ids2, sts_mask2, sts_scores = (batch['token_ids_1'], batch['attention_mask_1'],
                                                                        batch['token_ids_2'], batch['attention_mask_2'], 
                                                                        batch['labels'])
            batch_size = sts_ids1.size(0) # effective batch size 
            sts_ids1 = sts_ids1.to(device)
            sts_mask1 = sts_mask1.to(device)
            sts_ids2 = sts_ids2.to(device)
            sts_mask2 = sts_mask2.to(device)
            sts_scores = sts_scores.to(device).float()
                      
            optimizer.zero_grad()
            sts_logits = model.predict_similarity(sts_ids1, sts_mask1, sts_ids2, sts_mask2)
            loss = F.mse_loss(sts_logits.view(-1), sts_scores.view(-1), reduction='sum') / batch_size
            if args.use_siar:
                loss += args.siar_lamb * model.siar_reg_loss_similarity(sts_ids1, sts_mask1, sts_ids2, sts_mask2)/batch_size
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1
        
        train_loss = train_loss / (num_batches)

        train_corr, *_  = model_eval_sts(train_dataloader, model, device)
        dev_corr, dev_preds, dev_sts_sent_ids = model_eval_sts(dev_dataloader, model, device)
        test_preds, test_sts_sent_ids = model_eval_test_sts(test_dataloader, model, device)

        if dev_corr > best_dev_corr:
            best_dev_corr = dev_corr
            best_dev_preds = dev_preds
            best_test_preds = test_preds
            if args.save_checkpoints:
                save_checkpoint(model, config, args.dump_checkpoint_path)

        if not args.use_bagging:
            print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train corr :: {train_corr :.3f}, dev corr :: {dev_corr :.3f}")

    print(f'Best dev corr, {best_dev_corr}')
    dev_pred_dict = {sent_id: best_dev_preds[i] for i, sent_id in enumerate(dev_sts_sent_ids)}
    test_pred_dict = {sent_id: best_test_preds[i] for i, sent_id in enumerate(test_sts_sent_ids)}
    
    return dev_pred_dict, test_pred_dict

def train_bagg(args, train_data, dev_data, test_data, n_models):
    dev_labels = {x[3]: x[2] for x in dev_data}
    dev_pred_dicts, test_pred_dicts = [], []
    assert args.save_checkpoints == False, "Cannot save checkpoint bagging models, too much storage needed"

    for _ in tqdm(range(n_models)):
        train_data_model = random.choices(train_data, k=len(train_data)) # bootstrap
        dev_preds, test_preds = train(args, train_data_model, dev_data, test_data)
        dev_pred_dicts.append(dev_preds)
        test_pred_dicts.append(test_preds)

    dev_preds_out = pred_bagg(dev_pred_dicts, labels=dev_labels, corr_filepath=f"predictions/corr_{args.dump_checkpoint_path.replace('pt', 'txt')}")
    test_preds_out = pred_bagg(test_pred_dicts)
    return dev_preds_out, test_preds_out    

def pred_bagg(preds, labels=None, corr_filepath=None):
    # prediction: average across ensamble predictions
    preds_out = {}
    sent_ids = [k for k in preds[0].keys()]
    for sent_id in sent_ids:
        preds_ = [pred[sent_id] for pred in preds]
        preds_out[sent_id] = np.mean(preds_)
    
    if labels:
        labels_ = [v for k, v in labels.items()]
        pred_ = [preds_out[k] for k in labels.keys()]
        pearson_mat = np.corrcoef(pred_, labels_)
        corr = pearson_mat[1][0]
        print(f'\n Bagging corr on dev set, {corr}')
        with open(corr_filepath, 'w') as file:
            file.write(f'{corr}\n')
    
    return preds_out
    
def save_checkpoint(model, config, dump_checkpoint_path):
    save_info = {
        'bert': model.bert.cpu().state_dict(),
        'classifier': model.similarity_classifier.cpu().state_dict(),
        'config': config
    }
    torch.save(save_info, dump_checkpoint_path)
    print(f"save the model checkpoint to {dump_checkpoint_path}")

def write_pred_file(pred_dict, filepath):
    with open(filepath, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in pred_dict.items():
                f.write(f"{p} , {s} \n")      

        
def gen_config(fine_tune_mode, lora_rank=None, lora_svd_init=None, lora_pretrained=False,
               use_siar=False, siar_lamb=None, siar_eps=None):
    config = {
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
    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")
    

    # included for load_multitask_data, not used
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")
    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
    ##############################################################################

    parser.add_argument("--load_checkpoint_path", type=str, default="sts_full-model-siar-0.1-0.1-1-1e-05-0.0001-multitask.pt")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--fine-tune-mode", type=str,
                        choices=('last-linear-layer', 'full-model', 'lora'), default="full-model")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output-ft")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output-ft")
    
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

    _, _, _, sts_train_data = load_multitask_data(args.sst_train, args.para_train, args.sts_train, split='train')
    _, _, _, sts_dev_data = load_multitask_data(args.sst_dev, args.para_dev, args.sts_dev, split='dev')
    _, _, _, sts_test_data= load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

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
    
    args.dump_checkpoint_path = f'ft-{bagg_details}{siar_details}{lora_details}-{args.epochs}{args.load_checkpoint_path}' # Save path.
    args.sts_dev_out = f"{args.sts_dev_out}{siar_details}{lora_details}-{args.epochs}-{args.lr_bert}-{args.lr_class}{bagg_details}-ft.csv"
    args.sts_test_out = f"{args.sts_test_out}{siar_details}{lora_details}-{args.epochs}-{args.lr_bert}-{args.lr_class}{bagg_details}-ft.csv"
    
    print("=== Finetuning STS BERT ===")
    
    if args.use_bagging:
        TQDM_DISABLE = True
        dev_pred_dict, test_pred_dict = train_bagg(args, sts_train_data, sts_dev_data, sts_test_data, args.n_models_bagging)
    else:
        dev_pred_dict, test_pred_dict = train(args, sts_train_data, sts_dev_data, sts_test_data)
    
    write_pred_file(dev_pred_dict, args.sts_dev_out)
    write_pred_file(test_pred_dict, args.sts_test_out)
