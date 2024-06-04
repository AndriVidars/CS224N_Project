'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
#from optimizer import AdamW
from transformers import AdamW
from tqdm import tqdm
import math

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask


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


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        if config.fine_tune_mode == 'lora':
            self.bert = BertModel.from_pretrained('bert-base-uncased',
                                                  use_lora = True,
                                                  lora_rank = config.lora_rank,
                                                  lora_svd_init = config.lora_svd_init)
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')

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
        self.sentiment_classifier = nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)

        # TODO, tune the proj, is it to much to have two activations(and hidden layers)
        self.paraphrase_proj = nn.Sequential(
            nn.Linear(BERT_HIDDEN_SIZE, int(BERT_HIDDEN_SIZE / 2)),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(int(BERT_HIDDEN_SIZE / 2), int(BERT_HIDDEN_SIZE / 4)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(BERT_HIDDEN_SIZE / 4), int(BERT_HIDDEN_SIZE / 8)),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.paraphrase_classifier = nn.Linear(int(BERT_HIDDEN_SIZE / 8) * 2, 1)
        
        self.similarity_proj = nn.Sequential(
            nn.Linear(BERT_HIDDEN_SIZE, int(BERT_HIDDEN_SIZE / 2)),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(int(BERT_HIDDEN_SIZE / 2), int(BERT_HIDDEN_SIZE / 8)),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # SIAR parameters
        self.siar_lamb = config.siar_lamb
        self.siar_eps = config.siar_eps

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs['last_hidden_state'][:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        return cls_embedding
    
    # SIAR =========================
    def forward_with_noise(self, input_ids, attention_mask):
        'Produces noisy embeddings for the sentences, used for SIAR.'
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs['last_hidden_state'][:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        return cls_embedding + self.siar_eps * torch.randn_like(cls_embedding)
    # ==============================

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        cls_embedding = self.forward(input_ids, attention_mask)
        logits = self.sentiment_classifier(cls_embedding)
        return logits

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        cls_embedding_1 = self.paraphrase_proj(self.forward(input_ids_1, attention_mask_1))
        cls_embedding_2 = self.paraphrase_proj(self.forward(input_ids_2, attention_mask_2))
        cls_embedding = torch.cat([cls_embedding_1, cls_embedding_2], dim=1)

        logits = self.paraphrase_classifier(cls_embedding)
        return logits


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        cls_embedding_1 = self.similarity_proj(self.forward(input_ids_1, attention_mask_1))
        cls_embedding_2 = self.similarity_proj(self.forward(input_ids_2, attention_mask_2))
        logits = 5 * F.cosine_similarity(cls_embedding_1, cls_embedding_2, dim=1).unsqueeze(1)

        return logits
    
    # SIAR =========================
    def predict_sentiment_siar(self, input_ids, attention_mask):
        cls_embedding = self.forward_with_noise(input_ids, attention_mask)
        logits = self.sentiment_classifier(cls_embedding)
        return logits

    def predict_paraphrase_siar(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        cls_embedding_1 = self.paraphrase_proj(self.forward_with_noise(input_ids_1, attention_mask_1))
        cls_embedding_2 = self.paraphrase_proj(self.forward_with_noise(input_ids_2, attention_mask_2))
        cls_embedding = torch.cat([cls_embedding_1, cls_embedding_2], dim=1)

        logits = self.paraphrase_classifier(cls_embedding)
        return logits

    def predict_similarity_siar(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        cls_embedding_1 = self.similarity_proj(self.forward_with_noise(input_ids_1, attention_mask_1))
        cls_embedding_2 = self.similarity_proj(self.forward_with_noise(input_ids_2, attention_mask_2))
        logits = 5 * F.cosine_similarity(cls_embedding_1, cls_embedding_2, dim=1).unsqueeze(1)

        return logits
    
    def siar_reg_loss_sentiment(self, input_ids, attention_mask):
        '''SIAR regularization loss for sentiment classification task.'''
        logits = self.predict_sentiment(input_ids, attention_mask)
        perturbed_logits = self.predict_sentiment_siar(input_ids, attention_mask) # uses the noisy embeddings
        loss = F.kl_div(F.log_softmax(logits, dim=1), F.softmax(perturbed_logits, dim=1), reduction='batchmean')
        return loss
    
    def siar_reg_loss_paraphrase(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        '''SIAR regularization loss for paraphrase detection task.'''
        logits = self.predict_paraphrase(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        perturbed_logits = self.predict_paraphrase_siar(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        loss = F.kl_div(F.log_softmax(logits, dim=1), F.softmax(perturbed_logits, dim=1), reduction='batchmean')
        return loss
    
    def siar_reg_loss_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        '''SIAR regularization loss for similarity task.'''
        logits = self.predict_similarity(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        perturbed_logits = self.predict_similarity_siar(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        loss = F.kl_div(F.log_softmax(logits, dim=1), F.softmax(perturbed_logits, dim=1), reduction='batchmean')

        return loss
    # ==============================


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

def save_checkpoint(checkpoint_path, bert, classifier=None, projector=None):
    save_info = {
        'bert': bert.cpu().state_dict(),
        'classifier': classifier.cpu().state_dict() if classifier else None,
        'projector': projector.cpu().state_dict() if projector else None
    }
        
    torch.save(save_info, checkpoint_path)
    print(f"save the model checkpoint to {checkpoint_path}") 

def train_multitask(args):
    '''Train MultitaskBERT on all three tasks simultaneously.'''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    # Load training and validation data for all tasks
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(args.sst_train, args.para_train, args.sts_train, split='train')
    sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev, args.para_dev, args.sts_dev, split='dev')

    sst_train_data = random.sample(sst_train_data, int(len(sst_train_data) * args.train_ratio_sst))
    para_train_data = random.sample(para_train_data, int(len(para_train_data) * args.train_ratio_para))
    sts_train_data = random.sample(sts_train_data, int(len(sts_train_data) * args.train_ratio_sts))
    num_batches_sst = math.ceil(len(sst_train_data) / args.batch_size_sst)
    num_batches_para = math.ceil(len(para_train_data) / args.batch_size_para)
    num_batches_sts = math.ceil(len(sts_train_data) / args.batch_size_sts)
    print(f"Num samples, (sst, {len(sst_train_data)}), (para, {len(para_train_data)}), (sts, {len(sts_train_data)})")
    print(f"Num batches in epoch, (sst, {num_batches_sst}), (para, {num_batches_para}), (sts, {num_batches_sts})")

    assert num_batches_para >= num_batches_sst and num_batches_para >= num_batches_sts # at this stage, we are assuming this, can change later
    para_sst_ratio = math.ceil(num_batches_para / num_batches_sst)
    para_sts_ratio = math.ceil(num_batches_para / num_batches_sts)

    # Create the data and its corresponding datasets and dataloader
    # SST
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size_sst, collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size_sst, collate_fn=sst_dev_data.collate_fn)

    # paraphrase detection
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size_para, collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size_para, collate_fn=para_dev_data.collate_fn)

    # STS
    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size_sts, collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size_sts, collate_fn=sts_dev_data.collate_fn)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': BERT_HIDDEN_SIZE,
              'fine_tune_mode': args.fine_tune_mode,
              'lora_rank': args.lora_rank,
              'lora_svd_init': args.lora_svd_init,
              'siar_lamb': args.siar_lamb,
              'siar_eps': args.siar_eps,}
    
    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model.to(device)
    print(f"training device, {device}\n")

    # Init optimizer
    optimizer = AdamW([
    {'params': [param for name, param in model.named_parameters() if 'bert' in name], 'lr': args.lr_bert},
    {'params': [param for name, param in model.named_parameters() if 'bert' not in name], 'lr': args.lr_class},
    ], weight_decay=args.weight_decay)

    best_dev_acc = 0.0
    
    # run for the specified number of epochs
    for epoch in range(args.epochs):
        model.to(device)
        model.train()
        train_loss = 0.0
        num_batches = 0
        miss_iter = 0 # number of times the backward fails(due to memory)
        sst_train_dataloader_iter = iter(sst_train_dataloader)
        sts_train_dataloader_iter = iter(sts_train_dataloader)
        for i, para_batch, in tqdm(enumerate(para_train_dataloader), desc=f'train-{epoch}', disable=TQDM_DISABLE):
            # paraphrase detection task
            para_ids1, para_mask1, para_ids2, para_mask2, para_labels = (para_batch['token_ids_1'], 
                                                                         para_batch['attention_mask_1'],
                                                                          para_batch['token_ids_2'], 
                                                                          para_batch['attention_mask_2'], 
                                                                          para_batch['labels'])
            para_batch_size = para_ids1.size(0) # effective batch size
            para_ids1 = para_ids1.to(device)
            para_mask1 = para_mask1.to(device)
            para_ids2 = para_ids2.to(device)
            para_mask2 = para_mask2.to(device)
            para_labels = para_labels.to(device).float()
            
            optimizer.zero_grad()
            total_loss = 0
            all_miss = True # backward on all tasks fails (due to memory constraints)
            para_logits = None
            para_loss = 0
            try:
                para_logits = model.predict_paraphrase(para_ids1, para_mask1, para_ids2, para_mask2)
                para_loss = F.binary_cross_entropy_with_logits(para_logits.view(-1), para_labels.view(-1), reduction='sum') / para_batch_size
                 # SIAR =========
                if args.use_siar: 
                    para_loss += args.siar_lamb + model.siar_reg_loss_paraphrase(para_ids1, para_mask1, para_ids2, para_mask2)/ para_batch_size
                # =============== 
                total_loss += para_loss
                if args.backward_sep:
                    para_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    all_miss = False
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
            
            del para_ids1, para_mask1, para_ids2, para_mask2, para_labels, para_logits, para_loss

            if (i + 1) % para_sst_ratio == 0 or (i + 1) == num_batches_para:
                # SST task
                sst_batch = next(sst_train_dataloader_iter)
                sst_ids, sst_mask, sst_labels = sst_batch['token_ids'], sst_batch['attention_mask'],sst_batch['labels']
                sst_batch_size = sst_ids.size(0)
                sst_ids = sst_ids.to(device)
                sst_mask = sst_mask.to(device)
                sst_labels = sst_labels.to(device)
                sst_logits = None
                sst_loss = 0
                try:
                    sst_logits = model.predict_sentiment(sst_ids, sst_mask)
                    sst_loss = F.cross_entropy(sst_logits, sst_labels.view(-1), reduction='sum') / sst_batch_size
                    # SIAR =========
                    if args.use_siar:
                        sst_loss += args.siar_lamb * model.siar_reg_loss_sentiment(sst_ids, sst_mask)/ sst_batch_size
                    # =============== 
                    total_loss += sst_loss
                    if args.backward_sep:
                        sst_loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        all_miss = False
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                
                del sst_ids, sst_mask, sst_labels, sst_logits, sst_loss

            if (i + 1) % para_sts_ratio == 0 or (i + 1) == num_batches_para:
                # STS task
                sts_batch = next(sts_train_dataloader_iter)
                sts_ids1, sts_mask1, sts_ids2, sts_mask2, sts_scores = (sts_batch['token_ids_1'], sts_batch['attention_mask_1'],
                                                                        sts_batch['token_ids_2'], sts_batch['attention_mask_2'], 
                                                                        sts_batch['labels'])
                sts_batch_size = sts_ids1.size(0)
                sts_ids1 = sts_ids1.to(device)
                sts_mask1 = sts_mask1.to(device)
                sts_ids2 = sts_ids2.to(device)
                sts_mask2 = sts_mask2.to(device)
                sts_scores = sts_scores.to(device).float()
                sts_logits = None
                sts_loss = 0
                try:
                    sts_logits = model.predict_similarity(sts_ids1, sts_mask1, sts_ids2, sts_mask2)
                    sts_loss = F.mse_loss(sts_logits.view(-1), sts_scores.view(-1), reduction='sum') / sts_batch_size
                    # SIAR =========
                    if args.use_siar:
                        sts_loss += args.siar_lamb * model.siar_reg_loss_similarity(sts_ids1, sts_mask1, sts_ids2, sts_mask2)/sts_batch_size
                    # =============== 
                    total_loss += sts_loss
                    if args.backward_sep:
                        sts_loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        all_miss = False
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                
                del sts_ids1, sts_mask1, sts_ids2, sts_mask2, sts_scores, sts_logits, sts_loss
                
            # Should we use weighted sum? I think this should suffice
            if not args.backward_sep and total_loss != 0:
                try:
                    total_loss.backward()
                    optimizer.step()
                    train_loss += total_loss.item()
                    num_batches += 1
                except:
                    miss_iter += 1
                    torch.cuda.empty_cache()
            else:
                if all_miss:
                    miss_iter += 1
                else:
                    train_loss += total_loss.item()
                    num_batches += 1

        print(f'Number of missed iters in epoch, {miss_iter}')
        train_loss = train_loss / num_batches

        # Evaluate on SST dev sets using the multitask model (need to evaluate on the other datasets later)
        (sst_train_acc,_, _,para_train_acc, _,_,sts_train_corr, _,_) = model_eval_multitask(sst_train_dataloader, 
                                                                para_train_dataloader,
                                                                sts_train_dataloader,
                                                                model, device)

        (sst_dev_acc,_,_,para_dev_acc, _,_,sts_dev_corr, _,_) = model_eval_multitask(sst_dev_dataloader, 
                                                                para_dev_dataloader, 
                                                                sts_dev_dataloader, 
                                                                model, device)
        
        # accuracy is average of accuracy the three tasks
        # other options: check that the accuracy of each task improves, 
        # check if the accuracy of at least one task improves

        train_acc = (sst_train_acc + para_train_acc + sts_train_corr) / 3
        dev_acc = (sst_dev_acc + para_dev_acc + sts_dev_corr) / 3

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)
            save_checkpoint(f'sst_{args.filepath}', model.bert, model.sentiment_classifier)
            save_checkpoint(f'para_{args.filepath}', model.bert, model.paraphrase_classifier, model.paraphrase_proj)
            save_checkpoint( f'sts_{args.filepath}', model.bert, None, model.similarity_proj)


        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
        
        print(f'Sentiment classification train accuracy: {sst_train_acc:.3f}')
        print(f'Sentiment classification dev accuracy: {sst_dev_acc:.3f}')

        print(f'Paraphrase detection train accuracy: {para_train_acc:.3f}')
        print(f'Paraphrase detection dev accuracy: {para_dev_acc:.3f}')

        print(f'Semantic Textual Similarity train correlation: {sts_train_corr:.3f}')
        print(f'Semantic Textual Similarity dev correlation: {sts_dev_corr:.3f}')
    



def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size_sst,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size_sst,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size_para,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size_para,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size_sts,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size_sts,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv") # TODO use full dev set
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--fine-tune-mode", type=str,
                        choices=('last-linear-layer', 'full-model', 'lora'), default="full-model")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output-multi")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output-multi")
    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output-multi")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output-multi")
    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output-multi")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output-multi")
    
    # different batch sizes for each task
    parser.add_argument("--batch_size_sst", type=int, default=8) # dont know the memory cap, 32
    parser.add_argument("--batch_size_para", type=int, default=16) # 64
    parser.add_argument("--batch_size_sts", type=int, default=8) # 32
    parser.add_argument("--train_ratio_sst", type=float, default=1.0)
    parser.add_argument("--train_ratio_para", type=float, default=0.25) # by default not train on all quora data
    parser.add_argument("--train_ratio_sts", type=float, default=1.0)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)

    parser.add_argument("--lr_bert", type=float, help="learning rate for bert layers, full or lora",
                        default=1e-5)
    parser.add_argument("--lr_class", type=float, help="learning rate for classification layers",
                        default=1e-4)
    parser.add_argument("--lora_rank", type=int, help="rank of lora adapters",
                        default=8)
    parser.add_argument("--lora_svd_init", action='store_true')
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--backward_sep", action='store_true') # do backprop separately for each task

    # SIAR hyperparameters
    parser.add_argument("--use_siar", action='store_true')
    parser.add_argument("--siar_lamb", type=float, default=0.1)
    parser.add_argument("--siar_eps", type=float, default=0.1)


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print(f'{torch.cuda.is_available()=}')

    if args.fine_tune_mode == 'lora':
        lora_details = f'-lora-{args.lora_rank}-{args.lora_svd_init}'
    else:
        lora_details = ''
    
    if args.use_siar:
        siar_details = f'-siar-{args.siar_lamb}-{args.siar_eps}'
    else:
        siar_details = ''

    # hacky, sorry
    args.filepath = f'{args.fine_tune_mode}{siar_details}{lora_details}-{args.epochs}-{args.lr_bert}-{args.lr_class}-multitask.pt' # Save path.
    args.sst_dev_out = f"{args.sst_dev_out}{siar_details}{lora_details}-{args.epochs}-{args.lr_bert}-{args.lr_class}.csv"
    args.sst_test_out = f"{args.sst_test_out}{siar_details}{lora_details}-{args.epochs}-{args.lr_bert}-{args.lr_class}.csv"
    args.para_dev_out = f"{args.para_dev_out}{siar_details}{lora_details}-{args.epochs}-{args.lr_bert}-{args.lr_class}.csv"
    args.para_dev_out = f"{args.para_test_out}{siar_details}{lora_details}-{args.epochs}-{args.lr_bert}-{args.lr_class}.csv"
    args.sts_dev_out = f"{args.sts_dev_out}{siar_details}{lora_details}-{args.epochs}-{args.lr_bert}-{args.lr_class}.csv"
    args.sts_dev_out = f"{args.sts_test_out}{siar_details}{lora_details}-{args.epochs}-{args.lr_bert}-{args.lr_class}.csv"
    
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    print("=== Training Multitask BERT ===")
    train_multitask(args)
    print("=== Testing Multitask BERT ===")
    test_multitask(args)
