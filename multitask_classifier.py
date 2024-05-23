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
from optimizer import AdamW
from tqdm import tqdm

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

        # last-linear-layer mode does not require updating BERT paramters.
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
        # You will want to add layers here to perform the downstream tasks.
        ### TODO
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sentiment_classifier = nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        self.paraphrase_classifier = nn.Linear(BERT_HIDDEN_SIZE * 2, 1)
        self.similarity_classifier = nn.Linear(BERT_HIDDEN_SIZE * 2, 1)

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
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
        ### TODO
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
        ### TODO
        cls_embedding_1 = self.forward(input_ids_1, attention_mask_1)
        cls_embedding_2 = self.forward(input_ids_2, attention_mask_2)
        cls_embedding = torch.cat([cls_embedding_1, cls_embedding_2], dim=1)

        logits = self.paraphrase_classifier(cls_embedding)
        return logits


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        cls_embedding_1 = self.forward(input_ids_1, attention_mask_1)
        cls_embedding_2 = self.forward(input_ids_2, attention_mask_2)
        cls_embedding = torch.cat([cls_embedding_1, cls_embedding_2], dim=1)

        logits = self.similarity_classifier(cls_embedding)
        return logits




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

def train_multitask(args):
    '''Train MultitaskBERT on all three tasks simultaneously.'''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    # Load training and validation data for all tasks
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(args.sst_train, args.para_train, args.sts_train, split='train')
    sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev, args.para_dev, args.sts_dev, split='dev')

    # Create the data and its corresponding datasets and dataloader
    # SST
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sst_dev_data.collate_fn)

    # paraphrase detection
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=para_dev_data.collate_fn)

    # STS
    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sts_dev_data.collate_fn)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': BERT_HIDDEN_SIZE,
              'fine_tune_mode': args.fine_tune_mode,
              'lora_rank': args.lora_rank,
              'lora_svd_init': args.lora_svd_init
              }
    
    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model.to(device)
    print(f"training device, {device}\n")

    # Init optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    best_dev_acc = 0.0

    # run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0
        for sst_batch, para_batch, sts_batch in tqdm(zip(sst_train_dataloader, para_train_dataloader, sts_train_dataloader), 
                                                     desc=f'train-{epoch}', disable=TQDM_DISABLE):
            # SST task
            sst_ids, sst_mask, sst_labels = sst_batch['token_ids'], sst_batch['attention_mask'],sst_batch['labels']
            sst_ids = sst_ids.to(device)
            sst_mask = sst_mask.to(device)
            sst_labels = sst_labels.to(device)

            # paraphrase detection task
            para_ids1, para_mask1, para_ids2, para_mask2, para_labels = (para_batch['token_ids_1'], 
                                                                         para_batch['attention_mask_1'],
                                                                          para_batch['token_ids_2'], 
                                                                          para_batch['attention_mask_2'], 
                                                                          para_batch['labels'])
            para_ids1 = para_ids1.to(device)
            para_mask1 = para_mask1.to(device)
            para_ids2 = para_ids2.to(device)
            para_mask2 = para_mask2.to(device)
            para_labels = para_labels.to(device).float()

            # STS task
            sts_ids1, sts_mask1, sts_ids2, sts_mask2, sts_scores = (sts_batch['token_ids_1'], sts_batch['attention_mask_1'],
                                                                    sts_batch['token_ids_2'], sts_batch['attention_mask_2'], 
                                                                    sts_batch['labels'])
            sts_ids1 = sts_ids1.to(device)
            sts_mask1 = sts_mask1.to(device)
            sts_ids2 = sts_ids2.to(device)
            sts_mask2 = sts_mask2.to(device)
            sts_scores = sts_scores.to(device).float()

            optimizer.zero_grad()

            # Total loss function as the sum of the loss for the three tasks
            # SST
            sst_logits = model.predict_sentiment(sst_ids, sst_mask)
            sst_loss = F.cross_entropy(sst_logits, sst_labels.view(-1), reduction='sum') / args.batch_size
            # paraphrase detection
            para_logits = model.predict_paraphrase(para_ids1, para_mask1, para_ids2, para_mask2)
            para_loss = F.binary_cross_entropy_with_logits(para_logits.view(-1), para_labels.view(-1), reduction='sum') / args.batch_size
            # STS
            sts_logits = model.predict_similarity(sts_ids1, sts_mask1, sts_ids2, sts_mask2)
            sts_loss = F.mse_loss(sts_logits.view(-1), sts_scores.view(-1), reduction='sum') / args.batch_size
            # Total loss
            # Should we use weighted sum? 
            loss = sst_loss + para_loss + sts_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

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

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
        print(f'Sentiment classification train accuracy: {sst_train_acc:.3f}')
        print(f'Paraphrase detection train accuracy: {para_train_acc:.3f}')
        print(f'Semantic Textual Similarity train correlation: {sts_train_corr:.3f}')


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

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
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
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    parser.add_argument("--lr_bert", type=float, help="learning rate for bert layers, full or lora",
                        default=1e-3)
    parser.add_argument("--lr_class", type=float, help="learning rate for classification layers",
                        default=1e-3)
    parser.add_argument("--lora_rank", type=int, help="rank of lora adapters",
                        default=8)
    parser.add_argument("--lora_svd_init", action='store_true')
    parser.add_argument("--weight_decay", type=float, default=0.0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    if args.fine_tune_mode == 'lora':
        lora_details = f'_{args.lora_rank}_{args.lora_svd_init}'
    else:
        lora_details = ''

    args.filepath = f'{args.fine_tune_mode}{lora_details}-{args.epochs}-{args.lr}-multitask.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    print("=== Training Multitask BERT ===")
    train_multitask(args)
    print("=== Testing Multitask BERT ===")
    test_multitask(args)
