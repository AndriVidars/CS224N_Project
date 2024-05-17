import csv
import random
import torch
from transformers import MarianMTModel, MarianTokenizer

# this is not good, don't use

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'--------\ndevice, {device}\n--------')

# Load English-to-French model and tokenizer
model_name_en_to_fr = 'Helsinki-NLP/opus-mt-en-fr'
tokenizer_en_to_fr = MarianTokenizer.from_pretrained(model_name_en_to_fr)
model_en_to_fr = MarianMTModel.from_pretrained(model_name_en_to_fr).to(device)

# Load French-to-English model and tokenizer
model_name_fr_to_en = 'Helsinki-NLP/opus-mt-fr-en'
tokenizer_fr_to_en = MarianTokenizer.from_pretrained(model_name_fr_to_en)
model_fr_to_en = MarianMTModel.from_pretrained(model_name_fr_to_en).to(device)

def back_translate(text, model_translate, model_back_translate, tokenizer_source, tokenizer_dest):
    # Translate from source language to target language
    inputs = tokenizer_source(text, return_tensors="pt", padding=True, truncation=True).to(device)
    translated_tokens = model_translate.generate(**inputs)
    translated_text = tokenizer_source.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    # Translate back from target language to source language
    inputs_back = tokenizer_dest(translated_text, return_tensors="pt", padding=True, truncation=True).to(device)
    back_translated_tokens = model_back_translate.generate(**inputs_back)
    back_translated_text = tokenizer_dest.batch_decode(back_translated_tokens, skip_special_tokens=True)[0]

    return back_translated_text

def read_sentiment_train(filename):
    data = []
    with open(filename, 'r') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            sent = record['sentence'].lower().strip()
            sent_id = record['id'].lower().strip()
            label = int(record['sentiment'].strip())
            data.append((sent, label,sent_id))
    return data

if __name__ == "__main__":
    file = 'data/ids-sst-train.csv'
    train_data = read_sentiment_train(file)
    labels = [x[1] for x in train_data]
    sents = [x[0] for x in train_data]

    for sent_id in [random.randint(0, len(sents) - 1) for _ in range(25)]:
        s = sents[sent_id]
        print(f'Sentence: {s}\n')
        back_s = back_translate(s, model_en_to_fr, model_fr_to_en, tokenizer_en_to_fr, tokenizer_fr_to_en)
        print(f'Backtranslated: {back_s}')
        print("-"*40, '\n')

