import torch
import csv
import random
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import pickle
import numpy as np

model_name = "google/flan-t5-xxl"
class DataAugmentator():
    def __init__(self, prompt_base):
        self.prompt_base = prompt_base
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'--------\ndevice, {self.device}\n--------')
    
    def generate_paraphrases(self, text, rep):
        input_text = f"{self.prompt_base} {text} </s>"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        input_length = input_ids.shape[1] # input length in number of tokens
        max_length = int(1.5 * input_length + 10)
        num_beam_groups = 10  # Set this to a value larger than 1 for diversity
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                num_beams=10,
                num_return_sequences=rep,
                temperature=1.0, # todo, modify, or try different values
                top_k=50,
                top_p=0.95,
                diversity_penalty=1.25,
                num_beam_groups=num_beam_groups,
                no_repeat_ngram_size=2,
                early_stopping=True,
                do_sample=False  
            )

        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    def augment_dataset(self, text, labels, rep=5):
        # Augment training data
        # Get rep paraphrases for example in the dataset

        data_out = []
        for i, s in tqdm(enumerate(text)):
            paraphrases = self.generate_paraphrases(s, rep)
            data_out.append((s, labels[i], paraphrases))
        
        return data_out

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
    idxs = np.random.randint(0, len(train_data)-1, 1750) # 1750 samples
    sample_data = [train_data[i] for i in idxs]
    sents = [x[0] for x in sample_data]
    labels = [x[1] for x in sample_data]
    
    data_aug = DataAugmentator("paraphrase the following excerpt from a movie review:")
    data_augmented = data_aug.augment_dataset(sents, labels, 5)

    with open('ids-sst-train-aug.pkl', 'wb') as f:
        pickle.dump(data_augmented, f)

    # run the following to get a sense of the functionality
    """
    for sent_id in [random.randint(0, len(sents) - 1) for _ in range(25)]:
        s = sents[sent_id]
        print(f'Sentence: {s}\n')
        print('Generated paraphrases:\n')
        paraphrases = data_aug.generate_paraphrases(s, 5)
        for p in paraphrases:
            print(p, '\n')
        print("-"*40, '\n')
    """

        
