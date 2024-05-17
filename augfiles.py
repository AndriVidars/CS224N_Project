import pandas as pd
import random
import string
import pickle

def generate_custom_id(length=24):
    characters = string.ascii_lowercase + string.digits
    return ''.join(random.choices(characters, k=length))

def write_augdata_file(aug_data, source_data_file_path, file_path_out):
    source_df = pd.read_csv(source_data_file_path, delimiter='\t')
    if 'Unnamed: 0' in source_df.columns:
        source_df = source_df.drop(columns=['Unnamed: 0'])

    aug_df = pd.DataFrame(aug_data)
    merged_df = pd.concat([source_df, aug_df], ignore_index=True)
    merged_df.to_csv(file_path_out, sep='\t', index_label='number', index=True)
    # data loader shuffles before training so no need to do that here

def write_ids_sst_aug_file(aug_data, source_data_file_path, aug_data_file_path):
    data = {
        'id': [],
        'sentence':[],
        'sentiment':[]
    }
    for x in aug_data:
        sentiment = x[1]
        for s in x[2]:
            id = generate_custom_id()
            data['id'].append(id)
            data['sentence'].append(s)
            data['sentiment'].append(sentiment)
    
    write_augdata_file(data, source_data_file_path, aug_data_file_path)

if __name__ == '__main__':
    # write aug data file for ids-sst-train
    source_data_file_path = 'data/ids-sst-train.csv'
    aug_data_file_path = 'data/ids-sst-train-aug.csv'

    with open('ids-sst-train-aug.pkl', 'rb') as f:
        aug_data = pickle.load(f)
    
    write_ids_sst_aug_file(aug_data, source_data_file_path, aug_data_file_path)
