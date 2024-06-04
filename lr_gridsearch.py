from types import SimpleNamespace
from classifier import train
import pickle

def get_config(lr_bert, lr_class, weight_decay, fine_tune_mode):
    return SimpleNamespace(
        lr_bert=lr_bert,
        lr_class=lr_class,
        use_gpu=True,
        epochs=4, # maybe add to params
        batch_size=64,
        hidden_dropout_prob=0.3,
        train='data/ids-sst-train.csv', # modify to use augmented data
        dev='data/ids-sst-dev.csv',
        fine_tune_mode=fine_tune_mode,
        lora_rank=16, # maybe add to params
        lora_svd_init=True,
        weight_decay=weight_decay,
        save_models = False,
        train_verbose = False
    )

def gridsearch(params_list):
    results_out = [] # list tuple, (params, acc)
    for p in params_list:
        config = get_config(p['lr_bert'], p['lr_class'],
                             p['weight_decay'], p['fine_tune_mode'])
        acc = train(config)
        results_out.append((p, acc))
    
    return results_out


if __name__ == '__main__':
    #lr_bert = [5e-6, 1e-5]
    #lr_class = [1e-5, 5e-5, 1e-4]

    lr_bert = [5e-6, 8e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
    lr_class = [1e-5, 5e-5, 7.5e-5, 1e-4]
    
    params = [{'lr_bert': lb, 'lr_class': lc, 'weight_decay':0.0, 'fine_tune_mode': 'full-model'} 
              for lb in lr_bert for lc in lr_class]

    results = gridsearch(params)
    # todo, write results to file
    print(results)

    with open('grid.pkl', 'wb') as f:
        pickle.dump(results, f)
