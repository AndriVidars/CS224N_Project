import torch
import torch.nn as nn
import numpy as np
import loralib as lora

def set_nested_attr(obj, name, value):
    names = name.split('.')
    for n in names[:-1]:
        obj = getattr(obj, n)
    setattr(obj, names[-1], value)

def replace_with_lora_layers(model, layer_names, rank=8, svd_init=False):
    for name, module in model.named_modules():
        if name in layer_names and isinstance(module, nn.Linear):
            with torch.no_grad():
                weight = module.weight.data.cpu().numpy()
                lora_layer = lora.Linear(module.in_features, module.out_features, r=rank)
                lora_layer.weight.data = torch.tensor(weight)
                
                if module.bias is not None:
                    lora_layer.bias = nn.Parameter(module.bias.data.clone())
                
                if svd_init:
                    U, S, Vt = np.linalg.svd(weight, full_matrices=False)
                    U_r = U[:, :rank]
                    S_r = S[:rank]
                    Vt_r = Vt[:rank, :]
                    lora_layer.lora_B.data = torch.tensor(U_r * S_r)
                    lora_layer.lora_A.data = torch.tensor(Vt_r)

            set_nested_attr(model, name, lora_layer)