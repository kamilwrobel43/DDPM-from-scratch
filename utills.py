import torch

def load_checkpoint(ckpt_path, model, optimizer, ema, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    new_model_state_dict = {}
    for key, value in checkpoint['model'].items():
        if key.startswith("_orig_mod."):
            new_key = key.replace("_orig_mod.", "") 
        else:
            new_key = key
        new_model_state_dict[new_key] = value
        
    model.load_state_dict(new_model_state_dict)
    
    optimizer.load_state_dict(checkpoint['optimizer'])
    ema.shadow = checkpoint['ema']
    