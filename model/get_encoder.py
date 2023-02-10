import sys
sys.path.append('..')
import torch
from model.encoder import swin_transformer,simplenet,trans_plus_conv,resnet

moco_weight_path = {
    'resnet18':'../',
    'swinplusr18':'../'
}

def build_encoder(arch='resnet18', weights=None, **kwargs):
        
    arch = arch.lower()
    
    if arch.startswith('resnet'):
        backbone = resnet.__dict__[arch](classification=False,**kwargs)
    elif arch.startswith('swin_transformer'):
        backbone = swin_transformer.__dict__[arch](classification=False,**kwargs)
    elif arch.startswith('simplenet'):
        backbone = simplenet.__dict__[arch](**kwargs)
    elif arch.startswith('swinplus'):
        backbone = trans_plus_conv.__dict__[arch](classification=False,**kwargs)
    else:
        raise Exception('Architecture undefined!')

    if weights is not None and isinstance(moco_weight_path[arch], str):
        print('Loading weights for backbone')
        msg = backbone.load_state_dict(
            torch.load(moco_weight_path[arch], map_location=lambda storage, loc: storage)['state_dict'], strict=False)
        if arch.startswith('resnet'):
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
            print(">>>> loaded pre-trained model '{}' ".format(moco_weight_path[arch]))
        print(msg)
    
    return backbone
