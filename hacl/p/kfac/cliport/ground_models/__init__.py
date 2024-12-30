from .model_full import ModelFull
from .model_abl import ModelABL
from .model_abl import load_domain

def get_model(args):
    if args.structure_mode == 'full':
        Model = ModelFull
    elif args.structure_mode == 'abl':
        Model = ModelABL
    else:
        raise ValueError('Unknown structure mode: {}.'.format(args.structure_mode))

    return Model
