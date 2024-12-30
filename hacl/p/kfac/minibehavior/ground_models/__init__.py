from .model_full import ModelFull
from .model_abl import ModelABL


def get_model(args):
    if args.structure_mode == 'full':
        Model = ModelFull
    elif args.structure_mode == 'abl':
        Model = ModelABL
    else:
        raise ValueError('Unknown structure mode: {}.'.format(args.structure_mode))

    return Model
