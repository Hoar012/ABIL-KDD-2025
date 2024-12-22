from .IL_models import BCModel, ABIL_BCModel, DTModel, ABIL_DTModel

def get_model(model_name):
    model_name = model_name.lower()
    if model_name == "bc":
        Model = BCModel
    elif model_name == "abil-bc":
        Model = ABIL_BCModel
    elif model_name == "dt":
        Model = DTModel
    elif model_name == "abil-dt":
        Model = ABIL_DTModel
    else:
        raise ValueError('Unknown model: {}.'.format(model_name))

    return Model
