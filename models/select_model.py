
"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""


def define_Model(opt):
    model = opt['model']      # one input: L

    if model == 'plain':
        from models.model_plain import ModelPlain as M
    elif model == 'plain_nhnet':  # two inputs: L, C
        from models.model_plain_nhnet import ModelPlain as M
    elif model == 'plain2':  # two inputs: L, C
        from models.model_plain2 import ModelPlain2 as M
    elif model == 'plain_noparal':  # two inputs: L, C
        from models.model_plain_noparal import ModelPlain as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)
    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
