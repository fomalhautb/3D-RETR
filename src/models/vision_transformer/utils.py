from torch import nn


def load_state_dict_partially(model, state_dict):
    own_state = model.state_dict()

    for name, param in state_dict.items():
        if name not in own_state:
            print(f'Ignored parameter "{name}" on loading')
            continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state[name].copy_(param)
        except RuntimeError:
            print(f'Ignored parameter "{name}" on loading')
