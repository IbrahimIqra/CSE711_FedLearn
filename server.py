from collections import OrderedDict
from omegaconf import DictConfig

import torch

from model import Net, test

def get_on_fit_config(config: DictConfig):

    def fit_config_fn(server_round: int):

        # not needed rn
        # if server_round>50:
        #     lr = config.lr/10

        return {'lr':config.lr,
                'momentum':config.momentum,
                'local_epochs':config.local_epochs
            }
    

    return fit_config_fn


def get_evaluate_fn(num_classes: int, test_loader):

    #follows same structure of evaluate function on fedavg.py file
    def evaluate_fn(server_rounds: int, parameters, config):
        model = Net(num_classes=num_classes)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #copied from client.py set params
        params_dict =  zip(model.state_dict().keys(), params)
        state_dict = OrderedDict( {k:torch.Tensor(v) for k,v in params_dict} )
        model.load_state_dict( state_dict, strict=True )

        loss, acc = test(model, testloader=test_loader, device=device)

        return loss, { 'acc': acc }

    return evaluate_fn