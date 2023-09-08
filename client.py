from collections import OrderedDict
import torch
import flwr as fl

from model import Net, train, test

class FlwrClient(fl.client.NumpyClient):
    def __init__(self, train_loader, val_loader, num_classes):
        super().__init__()

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.model = Net(num_classes)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_params(self, params):
        params_dict =  zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict( {k:torch.Tensor(v) for k,v in params_dict} )

        self.model.load_state_dict( state_dict, strict=True )

    def get_params(self, config: Dict[str, Scalar]):

        p = []
        for _,val in self.model.state_dict().items():
            p.append(val.cpu().numpy())

        return  p

    '''
        here params is a list of nparrays
        represeting current state of the 
        global model

        config is a dictionary
    '''
    def fit(self, params, config):

        #copy the params from server to client local model
        self.set_params(params=params)

        #extract info from config
        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_epochs']

        optim = torch.optim.SGD(self.model.parameters(),
                                lr=lr,
                                momentum=momentum
                                )

        #train the models locally 
        train(net=self.model,
              trainloader=self.train_loader,
              optimizer=optim,

              )

                # params            # soemtimes needed     #metric
        return self.get_params(), len(self.train_loader), {}

    def evaluate(self, params, config: Dict[str, Scalar]):

        self.set_params(params=params)
        
        loss, acc = test(self.model, self.val_loader, self.device)
        
        return float(loss), len(self.val_loader), {'acc': acc}
    

def generate_client(train_loaders, val_loaders, num_classes):

    def client_fn(cid: str):

        return FlwrClient(train_loader=train_loaders[int(cid)],
                          val_loader=val_loaders[int(cid)],
                          num_classes=num_classes
                        )

    return client_fn