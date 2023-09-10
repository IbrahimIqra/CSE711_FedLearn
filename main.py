import hydra
from omegaconf import DictConfig, OmegaConf

import flwr as fl

from dataset import prepare_dataset
from client import generate_client
from server import get_on_fit_config, get_evaluate_fn

@hydra.main(config_path="conf",config_name="basic",version_base=None)
def main(cfg: DictConfig):

    #1. Parse config and get experiment output dir
    print("=====Config:=====\n")
    print(OmegaConf.to_yaml(cfg))
    print("=================\n\n")


    #2. Preparing Dataset
    train_loaders, \
        val_loaders, \
            test_loader \
                = prepare_dataset(
                    num_partition=cfg.num_clients,
                    batch_size=cfg.batch_size
                )

    print('-------Loaders Information-------')
    print( f"Total Train_Loaders (eqauls to the amount of clients): {len(train_loaders)},\
          \nEach Train_Loader has {len(train_loaders[0].dataset)} training data (for each clinents)\
          \nTotal Val_Loaders (eqauls to the amount of clients): {len(val_loaders)},\
          \nEach Val_Loader has {len(val_loaders[0].dataset)} validation data (for each clinents)\
          \nTotal Test_Loaders {len(test_loader)=}")
    print('---------------------------------\n\n')

    #3. Defining clients
    client = generate_client(train_loaders= train_loaders,
                            val_loaders= val_loaders,
                            num_classes= cfg.num_classes
                            )


    #4. Define the strategy
    strategy = fl.server.strategy.FedAvg(fraction_fit=0.00001,
                                         min_fit_clients=cfg.num_clients_per_round_fit,
                                         fraction_evaluate=0.00001,
                                         min_evaluate_clients=cfg.num_clients_per_round_eval,
                                         min_available_clients=cfg.num_clients,
                                        
                                         #local fit for clients
                                         on_fit_config_fn=get_on_fit_config(cfg.config_fit),
                                         #local evaluation for client
                                         evaluate_fn=get_evaluate_fn(cfg.num_classes,
                                                                     test_loader=test_loader)
                                        )

    #5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn = client,
        num_clients = cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy
    )

    #6. Saving the result

    pass



if __name__ == '__main__':
    main()