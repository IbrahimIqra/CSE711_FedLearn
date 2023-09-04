import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import prepare_dataset
from client import generate_client

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

    #5. Start Simulation

    #6. Saving the result

    pass



if __name__ == '__main__':
    main()