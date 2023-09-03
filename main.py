import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import prepare_dataset

@hydra.main(config_path="conf",config_name="basic",version_base=None)
def main(cfg: DictConfig):

    #1. Parse config and get experiment output dir
    print(OmegaConf.to_yaml(cfg))

    #2. Preparing Dataset
    train_loaders, \
        validation_loaders, \
            test_loader \
                = prepare_dataset(
                    num_partition=cfg.num_clients,
                    batch_size=cfg.batch_size
                )

    print( f"{len(train_loaders)=}\n{len(validation_loaders)=}\n{len(test_loader)=}" )
    print('----------')
    print( f"{len(train_loaders[0])=}\n{len(validation_loaders[0])=}\n{len(test_loader)=}" )
    #3. Defining clients

    #4. Define the strategy

    #5. Start Simulation

    #6. Saving the result

    pass



if __name__ == '__main__':
    main()