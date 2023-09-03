import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf",config_name="basic",version_base=None)
def main(cfg: DictConfig):

    #1. Parse config and get experiment output dir
    print(OmegaConf.to_yaml(cfg))

    #2. Preparing Dataset
    train_loader, validation_loader, test_loader = prepare_dataset()

    #3. Defining clients

    #4. Define the strategy

    #5. Start Simulation

    #6. Saving the result

    pass



if __name__ == '__main__':
    main()