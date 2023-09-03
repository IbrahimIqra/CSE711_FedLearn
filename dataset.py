import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST


def get_mnist(data_path: str = './data'):
    tr = Compose( (ToTensor(),Normalize( mean=(0.1307,),std=(0.3081) )) )
    
    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)

    return trainset, testset

def prepare_dataset(num_partition: int, batch_size: int, val_ratio: float = 0.1):
    
    trainset, testset = get_mnist()

    #split trainset into 'num_partitions' trainset
    num_images = len(trainset) // num_partition
    
    #creating a list which dictates how many data each client will get
    partition_len = [num_images] * num_partition

    #splitting the dataset into partitions
    trainsets = random_split(
                        dataset=trainset,
                        lengths=partition_len,
                        generator=torch.Generator().manual_seed(2023)
                        )

    train_loaders = []
    val_loaders = []
    
    #Create dataloaders with train+val support
    for client_trainset in trainsets:
        num_total = len(client_trainset)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        client_train, client_val = random_split(
                                        client_trainset,
                                        [num_train, num_val],
                                        generator=torch.Generator().manual_seed(2023)
                                        )

        train_loaders.append(
                        DataLoader(
                            dataset=client_train,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2
                            )
                        )
        
        val_loaders.append(
                        DataLoader(
                            dataset=client_val,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2
                            )
                        )

    test_loader = DataLoader(dataset=testset, batch_size=128)

    return train_loaders, val_loaders, test_loader