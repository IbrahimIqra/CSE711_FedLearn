from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST


def get_mnist(data_path: str = './date'):
    tr = Compose( (ToTensor(),Normalize( mean=(0.1307,),std=(0.3081) )) )
    
    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)

    return trainset, testset

def prepare():

    train_set, test_set = get_mnist()