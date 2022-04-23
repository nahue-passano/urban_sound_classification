from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def download_mnist_datasets(BATCH_SIZE =  128):
    train_data = datasets.MNIST(
        root = 'dataset',
        download = True,
        train = True,
        transform = ToTensor()  # Takes the image and normalizes between 0 and 1
    )
    validation_data = datasets.MNIST(
        root = 'dataset',
        download = True,
        train = False,
        transform = ToTensor()  # Takes the image and normalizes between 0 and 1
    )

    train_data = DataLoader(train_data, batch_size = BATCH_SIZE)
    validation_data = DataLoader(validation_data, batch_size = BATCH_SIZE)

    return train_data, validation_data



if __name__ == '__main__':
    train_data, _ = download_mnist_datasets()
    print('MNIST dataset downloaded')