from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(batch_size):
    training_data = datasets.MNIST(
        root="datasets",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    val_data = datasets.MNIST(
        root="datasets",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader