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


def get_celeba_loaders(images_size, batch_size):
    training_data = datasets.CelebA(
        root="datasets",
        split="train",
        download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(178),
            transforms.Resize(images_size),
            transforms.ToTensor()])
    )

    test_data = datasets.CelebA(
        root="datasets",
        split="test",
        download=False,
        transform=transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(images_size),
            transforms.ToTensor()])
    )

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
