import torch
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


features = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips",
            "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
            "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open",
            "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
            "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat",
            "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]


def get_celeba_filtered(feature_idx, feature_present=True, images_size=128):
    all_data = datasets.CelebA(
            root="/Users/alessio/torch-dataset",
            split="all",
            download=False,
            transform=transforms.Compose([
                    transforms.CenterCrop(178),
                    transforms.Resize(images_size),
                    transforms.ToTensor()
            ])
    )

    init_indexes = all_data.attr.T[feature_idx]

    if not feature_present:
        init_indexes = (init_indexes + 1) % 2

    indexes = init_indexes.nonzero(as_tuple=True)[0]

    filtered_dataset = torch.utils.data.Subset(all_data, indexes)

    return filtered_dataset
