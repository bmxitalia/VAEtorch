import torch
from vaetorch.loaders import get_mnist_loaders
from vaetorch.nets import AEnet
from vaetorch.models import AE

train_loader, val_loader = get_mnist_loaders(batch_size=512)
AE_net = AEnet([784, 512, 256, 10])
AE_model = AE(AE_net, torch.nn.MSELoss(), torch.optim.Adam(AE_net.parameters(), lr=0.001), "models/AE_MNIST_10latent.pth")

AE_model.train(train_loader, val_loader, 100, 5)