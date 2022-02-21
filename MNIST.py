import torch
from vaetorch.nets import AEnet
from vaetorch.models import AE

from vaetorch.loaders import get_mnist_loaders
from vaetorch.nets import VAEnet
from vaetorch.models import VAE

# Autoencoder

# 2 latent sizes
train_loader, val_loader = get_mnist_loaders(batch_size=512)
AE_net = AEnet([784, 512, 256, 2])
AE_model = AE(AE_net, torch.nn.MSELoss(), torch.optim.Adam(AE_net.parameters(), lr=0.001),
              "models/AE_MNIST_2latent.pth", "plots/MNIST/AE/2lat")
AE_model.train(train_loader, val_loader, 100, 5)

# 10 latent sizes
AE_net = AEnet([784, 512, 256, 10])
AE_model = AE(AE_net, torch.nn.MSELoss(), torch.optim.Adam(AE_net.parameters(), lr=0.001),
              "models/AE_MNIST_10latent.pth", "plots/MNIST/AE/10lat")
AE_model.train(train_loader, val_loader, 100, 5)

# Variational Autoencoder

kl_loss = lambda mu, log_var: -0.5 * torch.sum(1 + log_var - torch.exp(log_var) - torch.square(mu))

# 2 latent dims
VAE_net = VAEnet([784, 512, 256], 2)
VAE_model = VAE(VAE_net, torch.nn.MSELoss(reduction='sum'), kl_loss, torch.optim.Adam(VAE_net.parameters(), lr=0.001),
                "models/VAE_MNIST_2latent.pth", "plots/VAE/2lat")

VAE_model.train(train_loader, val_loader, 100, 5)

# 10 latent dims
VAE_net = VAEnet([784, 512, 256], 10)
VAE_model = VAE(VAE_net, torch.nn.MSELoss(reduction='sum'), kl_loss, torch.optim.Adam(VAE_net.parameters(), lr=0.001),
                "models/VAE_MNIST_10latent.pth", "plots/VAE/10lat")

VAE_model.train(train_loader, val_loader, 100, 5)