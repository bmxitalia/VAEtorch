import torch
from vaetorch.loaders import get_mnist_loaders
from vaetorch.nets import VAEnet
from vaetorch.models import VAE

kl_loss = lambda mu, log_var: -0.5 * torch.sum(1 + log_var - torch.exp(log_var) - torch.square(mu))

train_loader, val_loader = get_mnist_loaders(batch_size=512)
VAE_net = VAEnet([784, 512, 256], 10)
VAE_model = VAE(VAE_net, torch.nn.MSELoss(reduction='sum'), kl_loss, torch.optim.Adam(VAE_net.parameters(), lr=0.001),
                "models/VAE_MNIST_10latent.pth")

VAE_model.train(train_loader, val_loader, 100, 5)