import torch

from vaetorch.loaders import get_celeba_loaders
from vaetorch.models import VAE
from vaetorch.nets import CNNVAEnet

if __name__ == '__main__':
    def kl_loss(mu, log_var): return -0.5 * torch.sum(1 + log_var - torch.exp(log_var) - torch.square(mu))

    train_loader, val_loader = get_celeba_loaders(images_size=128, batch_size=512)
    VAE_net = CNNVAEnet([3, 64, 128, 256, 512], 200, 128)
    VAE_model = VAE(VAE_net, torch.nn.MSELoss(reduction='sum'), kl_loss,
                    torch.optim.Adam(VAE_net.parameters(), lr=0.001),
                    "models/VAE_celeba.pth", flat_input=False, save_img_path="imgs")

    VAE_model.train(train_loader, val_loader, 100, 5)
