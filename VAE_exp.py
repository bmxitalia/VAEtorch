import torch
from vaetorch.loaders import get_mnist_loaders
from vaetorch.nets import VAEnet
from vaetorch.models import VAE
from vaetorch.utils import create_latent_plot, create_reconstructed_images, create_generated_images, generate_manifold

kl_loss = lambda mu, log_var: -0.5 * torch.sum(1 + log_var - torch.exp(log_var) - torch.square(mu))

train_loader, val_loader = get_mnist_loaders(batch_size=512)
VAE_net_2lat = VAEnet([784, 512, 256], 2)
VAE_model_2lat =VAE(VAE_net_2lat, torch.nn.MSELoss(reduction='sum'), kl_loss, torch.optim.Adam(VAE_net_2lat.parameters(), lr=0.001),
                   "models/VAE_MNIST_2latent.pth")
VAE_model_2lat.load_model()
VAE_net_10lat = VAEnet([784, 512, 256], 10)
VAE_model_10lat = VAE(VAE_net_10lat, torch.nn.MSELoss(reduction='sum'), kl_loss, torch.optim.Adam(VAE_net_10lat.parameters(), lr=0.001),
                    "models/VAE_MNIST_10latent.pth")
VAE_model_10lat.load_model()

create_latent_plot(VAE_model_2lat, val_loader, "plots/vae_mnist_2latent.png",
                      "Variational Autoencoder latent space - MNIST validation set - 2 latent dimensions")

create_latent_plot(VAE_model_10lat, val_loader, "plots/vae_mnist_10latent.png",
                      "Variational Autoencoder latent space - MNIST validation set - 10 latent dimensions (T-SNE)", tsne=True)

create_reconstructed_images(VAE_model_2lat, val_loader, 10, "plots/vae_mnist_reconstruction_2latent.png",
                      "Variational Autoencoder reconstruction - MNIST validation set - 2 latent dimensions - Input images (top) "
                      "vs. reconstructed images (bottom)")

create_reconstructed_images(VAE_model_10lat, val_loader, 10, "plots/vae_mnist_reconstruction_10latent.png",
                      "Variational Autoencoder reconstruction - MNIST validation set - 10 latent dimensions - Input images (top) "
                      "vs. reconstructed images (bottom)")

create_generated_images(VAE_model_2lat, 9, "plots/vae_mnist_generation_2latent.png",
                      "Variational Autoencoder generation - MNIST - 2 latent dimensions")

create_generated_images(VAE_model_10lat, 9, "plots/vae_mnist_generation_10latent.png",
                      "Variational Autoencoder generation - MNIST - 10 latent dimensions")

generate_manifold(VAE_model_2lat, "plots/vae_mnist_manifold_2latent.png",
                      "Variational Autoencoder manifold - MNIST - 2 latent dimensions")