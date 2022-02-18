import torch
from vaetorch.loaders import get_mnist_loaders
from vaetorch.nets import AEnet
from vaetorch.models import AE
from vaetorch.utils import create_latent_plot, create_reconstructed_images, create_generated_images, generate_manifold

train_loader, val_loader = get_mnist_loaders(batch_size=512)
AE_net_2lat = AEnet([784, 512, 256, 2])
AE_model_2lat = AE(AE_net_2lat, torch.nn.MSELoss(), torch.optim.Adam(AE_net_2lat.parameters(), lr=0.001),
                   "models/AE_MNIST_2latent.pth")
AE_model_2lat.load_model()
AE_net_10lat = AEnet([784, 512, 256, 10])
AE_model_10lat = AE(AE_net_10lat, torch.nn.MSELoss(), torch.optim.Adam(AE_net_10lat.parameters(), lr=0.001),
                    "models/AE_MNIST_10latent.pth")
AE_model_10lat.load_model()

create_latent_plot(AE_model_2lat, val_loader, "plots/ae_mnist_2latent.png",
                      "Autoencoder latent space - MNIST validation set - 2 latent dimensions")

create_latent_plot(AE_model_10lat, val_loader, "plots/ae_mnist_10latent.png",
                      "Autoencoder latent space - MNIST validation set - 10 latent dimensions (T-SNE)", tsne=True)

create_reconstructed_images(AE_model_2lat, val_loader, 10, "plots/ae_mnist_reconstruction_2latent.png",
                      "Autoencoder reconstruction - MNIST validation set - 2 latent dimensions - Input images (top) "
                      "vs. reconstructed images (bottom)")

create_reconstructed_images(AE_model_10lat, val_loader, 10, "plots/ae_mnist_reconstruction_10latent.png",
                      "Autoencoder reconstruction - MNIST validation set - 10 latent dimensions - Input images (top) "
                      "vs. reconstructed images (bottom)")

create_generated_images(AE_model_2lat, 9, "plots/ae_mnist_generation_2latent.png",
                      "Autoencoder generation - MNIST - 2 latent dimensions")

create_generated_images(AE_model_10lat, 9, "plots/ae_mnist_generation_10latent.png",
                      "Autoencoder generation - MNIST - 10 latent dimensions")

generate_manifold(AE_model_2lat, "plots/ae_mnist_manifold_2latent.png",
                      "Autoencoder manifold - MNIST - 2 latent dimensions")