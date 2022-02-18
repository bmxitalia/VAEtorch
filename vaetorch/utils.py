import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
import numpy as np
from sklearn.manifold import TSNE
import vaetorch


def create_latent_plot(model, loader, path, title, tsne=False):
    # plot latent space of AE on MNIST
    zs_list = []
    y_list = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(loader):
            zs = model.get_latent(X)
            zs_list.append(zs)
            y_list.extend(y)
        zs = torch.vstack(zs_list).cpu()
    plt.figure(figsize=(10, 10))
    plt.title(title)
    if not tsne:
        scatter = plt.scatter(zs[:, 0], zs[:, 1], c=y_list, cmap='tab10')
    else:
        z_embedded = TSNE(n_components=2, init='random').fit_transform(zs.numpy())
        scatter = plt.scatter(z_embedded[:, 0], z_embedded[:, 1], c=y_list, cmap='tab10')
    plt.legend(handles=scatter.legend_elements()[0], labels=range(10))
    plt.savefig(path)


def create_reconstructed_images(model, loader, n_images, path, title):
    with torch.no_grad():
        batch_X, batch_y = next(iter(loader))
        input_images = batch_X[:n_images]
        rec_images = model.predict(input_images).view(-1, 1, 28, 28)
        images = torch.cat([input_images, rec_images], dim=0)
        grid = make_grid(images, nrow=n_images)
        plt.figure(figsize=(15, 5))
        plt.imshow(np.transpose(grid, (1, 2, 0)), interpolation='nearest', cmap='gray')
        plt.title(title)
        plt.savefig(path)


def create_generated_images(model, n_images, path, title):
    with torch.no_grad():
        eps = torch.randn((n_images, model.net.dec_layers[0].in_features))
        gen_images = model.net.dec(eps).view(-1, 1, 28, 28)
        grid = make_grid(gen_images, nrow=3)
        plt.figure(figsize=(10, 10))
        plt.imshow(np.transpose(grid, (1, 2, 0)), interpolation='nearest', cmap='gray')
        plt.title(title)
        plt.savefig(path)


def generate_manifold(model, path, title):
    z1 = np.linspace(-3, 3, 15)
    z2 = np.linspace(-3, 3, 15)
    z_grid = torch.from_numpy(np.dstack(np.meshgrid(z1, z2))).to(vaetorch.device)

    x_pred_grid = model.net.dec(z_grid.view(15 * 15, 2).float()).view(15, 15, 28, 28).cpu().detach().numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(np.block(list(map(list, x_pred_grid))), cmap='gray')
    plt.title(title)
    plt.savefig(path)
