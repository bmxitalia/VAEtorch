import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from vaetorch.loaders import get_celeba_filtered
from vaetorch.models import VAE
from vaetorch.nets import CNNVAEnet


def get_mean_vector_feature(dataset, n_images=1000):
    mean = []
    n_images = min([n_images, len(dataset)])
    for i in tqdm(range(n_images)):
        mean.append(VAE_model.get_latent(dataset[i][0].reshape(1, 3, 128, 128)))

    mean = torch.stack(mean)
    mean = mean.reshape(n_images, -1)
    return mean.mean(dim=0)


if __name__ == '__main__':
    def kl_loss(mu, log_var): return -0.5 * torch.sum(1 + log_var - torch.exp(log_var) - torch.square(mu))


    VAE_net = CNNVAEnet([3, 64, 128, 256, 512], 200, 128)

    VAE_net.load_state_dict(torch.load("models/VAE_celeba.pth", map_location=torch.device('cpu'))['model_state_dict'])
    VAE_net.eval()

    optim = torch.optim.Adam(VAE_net.parameters(), lr=0.001)
    optim.load_state_dict(torch.load("models/VAE_celeba.pth", map_location=torch.device('cpu'))['optimizer_state_dict'])

    VAE_model = VAE(VAE_net, torch.nn.MSELoss(reduction='sum'), kl_loss,
                    optim, "models/VAE_celeba.pth", flat_input=False, save_img_path="imgs")

    with_feature = get_celeba_filtered(15, feature_present=True)
    without_feature = get_celeba_filtered(15, feature_present=False)

    mean_with_feature = get_mean_vector_feature(with_feature, 2500)
    mean_without_feature = get_mean_vector_feature(without_feature, 2500)

    feature_vector = mean_with_feature - mean_without_feature

    x = VAE_net.dec(feature_vector.reshape(1, -1))

    plt.imshow(x.detach()[0].permute(1, 2, 0))
    plt.xticks([])
    plt.yticks([])
    plt.show()
